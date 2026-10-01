"""Persistent GLM-OCR engine; JSON lines over stdin/stdout, diagnostics on stderr."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from functools import lru_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--options', default='{}')
    args = parser.parse_args()
    # Native libraries also print to fd 1. Give the protocol its own descriptor.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from research.common.score_image import load_policy
    from research.common.score_image import normalize_score_image

    @lru_cache(maxsize=128)
    def bounded_json_grammar(schema):
        import xgrammar

        # Unlimited JSON whitespace can consume the entire output budget
        # when a model tries to stop before all required rows are present.
        return str(xgrammar.Grammar.from_json_schema(schema, max_whitespace_cnt=1))

    image_policy = load_policy(args.model)

    options = json.loads(args.options)
    speculative_tokens = options.pop('speculative_tokens', 0)
    options.setdefault('max_model_len', 8192)
    options.setdefault('max_num_seqs', 32)
    options.setdefault('max_num_batched_tokens', 8192)
    options.setdefault('gpu_memory_utilization', .40)
    options.setdefault('limit_mm_per_prompt', {'image': 3, 'video': 0})
    options.setdefault('mm_processor_cache_gb', 2)
    options.setdefault('enable_prefix_caching', True)
    options.setdefault('disable_log_stats', False)
    if speculative_tokens:
        options['speculative_config'] = {
            'method': 'mtp', 'num_speculative_tokens': speculative_tokens,
            **options.get('speculative_config', {}),
        }
    processor = AutoProcessor.from_pretrained(args.model)
    engine = LLM(model=args.model, dtype='bfloat16', trust_remote_code=True, **options)
    protocol.write(json.dumps({'ready': True}) + '\n')
    for line in sys.stdin:
        opened = []
        try:
            request = json.loads(line)
            if request.get('close'):
                break
            prompts = []
            loaded_images = {}
            for messages in request['messages']:
                images, uuids = [], []
                for message in messages:
                    content = message.get('content', [])
                    if isinstance(content, str):
                        continue
                    for item in content:
                        if item['type'] == 'image':
                            path = Path(item['url']).resolve()
                            if path not in loaded_images:
                                with Image.open(path) as im:
                                    image = normalize_score_image(im, image_policy) if image_policy else im.convert('RGB')
                                loaded_images[path] = image
                                opened.append(image)
                            image = loaded_images[path]
                            images.append(image)
                            stat = path.stat()
                            uuids.append(f'{path}:{stat.st_mtime_ns}:{stat.st_size}')
                value = {'prompt': processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}
                if images:
                    value.update(multi_modal_data={'image': images}, multi_modal_uuids={'image': uuids})
                prompts.append(value)
            def sampling(schema, grammar):
                if schema is not None and grammar is not None:
                    raise ValueError('Choose either a JSON schema or an M2 grammar')
                return SamplingParams(temperature=0, max_tokens=request['max_new_tokens'],
                                      skip_special_tokens=request.get('skip_special_tokens', True),
                                      structured_outputs=StructuredOutputsParams(grammar=bounded_json_grammar(json.dumps(schema))) if schema else
                                      StructuredOutputsParams(grammar=grammar) if grammar else None)

            schema = request.get('json_schema')
            grammar = request.get('grammar')
            if isinstance(schema, list) or isinstance(grammar, list):
                schemas = schema if isinstance(schema, list) else [schema] * len(prompts)
                grammars = grammar if isinstance(grammar, list) else [grammar] * len(prompts)
                if len(schemas) != len(prompts) or len(grammars) != len(prompts):
                    raise ValueError('Each prompt needs its own output schema')
                params = [sampling(s, g) for s, g in zip(schemas, grammars, strict=True)]
            else:
                params = sampling(schema, grammar)
            outputs = engine.generate(prompts, params, use_tqdm=False)
            values = [(r.outputs[0].text, len(r.outputs[0].token_ids)) for r in outputs]
            metrics = []
            if hasattr(engine, 'get_metrics'):
                for metric in engine.get_metrics():
                    if 'spec_decode' in metric.name:
                        metrics.append({'name': metric.name, 'value': getattr(metric, 'value', None),
                                        'values': getattr(metric, 'values', None)})
            protocol.write(json.dumps({'outputs': values, 'metrics': metrics}) + '\n')
        except Exception:
            protocol.write(json.dumps({'error': traceback.format_exc()}) + '\n')
        finally:
            for image in opened:
                image.close()


if __name__ == '__main__':
    main()
