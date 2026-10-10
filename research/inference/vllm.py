"""vLLM reference evaluation: client and worker protocol in separate Python processes."""

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from research.common.ocr import batch_tasks, load_task_adapter
from research.defaults import PROJECT_ROOT


def engine_python():
    override = os.environ.get('GUITAROCR_VLLM_PYTHON')
    return Path(override) if override else PROJECT_ROOT / 'tools/vllm-venv/bin/python'


class VllmBackend:
    def task_adapter_applied(self, task):
        return self.task_adapter is not None and self.task_adapter.applies(task)

    def __init__(self, model_path, device='cuda:0', *, options=None):
        self.task_adapter = load_task_adapter(model_path, 'peft')
        log_dir = PROJECT_ROOT / 'output/runtime'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f'vllm-{os.getpid()}-{device.replace(":", "-")}.log'
        self.log = self.log_path.open('a')
        env = dict(os.environ, OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn')
        if device.startswith('cuda'):
            ordinal = int(device.partition(':')[2] or 0)
            visible = env.get('CUDA_VISIBLE_DEVICES')
            env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[ordinal] if visible else str(ordinal)
        else:
            raise ValueError('vLLM acceleration requires CUDA')
        self.process = subprocess.Popen(
            [str(engine_python()), '-m', 'research.inference.vllm', '--model', str(Path(model_path).resolve()),
             '--options', json.dumps(options or {})],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, cwd=PROJECT_ROOT, env=env,
        )
        try:
            if not self._receive().get('ready'):
                raise RuntimeError('vLLM did not start')
        except BaseException:
            self.close()
            raise

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            excerpt = self.log_path.read_text(errors='replace')[-6000:]
            raise RuntimeError(f'vLLM stopped. {self.log_path}\n{excerpt}')
        value = json.loads(line)
        if value.get('error'):
            raise RuntimeError(value['error'])
        return value

    def generate_batch(self, messages, max_new_tokens, *, task=None):
        tasks = batch_tasks(task, len(messages))
        if not messages:
            return []
        self.process.stdin.write(json.dumps({'messages': messages,
                                            'max_new_tokens': max_new_tokens, 'task': tasks}) + '\n')
        self.process.stdin.flush()
        return [tuple(v) for v in self._receive()['outputs']]

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.write('{"close":true}\n')
                process.stdin.flush()
                process.wait(timeout=15)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if hasattr(self, 'log'):
            self.log.close()


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

    from research.common.ocr import (
        batch_tasks,
        load_policy,
        load_task_adapter,
        normalize_score_image,
    )

    image_policy = load_policy(args.model)

    options = json.loads(args.options)
    adapter = load_task_adapter(args.model, 'peft')
    if adapter is not None:
        from vllm.lora.request import LoRARequest
        adapter_config = json.loads((adapter.path / 'adapter_config.json').read_text())
        options.update(enable_lora=True, max_loras=1,
                       max_lora_rank=int(adapter_config['r']))
        adapter_request = LoRARequest('information', 1, str(adapter.path))
    else:
        adapter_request = None
    options.setdefault('max_model_len', 8192)
    options.setdefault('max_num_seqs', 32)
    options.setdefault('max_num_batched_tokens', 8192)
    options.setdefault('gpu_memory_utilization', .40)
    options.setdefault('limit_mm_per_prompt', {'image': 3, 'video': 0})
    options.setdefault('mm_processor_cache_gb', 2)
    options.setdefault('enable_prefix_caching', True)
    options.setdefault('disable_log_stats', False)
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
                            key = str(path)
                            stat = path.stat()
                            identity = f'{path}:{stat.st_mtime_ns}:{stat.st_size}'
                            if key not in loaded_images:
                                with Image.open(path) as im:
                                    image = normalize_score_image(im, image_policy) if image_policy else im.convert('RGB')
                                loaded_images[key] = image
                                opened.append(image)
                            image = loaded_images[key]
                            images.append(image)
                            uuids.append(identity)
                value = {'prompt': processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}
                if images:
                    value.update(multi_modal_data={'image': images}, multi_modal_uuids={'image': uuids})
                prompts.append(value)
            params = SamplingParams(temperature=0, max_tokens=request['max_new_tokens'])
            tasks = batch_tasks(request.get('task'), len(prompts))
            loras = [adapter_request if adapter is not None and adapter.applies(task) else None for task in tasks]
            outputs = engine.generate(prompts, params, use_tqdm=False,
                                      **({'lora_request': loras} if adapter_request is not None else {}))
            values = [(r.outputs[0].text, len(r.outputs[0].token_ids)) for r in outputs]
            protocol.write(json.dumps({'outputs': values}) + '\n')
        except Exception:
            protocol.write(json.dumps({'error': traceback.format_exc()}) + '\n')
        finally:
            for image in opened:
                image.close()

if __name__ == '__main__':
    main()
