"""Optional OCR runtime over llama-server's local multimodal HTTP API."""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
from http.client import RemoteDisconnected
import json
import atexit
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from shared.score_image import load_policy, normalize_messages


class LlamaCppBackend:
    supports_ragged_batch = True
    supports_json_schema = True

    def __init__(self, model_path, endpoint=None, *, device='cpu', options=None):
        self.process = None
        self.log = None
        if not endpoint:
            model_path = Path(model_path)
            folder = model_path.parent / 'gguf'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            endpoint = f'http://127.0.0.1:{port}'
            from shared.paths import PROJECT_ROOT
            logs = PROJECT_ROOT / 'output/runtime'
            logs.mkdir(parents=True, exist_ok=True)
            self.log_path = logs / f'llama-{model_path.parent.name}-{os.getpid()}.log'
            self.log = self.log_path.open('a')
            context = int((options or {}).get('max_model_len', 16384))
            command = [os.environ.get('GUITAROCR_LLAMA_SERVER', 'llama-server'),
                       '-m', str(folder / 'model-Q8_0.gguf'), '--mmproj', str(folder / 'vision-F16.gguf'),
                       '--host', '127.0.0.1', '--port', str(port), '-c', str(context * 4), '-np', '4',
                       '-ngl', os.environ.get('GUITAROCR_LLAMA_GPU_LAYERS', '99' if device.startswith('cuda') else '0')]
            try:
                # The pinned CUDA graph implementation crashes in GLM's
                # vision encoder when successive crops have different shapes.
                # Ordinary CUDA execution retains GPU offload and model quality.
                environment = {**os.environ, 'GGML_CUDA_DISABLE_GRAPHS': '1'}
                self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log, env=environment)
                atexit.register(self.close)
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f'llama-server 启动失败：{self.log_path.read_text(errors="replace")[-3000:]}')
                    try:
                        with urlopen(endpoint + '/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(.2)
                else:
                    raise TimeoutError(f'llama-server 启动超时，请查看 {self.log_path}')
            except BaseException:
                self.close()
                raise
        self.endpoint = endpoint.rstrip('/')
        self.image_policy = load_policy(model_path)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()

    def generate(self, messages, max_new_tokens, *, skip_special_tokens=True, json_schema=None, grammar=None):
        if json_schema is not None and grammar is not None:
            raise ValueError('Choose either a JSON schema or an M2 grammar')
        normalized = normalize_messages(messages, self.image_policy)
        payload = []
        for message in normalized:
            if isinstance(message.get('content'), str):
                payload.append(message)
                continue
            content = []
            for item in message.get('content', []):
                if item.get('type') != 'image':
                    content.append(item)
                    continue
                buffer = io.BytesIO()
                if 'image' in item:
                    item['image'].save(buffer, format='PNG')
                else:
                    with Image.open(item['url']) as source:
                        source.convert('RGB').save(buffer, format='PNG')
                encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
                content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}})
            payload.append({**message, 'content': content})
        body = {'messages': payload, 'max_tokens': max_new_tokens, 'temperature': 0, 'cache_prompt': False}
        if json_schema is not None:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'score_annotation', 'schema': json_schema}}
        if grammar is not None:
            body['grammar'] = grammar
        request = Request(self.endpoint + '/v1/chat/completions', headers={'Content-Type': 'application/json'},
                          data=json.dumps(body).encode())
        try:
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read(4096).decode(errors='replace')
            raise RuntimeError(f'llama-server returned HTTP {error.code}: {detail}') from error
        except (URLError, RemoteDisconnected) as error:
            if self.process is not None and self.process.poll() is not None:
                detail = self.log_path.read_text(errors='replace')[-2500:]
                raise RuntimeError(f'llama-server 已停止，请查看 {self.log_path}\n{detail}') from error
            raise
        text = result['choices'][0]['message'].get('content') or ''
        return text, int(result['usage']['completion_tokens'])

    def generate_batch(self, messages, max_new_tokens, *, skip_special_tokens=True, json_schema=None, grammar=None):
        schemas = json_schema if isinstance(json_schema, list) else [json_schema] * len(messages)
        grammars = grammar if isinstance(grammar, list) else [grammar] * len(messages)
        if len(schemas) != len(messages) or len(grammars) != len(messages):
            raise ValueError('Each prompt needs its own output schema')
        # llama-server owns the inference slots and KV memory; preserve input order.
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(messages)))) as pool:
            return list(pool.map(lambda item: self.generate(item[0], max_new_tokens,
                skip_special_tokens=skip_special_tokens, json_schema=item[1], grammar=item[2]),
                zip(messages, schemas, grammars, strict=True)))
