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
import secrets
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from research.common.score_image import load_policy
from research.common.score_image import normalize_messages


class LlamaCppBackend:
    supports_ragged_batch = True
    supports_json_schema = True

    def __init__(self, model_path, endpoint=None, *, device='cpu', options=None):
        self.process = None
        self.log = None
        self._api_key = None
        options = options or {}
        self.parallel = int(os.environ.get('GUITAROCR_LLAMA_PARALLEL', options.get('parallel', 1)))
        self.threads = int(os.environ.get('GUITAROCR_LLAMA_THREADS', options.get('threads', min(4, os.cpu_count() or 1))))
        self.context = int(os.environ.get('GUITAROCR_LLAMA_CONTEXT', options.get('context', 8192)))
        if self.threads < 1:
            raise ValueError('llama.cpp threads 必须大于 0')
        if self.context < 2048:
            raise ValueError('llama.cpp context 至少需要 2048')
        if not 1 <= self.parallel <= 4:
            raise ValueError('llama.cpp parallel 必须为 1–4')
        self.request_timeout = float(os.environ.get('GUITAROCR_LLAMA_TIMEOUT', '1800' if device == 'cpu' else '600'))
        if self.request_timeout <= 0:
            raise ValueError('llama.cpp timeout 必须大于 0')
        if not endpoint:
            model_path = Path(model_path)
            folder = model_path.parent / 'gguf'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            endpoint = f'http://127.0.0.1:{port}'
            self._api_key = secrets.token_urlsafe(32)
            from research.paths import PROJECT_ROOT
            logs = PROJECT_ROOT / 'output/runtime'
            logs.mkdir(parents=True, exist_ok=True)
            self.log_path = logs / f'llama-{model_path.parent.name}-{os.getpid()}.log'
            self.log = self.log_path.open('a')
            command = [os.environ.get('GUITAROCR_LLAMA_SERVER', 'llama-server'),
                       '--log-verbosity', '3', '--api-key', self._api_key, '--no-webui',
                       '--cors-origins', endpoint, '--no-cors-credentials',
                       '-m', str(folder / 'model-Q8_0.gguf'), '--mmproj', str(folder / 'vision-F16.gguf'),
                       '--host', '127.0.0.1', '--port', str(port), '-c', str(self.context * self.parallel), '-np', str(self.parallel), '-t', str(self.threads),
                       '-ngl', os.environ.get('GUITAROCR_LLAMA_GPU_LAYERS', '99' if device.startswith('cuda') or device == 'metal' else '0')]
            try:
                # The pinned CUDA graph implementation crashes in GLM's
                # vision encoder when successive crops have different shapes.
                # Ordinary CUDA execution retains GPU offload and model quality.
                environment = {key: value for key, value in os.environ.items()
                               if key not in {'LLAMA_API_KEY', 'LLAMA_API_KEY_FILE', 'LLAMA_ARG_API_KEY_FILE', 'LLAMA_ARG_LOG_VERBOSITY', 'LLAMA_LOG_VERBOSITY'}}
                environment['GGML_CUDA_DISABLE_GRAPHS'] = '1'
                self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log, env=environment)
                atexit.register(self.close)
                deadline = time.monotonic() + float(os.environ.get('GUITAROCR_LLAMA_START_TIMEOUT', '600'))
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f'llama-server 启动失败：{self.log_path.read_text(errors="replace")[-3000:]}')
                    try:
                        with urlopen(Request(endpoint + '/health', headers=self._auth_headers()), timeout=2) as response:
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

    def _auth_headers(self):
        return {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}

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
        self._api_key = None
        # Popen retains argv after exit; do not retain the ephemeral credential.
        if self.process is not None and isinstance(self.process.args, (list, tuple)):
            args = list(self.process.args)
            if '--api-key' in args:
                args[args.index('--api-key') + 1] = '<redacted>'
                self.process.args = args

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
        request = Request(self.endpoint + '/v1/chat/completions', headers={'Content-Type': 'application/json', **self._auth_headers()},
                          data=json.dumps(body).encode())
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
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
        with ThreadPoolExecutor(max_workers=min(self.parallel, max(1, len(messages)))) as pool:
            return list(pool.map(lambda item: self.generate(item[0], max_new_tokens,
                skip_special_tokens=skip_special_tokens, json_schema=item[1], grammar=item[2]),
                zip(messages, schemas, grammars, strict=True)))
