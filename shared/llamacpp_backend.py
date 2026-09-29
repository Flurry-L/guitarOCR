"""Optional OCR runtime over llama-server's local multimodal HTTP API."""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import atexit
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from shared.score_image import load_policy, normalize_messages


class LlamaCppBackend:
    supports_ragged_batch = True

    def __init__(self, model_path, endpoint=None, *, device='cpu'):
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
            command = [os.environ.get('GUITAROCR_LLAMA_SERVER', 'llama-server'),
                       '-m', str(folder / 'model-Q8_0.gguf'), '--mmproj', str(folder / 'vision-F16.gguf'),
                       '--host', '127.0.0.1', '--port', str(port), '-c', '16384', '-np', '4',
                       '-ngl', os.environ.get('GUITAROCR_LLAMA_GPU_LAYERS', '99' if device.startswith('cuda') else '0')]
            try:
                self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log)
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

    def generate(self, messages, max_new_tokens, *, skip_special_tokens=True):
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
        request = Request(self.endpoint + '/v1/chat/completions', headers={'Content-Type': 'application/json'},
                          data=json.dumps({'messages': payload, 'max_tokens': max_new_tokens,
                                           'temperature': 0, 'cache_prompt': False}).encode())
        try:
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read(4096).decode(errors='replace')
            raise RuntimeError(f'llama-server returned HTTP {error.code}: {detail}') from error
        text = result['choices'][0]['message'].get('content') or ''
        return text, int(result['usage']['completion_tokens'])

    def generate_batch(self, messages, max_new_tokens, *, skip_special_tokens=True):
        # llama-server owns the inference slots and KV memory; preserve input order.
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(messages)))) as pool:
            return list(pool.map(lambda value: self.generate(value, max_new_tokens,
                                                            skip_special_tokens=skip_special_tokens), messages))
