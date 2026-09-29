"""Optional OCR runtime over llama-server's local multimodal HTTP API."""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from shared.score_image import load_policy, normalize_messages


class LlamaCppBackend:
    supports_ragged_batch = True

    def __init__(self, model_path, endpoint):
        self.endpoint = endpoint.rstrip('/')
        self.image_policy = load_policy(model_path)

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
