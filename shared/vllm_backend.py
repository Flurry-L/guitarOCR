"""Run the accelerated engine in its own compatible Python environment."""

import atexit
import json
import os
from pathlib import Path
import subprocess
import threading

from shared.paths import PROJECT_ROOT


def engine_python():
    override = os.environ.get('GUITAROCR_VLLM_PYTHON')
    return Path(override) if override else PROJECT_ROOT / 'tools/vllm-venv/bin/python'


class VllmBackend:
    supports_ragged_batch = True

    def __init__(self, model_path, device='cuda:0', *, options=None):
        self.lock = threading.RLock()
        self.metrics = []
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
            [str(engine_python()), '-m', 'shared.vllm_worker', '--model', str(Path(model_path).resolve()),
             '--options', json.dumps(options or {})],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, cwd=PROJECT_ROOT, env=env,
        )
        atexit.register(self.close)
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

    def generate_batch(self, messages, max_new_tokens, *, skip_special_tokens=True):
        if not messages:
            return []
        with self.lock:
            self.process.stdin.write(json.dumps({'messages': messages, 'max_new_tokens': max_new_tokens,
                                                'skip_special_tokens': skip_special_tokens}) + '\n')
            self.process.stdin.flush()
            result = self._receive()
            self.metrics = result.get('metrics', [])
            return [tuple(v) for v in result['outputs']]

    def generate(self, messages, max_new_tokens, **kwargs):
        return self.generate_batch([messages], max_new_tokens, **kwargs)[0]

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
