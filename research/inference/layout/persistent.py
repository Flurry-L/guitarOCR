"""Keep the page detector resident in its Paddle environment."""

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock

from research.defaults import LAYOUT_MODEL
from research.defaults import paddle_python
from research.paths import PROJECT_ROOT


class LayoutBackend:
    def __init__(self, model=None, python=None, device='cuda'):
        self.model = Path(model or LAYOUT_MODEL).resolve()
        candidate = paddle_python()
        self.python = Path(python or (candidate if candidate.is_file() else sys.executable)).absolute()
        self.device = device
        self.process = None
        self.onnx = None
        self.lock = RLock()
        self.log = None
        atexit.register(self.close)

    def __call__(self, pages):
        with self.lock:
            auxiliary = os.environ.get('GUITAROCR_AUX_MODELS')
            if auxiliary:
                from research.inference.layout.onnx_detector import OnnxDetector
                from research.inference.layout.detector import detect_pages
                if self.onnx is None:
                    self.onnx = OnnxDetector(Path(auxiliary) / 'layout.onnx')
                return detect_pages([str(p) for p in pages], self.model, .25, True, model=self.onnx)
            if self.process is None or self.process.poll() is not None:
                if self.log is not None:
                    self.log.close()
                folder = PROJECT_ROOT / 'output/runtime'
                folder.mkdir(parents=True, exist_ok=True)
                self.log_path = folder / f'layout-{os.getpid()}.log'
                self.log = self.log_path.open('a')
                env = {**os.environ, 'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK': 'True'}
                if self.device.startswith('cuda'):
                    ordinal = int(self.device.partition(':')[2] or 0)
                    visible = env.get('CUDA_VISIBLE_DEVICES')
                    env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[ordinal] if visible else str(ordinal)
                else:
                    env['CUDA_VISIBLE_DEVICES'] = ''
                self.process = subprocess.Popen(
                    [str(self.python), '-m', 'research.inference.layout.persistent', '--model', str(self.model)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                    text=True, bufsize=1, cwd=PROJECT_ROOT, env=env,
                )
            self.process.stdin.write(json.dumps([str(p) for p in pages]) + '\n')
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f'Layout engine stopped: {self.log_path}\n' + self.log_path.read_text(errors='replace')[-4000:])
            result = json.loads(line)
            if isinstance(result, dict) and 'error' in result:
                raise RuntimeError(result['error'])
            return result

    def close(self):
        process = self.process
        if process is not None and process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if self.log is not None:
            self.log.close()


def main():
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import paddle
    from paddlex import create_model
    from research.inference.layout.detector import detect_pages

    device = 'gpu:0' if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() else 'cpu'
    model = create_model(model_name='PP-DocLayoutV3', model_dir=str(args.model), device=device)
    for line in sys.stdin:
        try:
            result = detect_pages(json.loads(line), args.model, .25, True, model=model)
        except Exception:
            result = {'error': traceback.format_exc()}
        protocol.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
