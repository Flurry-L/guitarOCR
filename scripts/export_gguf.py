"""Export both trained OCR models for llama.cpp, retaining the music vocabulary."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_REVISION = '8019dc563b1ecbae6b161a70c3a1359f1b206c1e'


def export(llama_cpp, output, tasks):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=llama_cpp, text=True).strip()
    if revision != LLAMA_CPP_REVISION:
        raise ValueError(f'Use the verified llama.cpp revision: git checkout {LLAMA_CPP_REVISION}')
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for task in tasks:
        source = ROOT / 'weights' / task / 'merged'
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        for vision, name, dtype in [(False, 'model-Q8_0.gguf', 'q8_0'), (True, 'vision-F16.gguf', 'f16')]:
            subprocess.run([sys.executable, str(llama_cpp / 'convert_hf_to_gguf.py'), str(source),
                            '--outfile', str(folder / name), '--outtype', dtype,
                            *(['--mmproj'] if vision else [])], check=True)
        metadata = {'task': task, 'source_commit': source_commit, 'llama_cpp_commit': revision,
                    'text_quantization': 'Q8_0', 'vision_dtype': 'F16',
                    'mtp_runtime': False, 'scope': 'OCR only; layout and signature classifier are separate'}
        (folder / 'export.json').write_text(json.dumps(metadata, indent=2) + '\n')
        destination = output / f'GuitarOCR-0.1-{task}-GGUF.zip'
        with ZipFile(destination, 'w', ZIP_STORED) as archive:
            for name in ('model-Q8_0.gguf', 'vision-F16.gguf', 'export.json'):
                archive.write(folder / name, f'{task}/{name}')
            for name in ('score_image_policy.json', 'music_vocabulary.json'):
                if (source / name).is_file():
                    archive.write(source / name, f'{task}/{name}')
            for path in (ROOT / 'weights/licenses').rglob('*'):
                if path.is_file():
                    archive.write(path, 'licenses/' + str(path.relative_to(ROOT / 'weights/licenses')))
            archive.write(ROOT / 'THIRD_PARTY_NOTICES.md', 'THIRD_PARTY_NOTICES.md')
            archive.writestr('README.txt',
                            'GuitarOCR 0.1 optional OCR models for llama.cpp.\n'
                            'Load model-Q8_0.gguf together with its matching vision-F16.gguf.\n'
                            'Keep the two tasks separate: their visual encoders are different.\n'
                            'This is not an Android/iOS application or a full PDF recognition pipeline.\n'
                            'MTP parameters are retained but unused by this llama.cpp runtime.\n'
                            'Setup, image preprocessing and platform validation status:\n'
                            'https://github.com/Flurry-L/guitarOCR/blob/main/docs/setup.md#gguf-可选后端\n')
        if destination.stat().st_size >= 2**31:
            raise ValueError(f'{destination} exceeds the GitHub release asset limit')
        print(f'{destination}: {destination.stat().st_size / 1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-cpp', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/gguf')
    parser.add_argument('--tasks', nargs='+', choices=('measure_ocr', 'document_info'),
                        default=['measure_ocr', 'document_info'])
    args = parser.parse_args()
    export(args.llama_cpp.resolve(), args.output.resolve(), args.tasks)
