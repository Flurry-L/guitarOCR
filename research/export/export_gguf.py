"""Export the unified OCR model for llama.cpp, retaining the music vocabulary."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LLAMA_CPP_REVISION = '8019dc563b1ecbae6b161a70c3a1359f1b206c1e'


def strip_unused_mtp(path, llama_cpp):
    """Copy active tensors byte-for-byte, removing the unused NextN block."""
    sys.path.insert(0, str(llama_cpp / 'gguf-py'))
    import gguf

    reader = gguf.GGUFReader(path)
    arch = reader.get_field('general.architecture').contents()
    count = reader.get_field(f'{arch}.block_count').contents()
    nextn = reader.get_field(f'{arch}.nextn_predict_layers')
    extra = nextn.contents() if nextn else 0
    if not extra:
        return
    active = count - extra
    temporary = path.with_suffix('.slim.gguf')
    writer = gguf.GGUFWriter(temporary, arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == 'general.architecture' or field.name.startswith('GGUF.'):
            continue
        value = field.contents()
        if field.name == f'{arch}.block_count':
            value = active
        elif field.name == f'{arch}.nextn_predict_layers':
            value = 0
        subtype = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, field.types[0], sub_type=subtype)
    tensors = [t for t in reader.tensors
               if not (t.name.startswith('blk.') and int(t.name.split('.')[1]) >= active)]
    for tensor in tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    del tensors, reader
    temporary.replace(path)


def export(llama_cpp, output, tasks, weights_root=ROOT / 'weights'):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=llama_cpp, text=True).strip()
    if revision != LLAMA_CPP_REVISION:
        raise ValueError(f'Use the verified llama.cpp revision: git checkout {LLAMA_CPP_REVISION}')
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source_dirty = bool(subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT, text=True).strip())
    for task in tasks:
        source = weights_root / task / 'merged'
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        for vision, name, dtype in [(False, 'model-Q8_0.gguf', 'q8_0'), (True, 'vision-F16.gguf', 'f16')]:
            subprocess.run([sys.executable, str(llama_cpp / 'convert_hf_to_gguf.py'), str(source),
                            '--outfile', str(folder / name), '--outtype', dtype,
                            *(['--mmproj'] if vision else [])], check=True)
        strip_unused_mtp(folder / 'model-Q8_0.gguf', llama_cpp)
        metadata = {'task': task, 'source_commit': source_commit, 'source_dirty': source_dirty,
                    'llama_cpp_commit': revision,
                    'text_quantization': 'Q8_0', 'vision_dtype': 'F16',
                    'mtp_runtime': False, 'scope': 'OCR only; layout and signature classifier are separate'}
        (folder / 'export.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for path in folder.glob('*.gguf'):
            if path.stat().st_size >= 2**31:
                raise ValueError(f'{path} exceeds the GitHub release asset limit')
            print(f'{path}: {path.stat().st_size / 1e9:.2f} GB', flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-cpp', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/gguf')
    parser.add_argument('--weights-root', type=Path, default=ROOT / 'weights',
                        help='Model directory; allows export before installing staged weights')
    parser.add_argument('--tasks', nargs='+', choices=('score_ocr', 'measure_ocr', 'document_info'),
                        default=['score_ocr'])
    args = parser.parse_args()
    export(args.llama_cpp.resolve(), args.output.resolve(), args.tasks, args.weights_root.resolve())
