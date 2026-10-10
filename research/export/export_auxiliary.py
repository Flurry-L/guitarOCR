"""Export inference-only auxiliary networks; conversion dependencies stay on the build machine."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def strip_source_paths(message):
    for field, value in message.ListFields():
        if field.name == 'doc_string':
            message.ClearField(field.name)
        elif field.message_type is not None:
            for child in value if field.is_repeated else [value]:
                strip_source_paths(child)


def export(output, paddle_python, layout_model=None, signature_model=None, only=None):
    import onnx

    from research.defaults import LAYOUT_MODEL, WEIGHTS_ROOT

    layout_model = layout_model or LAYOUT_MODEL
    signature_model = signature_model or WEIGHTS_ROOT / 'checkpoints/signature'

    output.mkdir(parents=True, exist_ok=True)
    if only != 'signature':
        subprocess.run([str(paddle_python.parent / ('paddle2onnx.exe' if sys.platform == 'win32' else 'paddle2onnx')),
                        '--model_dir', str(layout_model), '--model_filename', 'inference.json',
                        '--params_filename', 'inference.pdiparams', '--save_file', str(output / 'layout.onnx'),
                        '--opset_version', '17', '--optimize_tool', 'None'], check=True)
        # The product consumes rectangular boxes; discard unused segmentation outputs.
        path = output / 'layout.onnx'
        onnx.utils.extract_model(str(path), str(path), ['im_shape', 'image', 'scale_factor'], ['fetch_name_0'])
    if only != 'layout':
        import torch
        from safetensors.torch import load_file

        from research.common.signature_image import signature_config
        from research.models.auxiliary import make_signature_network

        state_root = signature_model
        config = signature_config(state_root)
        model = make_signature_network(config).eval()
        model.load_state_dict(load_file(str(state_root / 'state.safetensors')))
        shape = config['input']
        sample = torch.zeros(1, 3, shape['height'], shape['width'])
        torch.set_num_threads(4)
        torch.onnx.export(model, sample, str(output / 'signature.onnx'),
                          input_names=['images'], output_names=['key', 'numerator', 'denominator'],
                          dynamic_axes={name: {0: 'batch'} for name in ('images', 'key', 'numerator', 'denominator')},
                          opset_version=17, dynamo=False)
    # Exported debug strings contain local source paths. Keep deployment files
    # reproducible across the developer machine and release runners.
    for name in ([only] if only else ['layout', 'signature']):
        path = output / f'{name}.onnx'
        model = onnx.load(path)
        strip_source_paths(model)
        model.producer_name = 'guitarocr'
        model.producer_version = '0.1'
        if name == 'layout':
            from research.common.images import (
                METADATA_KEY,
                SIZE_METADATA_KEY,
                paddle_preprocessing,
            )

            metadata = {entry.key: entry.value for entry in model.metadata_props}
            contract, input_size = paddle_preprocessing(layout_model)
            metadata[METADATA_KEY] = contract
            metadata[SIZE_METADATA_KEY] = str(input_size)
            onnx.helper.set_model_props(model, metadata)
        if name == 'signature':
            metadata = {entry.key: entry.value for entry in model.metadata_props}
            metadata['guitarocr.signature_config'] = json.dumps(config, separators=(',', ':'))
            calibration_path = state_root / 'calibration.json'
            if calibration_path.is_file():
                metadata['guitarocr.signature_calibration'] = json.dumps(json.loads(calibration_path.read_text()), separators=(',', ':'))
            onnx.helper.set_model_props(model, metadata)
        onnx.save(model, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/distribution/auxiliary')
    parser.add_argument('--paddle-python', type=Path, required=True)
    parser.add_argument('--layout-model', type=Path)
    parser.add_argument('--signature-model', type=Path,
                        help='Signature training directory containing state.safetensors and config.json')
    parser.add_argument('--only', choices=['layout', 'signature'])
    args = parser.parse_args()
    export(args.output.resolve(), args.paddle_python.absolute(), args.layout_model, args.signature_model, args.only)
