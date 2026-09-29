"""Export inference-only auxiliary networks; conversion dependencies stay on the build machine."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def strip_source_paths(message):
    for field, value in message.ListFields():
        if field.name == 'doc_string':
            message.ClearField(field.name)
        elif field.message_type is not None:
            for child in value if field.is_repeated else [value]:
                strip_source_paths(child)


def export(output, paddle_python):
    import torch
    from safetensors.torch import load_file
    from measure_ocr.state_network import SignatureNetwork

    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(paddle_python.parent / ('paddle2onnx.exe' if sys.platform == 'win32' else 'paddle2onnx')),
                    '--model_dir', str(ROOT / 'weights/layout'), '--model_filename', 'inference.json',
                    '--params_filename', 'inference.pdiparams', '--save_file', str(output / 'layout.onnx'),
                    '--opset_version', '17', '--optimize_tool', 'None'], check=True)
    # The product consumes rectangular boxes; discard unused segmentation outputs.
    import onnx
    path = output / 'layout.onnx'
    onnx.utils.extract_model(str(path), str(path), ['im_shape', 'image', 'scale_factor'], ['fetch_name_0'])
    model = SignatureNetwork().eval()
    model.load_state_dict(load_file(str(ROOT / 'weights/measure_ocr/merged/state_reader/state.safetensors')))
    # At the fixed 192 x 512 input, the feature map is exactly 6 x 16.
    model.features[-1] = torch.nn.AvgPool2d((6, 4))
    torch.set_num_threads(4)
    torch.onnx.export(model, torch.zeros(1, 2, 3, 192, 512), str(output / 'signature.onnx'),
                      input_names=['images'], output_names=['key', 'numerator', 'denominator'],
                      dynamic_axes={name: {0: 'batch'} for name in ('images', 'key', 'numerator', 'denominator')},
                      opset_version=17, dynamo=False)
    # Exported debug strings contain local source paths. Keep deployment files
    # reproducible across the developer machine and release runners.
    for name in ('layout', 'signature'):
        path = output / f'{name}.onnx'
        model = onnx.load(path)
        strip_source_paths(model)
        model.producer_name = 'guitarocr'
        model.producer_version = '0.1'
        onnx.save(model, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/distribution/auxiliary')
    parser.add_argument('--paddle-python', type=Path, required=True)
    args = parser.parse_args()
    export(args.output.resolve(), args.paddle_python.absolute())
