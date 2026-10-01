"""Export inference-only auxiliary networks; conversion dependencies stay on the build machine."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def strip_source_paths(message):
    for field, value in message.ListFields():
        if field.name == 'doc_string':
            message.ClearField(field.name)
        elif field.message_type is not None:
            for child in value if field.is_repeated else [value]:
                strip_source_paths(child)


def export(output, paddle_python, layout_model=None, score_model=None, only=None):
    import onnx
    from research.defaults import LAYOUT_MODEL
    from research.defaults import MEASURE_ADAPTER

    layout_model = layout_model or LAYOUT_MODEL
    score_model = score_model or MEASURE_ADAPTER / 'merged'

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
        from research.models.state_network import SignatureNetwork

        model = SignatureNetwork().eval()
        model.load_state_dict(load_file(str(score_model / 'state_reader/state.safetensors')))
        # At the fixed 192 x 512 input, the feature map is exactly 6 x 16.
        model.features[-1] = torch.nn.AvgPool2d((6, 4))
        torch.set_num_threads(4)
        torch.onnx.export(model, torch.zeros(1, 2, 3, 192, 512), str(output / 'signature.onnx'),
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
        onnx.save(model, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/distribution/auxiliary')
    parser.add_argument('--paddle-python', type=Path, required=True)
    parser.add_argument('--layout-model', type=Path)
    parser.add_argument('--score-model', type=Path)
    parser.add_argument('--only', choices=['layout', 'signature'])
    args = parser.parse_args()
    export(args.output.resolve(), args.paddle_python.absolute(), args.layout_model, args.score_model, args.only)
