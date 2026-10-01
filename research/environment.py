"""Inspect the research environment without installing or downloading anything."""
from importlib import metadata, util
from pathlib import Path
import argparse
import json
from research.defaults import MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    packages = {'scorelib': 'scorelib', 'gpbridge': 'gpbridge', 'numpy': 'numpy',
                'PIL': 'Pillow', 'guitarpro': 'PyGuitarPro', 'pypdfium2': 'pypdfium2'}
    if args.training:
        packages.update(torch='torch', transformers='transformers', peft='peft', accelerate='accelerate')
    checks = {}
    for module, distribution in packages.items():
        try:
            checks[distribution] = metadata.version(distribution) if util.find_spec(module) else 'missing'
        except (ImportError, metadata.PackageNotFoundError):
            checks[distribution] = 'missing'
    if checks['scorelib'] != 'missing':
        from scorelib import parse_measure_target
        parse_measure_target('M2 | V0{@0:w:r}')
    checks['model'] = str(args.model.resolve()) if (args.model / 'config.json').is_file() else 'missing'
    if args.training:
        import torch
        value = torch.ones((8, 8), device=args.device)
        (value @ value).sum().item()
        checks['device'] = torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else args.device
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return int('missing' in checks.values())


if __name__ == '__main__':
    raise SystemExit(main())
