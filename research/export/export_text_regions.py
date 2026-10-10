"""Attach the explicit GuitarOCR input/decoder contract to an upstream DB graph."""

import argparse
import json
from pathlib import Path

import onnx


def export(source, destination):
    graph = onnx.load(source)
    for old, new in [
        (graph.graph.input[0].name, "image"),
        (graph.graph.output[0].name, "probability"),
    ]:
        for node in graph.graph.node:
            for names in [node.input, node.output]:
                for index, name in enumerate(names):
                    if name == old:
                        names[index] = new
        for value in [*graph.graph.input, *graph.graph.output, *graph.graph.value_info]:
            if value.name == old:
                value.name = new
    used = {name for node in graph.graph.node for name in node.input}
    retained = [value for value in graph.graph.initializer if value.name in used]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(retained)
    config = {
        "version": 1,
        "source": "PaddlePaddle/PP-OCRv5_mobile_det",
        "resize": "contain_lanczos_stride32_white",
        "normalization": "bgr_imagenet",
        "decoder": "probability_components_rect_v1",
        "maximum_side": 960,
        "pixel_threshold": 0.3,
        "box_threshold": 0.6,
        "unclip_ratio": 1.5,
        "minimum_component_side": 3,
        "max_candidates": 1000,
    }
    onnx.helper.set_model_props(graph, {"guitarocr.text_regions": json.dumps(config)})
    onnx.checker.check_model(graph)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, destination)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))
