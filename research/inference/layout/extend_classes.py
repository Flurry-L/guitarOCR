"""Keep existing detector classes when appending labels for fine-tuning."""

import argparse
from pathlib import Path


def extend(source: Path, output: Path, classes: int):
    import paddle

    paddle.set_device("cpu")
    paddle.seed(20260927)
    state = paddle.load(str(source))
    weight = state["transformer.score_head.weight"]
    old = weight.shape[1]
    if classes <= old:
        raise ValueError("The new category list must append to the existing labels")
    for name, axis in (("transformer.score_head.weight", 1),
                       ("transformer.score_head.bias", 0),
                       ("transformer.denoising_class_embed.weight", 0)):
        value = state[name]
        if value.shape[axis] != old:
            raise ValueError(f"Unexpected class axis: {name}")
        shape = list(value.shape)
        shape[axis] = classes - old
        added = (paddle.full(shape, -4.59511985, dtype=value.dtype) if name.endswith("bias")
                 else paddle.randn(shape, dtype=value.dtype) * 0.02)
        state[name] = paddle.concat([value, added], axis=axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    paddle.save(state, str(output))
    print({"old_classes": old, "new_classes": classes, "checkpoint": str(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    extend(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
