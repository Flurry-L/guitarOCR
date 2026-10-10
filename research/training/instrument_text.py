"""Train and calibrate literal-name classification without document-specific rules."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from research.models.auxiliary import InstrumentTextNetwork
from research.training import fit_classifier

LABELS = ["unknown", "guitar", "bass", "pitched", "drums"]


LENGTH = 160


NEGATIVES = [
    "Intro",
    "Verse",
    "Chorus",
    "Bridge",
    "Outro",
    "Refrain",
    "Coda",
    "Segno",
    "Solo",
    "Lead",
    "Rhythm",
    "Melody",
    "Part",
    "Track",
    "Voice",
    "Standard Tuning",
    "Drop D",
    "Capo",
    "TAB",
    "Moderato",
    "Andante",
    "Allegro",
    "Adagio",
    "Largo",
    "sim.",
    "let ring",
    "P.M.",
    "palm mute",
    "H",
    "P",
    "mf",
    "ff",
    "mp",
    "pp",
    "D.S. al Coda",
    "D.C. al Fine",
    "A7",
    "Cmaj7",
    "G#m7",
    "N.C.",
    "swing",
    "shuffle",
    "John",
    "Mary",
    "Alex",
    "Sam",
    "Alice",
    "Bob",
    "Acoustic",
    "Electric",
    "Distorted",
    "Clean",
    "Overdrive",
    "Finger",
    "Pick",
    "Fretless",
]


def text_tokens(text):
    values = list(text.encode("utf-8"))
    if len(values) > LENGTH:
        raise ValueError("Instrument text exceeds the model input limit")
    tokens = np.zeros(LENGTH, dtype=np.int64)
    tokens[: len(values)] = np.asarray(values, dtype=np.int64) + 1
    return tokens


def musical_text(rng):
    """Generic musical text negatives, independent of held-out OCR answers."""
    note = rng.choice('ABCDEFG')
    accidental = rng.choice(['', '#', 'b', '♯', '♭', '♮', '𝄪', '𝄫'])
    number = str(rng.randint(0, 9)) if rng.random() < 0.5 else ''
    separator = rng.choice(['', '.', ' ', ' . '])
    values = [note + separator + accidental + number + rng.choice(['', '.']),
              f's{rng.randint(1, 8)}f{rng.randint(0, 24)}p{rng.randint(24, 100)}{accidental}',
              str(rng.randint(20, 100)), f'{rng.randint(1, 12)}/{rng.choice([2, 4, 8, 16])}']
    return rng.choice(values)


def sources(root, split):
    synthetic = [
        json.loads(line) for line in (Path(root) / f"{split}.jsonl").read_text().splitlines()
    ]
    records = []
    for row in synthetic:
        prov = row["provenance"]
        answer = row["messages"][-1]["content"]
        if not answer.lstrip().startswith('{'):
            continue
        target = json.loads(answer)
        text = prov.get("visible_name", target.get("name"))
        if (
            "string_count" not in target
            or not text
            or target.get("instrument") not in LABELS
        ):
            continue
        records.append(
            {
                "text": text.strip(),
                "instrument": target["instrument"],
                "family": prov["family"],
                "source": prov.get("dataset", "native"),
            }
        )
    return records


class InstrumentTextData(Dataset):
    def __init__(self, records, count, training):
        # Repeated bass tracks must not outweigh a rarer explicit name such
        # as Cello. This balances text supervision, not runtime predictions.
        self.records = list({(r["text"], r["instrument"]): r for r in records}.values())
        self.count = count
        self.training = training

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = random.Random(650781 + index + (0 if self.training else 1000000))
        record = self.records[index % len(self.records)]
        text = record["text"]
        label = LABELS.index(record["instrument"])
        if rng.random() < 0.4:
            label = 0
            if rng.random() < 0.3:
                text = musical_text(rng)
            elif rng.random() < 0.35 and len(text) > 4:
                text = text[
                    : rng.randint(1, max(1, len(text) - 3))
                ] + "." * rng.randint(2, 6)
            elif rng.random() < 0.2:
                text = str(rng.randint(0, 500))
            else:
                text = rng.choice(NEGATIVES)
                if rng.random() < 0.4:
                    text += " " + str(rng.randint(1, 12))
        else:
            if rng.random() < 0.35:
                text += " " + rng.choice(["I", "II", "III", "1", "2", "3", "L", "R"])
            if rng.random() < 0.25:
                text = text.replace(".", " ")
            if rng.random() < 0.15 and len(text) > 5:
                position = rng.randrange(len(text))
                original = text[position]
                replacements = {
                    "l": "I",
                    "I": "l",
                    "o": "0",
                    "O": "0",
                    "i": "l",
                    "m": "rn",
                    "n": "r",
                    "u": "v",
                }
                if original in replacements:
                    text = (
                        text[:position] + replacements[original] + text[position + 1 :]
                    )
        case = rng.randrange(4)
        if case == 0:
            text = text.upper()
        elif case == 1:
            text = text.lower()
        elif case == 2:
            text = text.title()
        return text_tokens(text), np.int64(label)


def evaluate(model, loader):
    model.eval()
    probabilities = []
    targets = []
    with torch.inference_mode():
        for tokens, labels in loader:
            probabilities.extend(model(tokens.cuda()).softmax(-1).cpu().tolist())
            targets.extend(labels.tolist())
    probs = np.asarray(probabilities)
    truth = np.asarray(targets)
    pred = probs.argmax(1)
    confidence = probs.max(1)
    values = []
    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]:
        accepted = (confidence >= threshold) & (pred > 0)
        correct = accepted & (pred == truth)
        wrong = int((accepted & ~correct).sum())
        values.append(
            {
                "threshold": threshold,
                "accepted": int(accepted.sum()),
                "correct": int(correct.sum()),
                "wrong": wrong,
                "false_identity_on_unknown": int((accepted & (truth == 0)).sum()),
            }
        )
    eligible = [
        v
        for v in values
        if v["accepted"]
        and v["wrong"] / v["accepted"] <= 0.005
        and v["false_identity_on_unknown"] == 0
    ]
    choice = (
        max(eligible, key=lambda v: (v["correct"], -v["threshold"]))
        if eligible
        else {
            "threshold": 1.0,
            "accepted": 0,
            "correct": 0,
            "wrong": 0,
            "false_identity_on_unknown": 0,
        }
    )
    return {
        "selected": choice,
        "thresholds": values,
        "argmax_correct": int((pred == truth).sum()),
        "samples": len(truth),
        "probs": probs.tolist(),
        "targets": truth.tolist(),
    }


def export(directory):
    import onnx
    from research.inference import cpu_session

    torch.set_num_threads(2)
    validation = json.loads((directory / "validation.json").read_text())
    model = InstrumentTextNetwork().eval()
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    )
    tokens = torch.from_numpy(
        np.stack([text_tokens(t) for t in ["Guitar", "Bass", "Violin", "Drums", "合唱", "Intro"]])
    )
    target = directory / "instrument_text.onnx"
    torch.onnx.export(
        model, tokens[:1], target, input_names=["tokens"], output_names=["logits"],
        dynamic_axes={key: {0: "batch"} for key in ["tokens", "logits"]},
        opset_version=17, dynamo=False,
    )
    graph = onnx.load(target)
    config = dict(
        version=1, labels=LABELS, length=LENGTH,
        encoding="utf8_bytes_plus_one_zero_pad", overlong="abstain_without_truncation",
        empty="abstain", threshold=validation["selected"]["threshold"],
        source_checkpoint=validation["step"],
    )
    onnx.helper.set_model_props(graph, {"guitarocr.instrument_text": json.dumps(config)})
    onnx.save(graph, target)
    with torch.inference_mode():
        expected = model(tokens).numpy()
    actual = cpu_session(target).run(["logits"], {"tokens": tokens.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-4)
    print(json.dumps(dict(
        output=str(target), threshold=config["threshold"],
        torch_onnx_max_logit_error=float(np.abs(actual - expected).max()),
    )))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--export", type=Path, metavar="MODEL_DIR")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--context-data", type=Path)
    args = p.parse_args()
    if args.export:
        export(args.export)
        return
    if args.data is None or args.output is None:
        p.error("Training requires --data and --output")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(910287)
    rs = {s: sources(args.data, s) for s in ["train", "validation"]}
    if args.context_data:
        for split in rs:
            for line in (
                (args.context_data / f"{split}-labels.jsonl").read_text().splitlines()
            ):
                row = json.loads(line)
                if row.get("name") and row.get("instrument") in LABELS:
                    rs[split].append(
                        {
                            "text": row["name"],
                            "instrument": row["instrument"],
                            "family": row["family"],
                            "source": "authored_context_labels",
                        }
                    )
    assert not {r["family"] for r in rs["train"]} & {
        r["family"] for r in rs["validation"]
    }
    train = DataLoader(
        InstrumentTextData(rs["train"], 30000, True),
        batch_size=128,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    validation = DataLoader(
        InstrumentTextData(rs["validation"], 3000, False), batch_size=256, num_workers=2
    )
    model = InstrumentTextNetwork().cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    report = fit_classifier(
        model,
        train,
        optim,
        steps=args.steps,
        output=args.output,
        evaluate=lambda current: evaluate(current, validation),
    )
    report.update(
        labels=LABELS,
        length=LENGTH,
        seed=910287,
        source_counts={split: len(rows) for split, rows in rs.items()},
        data=str(args.data),
        context_data=str(args.context_data) if args.context_data else None,
    )
    (args.output / "training.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
