"""Small training routines shared by the auxiliary classifiers."""


def fit_classifier(
    model,
    loader,
    optimizer,
    *,
    steps,
    output,
    evaluate,
    score_field="correct",
    inputs=1,
    scheduler=None,
    clip=None,
):
    """Fit a classifier and retain its best validation checkpoint."""
    import json
    import time

    import torch

    if steps < 1 or not len(loader):
        raise ValueError("Training requires positive steps and a nonempty loader")
    device = next(model.parameters()).device
    started = time.monotonic()
    step, best, history = 0, -1, []
    while step < steps:
        model.train()
        for batch in loader:
            values = [
                value.to(device, non_blocking=True) for value in batch[: inputs + 1]
            ]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(*values[:inputs]), values[inputs]
            )
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            if step % 100 == 0:
                print(
                    dict(
                        step=step,
                        loss=float(loss.detach()),
                        seconds=time.monotonic() - started,
                    ),
                    flush=True,
                )
            if step % 300 == 0 or step == steps:
                report = evaluate(model)
                entry = dict(
                    step=step,
                    selected=report["selected"],
                    argmax_correct=report["argmax_correct"],
                )
                history.append(entry)
                print(json.dumps(entry), flush=True)
                score = report["selected"][score_field]
                if score > best:
                    best = score
                    state = {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                    torch.save(state, output / "model.pt")
                    (output / "validation.json").write_text(
                        json.dumps(dict(step=step, **report), indent=2) + "\n"
                    )
                model.train()
            if step == steps:
                break
    return dict(
        history=history,
        steps=step,
        seconds=time.monotonic() - started,
        parameters=sum(p.numel() for p in model.parameters()),
    )
