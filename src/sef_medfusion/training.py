from __future__ import annotations
import copy, time
import numpy as np
import torch
from torch.nn import functional as F
from .metrics import classification
from .models import CompactCNN
from .utils import seed_all


@torch.no_grad()
def predict(model, images, batch_size=64):
    model.eval()
    return (
        torch.cat([model(batch).softmax(1) for batch in images.split(batch_size)])
        .cpu()
        .numpy()
    )


def augment(batch):
    x = batch.clone()
    mask = torch.rand(len(x)) < 0.5
    x[mask] = x[mask].flip(-1)
    mask = torch.rand(len(x)) < 0.5
    x[mask] = x[mask].flip(-2)
    rotations = torch.randint(4, (len(x),))
    for k in [1, 2, 3]:
        mask = rotations == k
        x[mask] = torch.rot90(x[mask], k, (-2, -1))
    return x


def train(model, train_x, train_y, tune_x, tune_y, config, progress=None):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    history = []
    best_f1 = -1.0
    best_state = None
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        for ids in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(augment(train_x[ids]))
            loss = F.cross_entropy(logits, train_y[ids])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        metrics = classification(
            tune_y.numpy(), predict(model, tune_x, config["batch_size"])
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total / len(train_x),
            "tune_macro_f1": metrics["macro_f1"],
            "tune_accuracy": metrics["accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        scheduler.step()
        if progress:
            progress(row)
    model.load_state_dict(best_state)
    model.eval()
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_tune_macro_f1": best_f1,
        "train_seconds": time.perf_counter() - started,
    }


def average_states(states, weights):
    """FedAvg weighted by local sample count, with explicit integer handling."""
    if len(states) != len(weights) or not states:
        raise ValueError("states/weights mismatch")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")
    total = sum(weights)
    result = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            result[key] = sum(
                state[key] * (weight / total) for state, weight in zip(states, weights)
            )
        else:
            result[key] = states[int(np.argmax(weights))][key].clone()
    return result


def federated_simulation(
    width, classes, train_x, train_y, test_x, test_y, config, seed
):
    """Single-process, balanced IID client simulation; no privacy guarantee."""
    seed_all(seed, config["threads"])
    model = CompactCNN(width, classes)
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(config["federated_clients"])]
    for label in sorted(train_y.unique().tolist()):
        ids = np.flatnonzero(train_y.numpy() == label)
        rng.shuffle(ids)
        for target, subset in zip(clients, np.array_split(ids, len(clients))):
            target.extend(subset.tolist())
    rounds = []
    for round_id in range(config["federated_rounds"]):
        states = []
        for client_ids in clients:
            local = copy.deepcopy(model)
            local.train()
            optimizer = torch.optim.AdamW(
                local.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            ids = torch.tensor(client_ids)
            for _ in range(config["federated_local_epochs"]):
                for batch in ids[torch.randperm(len(ids))].split(config["batch_size"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(
                        local(augment(train_x[batch])), train_y[batch]
                    )
                    loss.backward()
                    optimizer.step()
            states.append(copy.deepcopy(local.state_dict()))
        model.load_state_dict(average_states(states, [len(c) for c in clients]))
        rounds.append(
            {
                "round": round_id + 1,
                **classification(test_y.numpy(), predict(model, test_x)),
            }
        )
        print(
            "FedAvg round", round_id + 1, "accuracy", rounds[-1]["accuracy"], flush=True
        )
    return model, {
        "setting": "single-process balanced IID simulation; no privacy guarantees",
        "client_sizes": [len(c) for c in clients],
        "rounds": rounds,
    }
