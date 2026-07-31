#!/usr/bin/env python3
"""Train a root-only dense GCN policy from extracted Stockfish binpack samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


MAGIC = b"EGNNROOT1"
RECORD_SIZE = 77
HEADER_SIZE = len(MAGIC) + 4 + 4
FEATURES = 16
POLICY_SIZE = 64 * 73


DTYPE = np.dtype(
    [
        ("board", "u1", (64,)),
        ("stm", "u1"),
        ("from_sq", "u1"),
        ("to_sq", "u1"),
        ("promo", "u1"),
        ("move_index", "<u2"),
        ("score", "<i2"),
        ("result", "u1"),
        ("ply", "<u2"),
        ("reserved", "<u2"),
    ]
)


class RootPolicySamples(Dataset):
    def __init__(self, path: Path, indices: np.ndarray):
        self.path = path
        self.indices = indices
        self.records = open_records(path)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        rec = self.records[int(self.indices[idx])]
        return (
            torch.from_numpy(np.array(rec["board"], copy=True)).long(),
            torch.tensor(int(rec["stm"]), dtype=torch.long),
            torch.tensor(int(rec["move_index"]), dtype=torch.long),
        )


class DenseRootGCN(nn.Module):
    def __init__(self, hidden: int = 96, layers: int = 3):
        super().__init__()
        self.input = nn.Linear(FEATURES, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.policy = nn.Linear(hidden, 73)
        self.register_buffer("adjacency", build_static_adjacency(), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input(x))
        adjacency = self.adjacency.to(dtype=h.dtype)
        for layer in self.layers:
            mixed = torch.matmul(adjacency, h)
            h = torch.relu(layer(mixed))
        logits = self.policy(h)
        return logits.reshape(x.shape[0], POLICY_SIZE)


def open_records(path: Path) -> np.memmap:
    with path.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path} has bad magic {magic!r}")
        record_size = int.from_bytes(f.read(4), "little")
        if record_size != RECORD_SIZE:
            raise ValueError(f"{path} uses record size {record_size}, expected {RECORD_SIZE}")
        f.read(4)
    size = path.stat().st_size - HEADER_SIZE
    if size % RECORD_SIZE:
        raise ValueError(f"{path} has truncated records")
    return np.memmap(path, dtype=DTYPE, mode="r", offset=HEADER_SIZE, shape=(size // RECORD_SIZE,))


def build_static_adjacency() -> torch.Tensor:
    adj = torch.eye(64, dtype=torch.float32)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    knights = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    for sq in range(64):
        x, y = sq % 8, sq // 8
        for dx, dy in directions:
            for dist in range(1, 8):
                nx, ny = x + dx * dist, y + dy * dist
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    break
                adj[sq, ny * 8 + nx] = 1.0
        for dx, dy in knights:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                adj[sq, ny * 8 + nx] = 1.0
    degree = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / degree


def batch_to_features(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    board, stm, label = batch
    board = board.to(device, non_blocking=True)
    stm = stm.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)

    piece = torch.nn.functional.one_hot(board, num_classes=13).to(torch.float32)
    stm_feature = stm.to(torch.float32).view(-1, 1, 1).expand(-1, 64, 1)
    files = torch.linspace(0.0, 1.0, 8, device=device).repeat(8).view(1, 64, 1)
    ranks = torch.linspace(0.0, 1.0, 8, device=device).repeat_interleave(8).view(1, 64, 1)
    coords = torch.cat(
        [files.expand(board.shape[0], -1, -1), ranks.expand(board.shape[0], -1, -1)], dim=2
    )
    features = torch.cat([piece, stm_feature, coords], dim=2)
    return features, label


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> dict:
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = top3 = top5 = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        features, label = batch_to_features(batch, device)
        logits = model(features)
        loss_sum += float(loss_fn(logits, label).item())
        top = logits.topk(5, dim=1).indices
        total += label.numel()
        top1 += int((top[:, :1] == label[:, None]).any(dim=1).sum().item())
        top3 += int((top[:, :3] == label[:, None]).any(dim=1).sum().item())
        top5 += int((top[:, :5] == label[:, None]).any(dim=1).sum().item())
    return {
        "loss": loss_sum / max(total, 1),
        "top1": top1 / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "samples": total,
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    records = open_records(args.samples)
    count = len(records)
    if args.limit and args.limit < count:
        count = args.limit
    order = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(order)
    valid_count = max(1, int(count * args.valid_fraction))
    valid_idx = order[:valid_count]
    train_idx = order[valid_count:]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RootPolicySamples(args.samples, train_idx)
    valid_ds = RootPolicySamples(args.samples, valid_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = DenseRootGCN(hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    loss_fn = nn.CrossEntropyLoss()
    metrics = {
        "samples_path": str(args.samples),
        "record_count": int(count),
        "train_count": int(len(train_ds)),
        "valid_count": int(len(valid_ds)),
        "device": str(device),
        "hidden": args.hidden,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs_log": [],
    }

    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        total = 0
        loss_total = 0.0
        for batch in train_loader:
            features, label = batch_to_features(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and args.amp):
                logits = model(features)
                loss = loss_fn(logits, label)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += label.numel()
            loss_total += float(loss.item()) * label.numel()

        elapsed = time.time() - started
        valid = evaluate(model, valid_loader, device, args.valid_batches)
        row = {
            "epoch": epoch,
            "train_loss": loss_total / max(total, 1),
            "train_samples": total,
            "elapsed_s": elapsed,
            "samples_per_s": total / max(elapsed, 1e-9),
            "valid": valid,
        }
        metrics["epochs_log"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        ckpt = args.output_dir / f"checkpoint-epoch-{epoch}.pt"
        torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, ckpt)
        if valid["top1"] > best_top1:
            best_top1 = valid["top1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, args.output_dir / "best.pt")

    model.eval()
    dummy = torch.zeros(1, 64, FEATURES, dtype=torch.float32, device=device)
    onnx_path = args.output_dir / "root_policy.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
    )
    metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metrics["onnx_path"] = str(onnx_path)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"features": np.zeros((1, 64, FEATURES), dtype=np.float32)})[0]
        if got.shape != (1, POLICY_SIZE):
            raise RuntimeError(f"unexpected ONNX output shape {got.shape}")
        (args.output_dir / "onnxruntime-validation.txt").write_text(
            f"ok shape={got.shape} min={got.min()} max={got.max()}\n"
        )
    except Exception as exc:  # noqa: BLE001
        (args.output_dir / "onnxruntime-validation.txt").write_text(f"failed: {exc}\n")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--valid-batches", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
