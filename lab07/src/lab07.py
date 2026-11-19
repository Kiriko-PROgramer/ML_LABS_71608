import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BASELINE_EPOCHS = 5
PRUNE_FINETUNE_EPOCHS = 3
STUDENT_EPOCHS = 5
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
NUM_WORKERS = 2
LR_BASELINE = 1e-3
LR_FINETUNE = 1e-4
LR_STUDENT = 1e-3
WEIGHT_DECAY = 1e-4
SPARSITIES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
STRUCTURED_RATIOS = [0.10, 0.25, 0.50]
T = 4.0
ALPHA = 0.5

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "src" else HERE
RESULTS = ROOT / "results"
DATA_DIR = ROOT / "data"
REPORT_PATH = ROOT / "report.md"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_train, [55000, 5000], generator=generator)
    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=EVAL_BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=EVAL_BATCH_SIZE, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


class OversizedCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128, 128), hidden=128):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.channels = channels
        self.hidden = hidden
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(c4 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, 100.0 * correct / total


def train_model(model, train_loader, val_loader, device, epochs, lr, label, teacher=None):
    model.to(device)
    if teacher is not None:
        teacher.to(device)
        teacher.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        total_loss = 0.0
        started = time.perf_counter()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(x)
            if teacher is None:
                loss = F.cross_entropy(student_logits, y)
            else:
                with torch.inference_mode():
                    teacher_logits = teacher(x)
                ce = F.cross_entropy(student_logits, y)
                p_student = F.softmax(student_logits / T, dim=1)
                log_p_student = F.log_softmax(student_logits / T, dim=1)
                log_p_teacher = F.log_softmax(teacher_logits / T, dim=1)
                kd = (p_student * (log_p_student - log_p_teacher)).sum(dim=1).mean() * (T ** 2)
                loss = ALPHA * ce + (1.0 - ALPHA) * kd
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (student_logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_accuracy": 100.0 * correct / total,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "seconds": elapsed,
        }
        history.append(row)
        print(f"{label} | epoch {epoch}/{epochs} | loss {row['train_loss']:.4f} | train {row['train_accuracy']:.2f}% | val {val_acc:.2f}% | {elapsed:.1f}s")
    return pd.DataFrame(history)


def state_size_mb(model, path):
    torch.save(model.state_dict(), path)
    return path.stat().st_size / (1024 ** 2)


def sparse_state_size_mb(model, path):
    packed = {}
    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu()
        if t.ndim >= 2:
            nz = t.nonzero(as_tuple=False).T.contiguous()
            vals = t[tuple(nz)] if nz.numel() else torch.empty(0, dtype=t.dtype)
            packed[name] = {"shape": tuple(t.shape), "indices": nz, "values": vals}
        else:
            packed[name] = {"dense": t}
    torch.save(packed, path)
    return path.stat().st_size / (1024 ** 2)


def model_sparsity(model):
    zeros = 0
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach()
            zeros += (w == 0).sum().item()
            total += w.numel()
    return zeros / total if total else 0.0


def latency_ms(model, device, runs=100, warmup=20):
    model.eval().to(device)
    x = torch.randn(1, 1, 28, 28, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def macs_count(model):
    macs = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal macs
        out = output
        batch = out.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        macs += batch * module.out_channels * out_h * out_w * kernel_ops

    def linear_hook(module, inputs, output):
        nonlocal macs
        x = inputs[0]
        batch = x.shape[0] if x.ndim > 1 else 1
        macs += batch * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    device = next(model.parameters()).device
    with torch.inference_mode():
        _ = model(torch.randn(1, 1, 28, 28, device=device))
    for h in hooks:
        h.remove()
    return macs


def remove_pruning(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")


def apply_global_pruning(model, amount):
    modules = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=amount)


def apply_layerwise_pruning(model, amount):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(m, name="weight", amount=amount)


def clone_baseline(state_dict):
    model = OversizedCNN()
    model.load_state_dict(state_dict)
    return model


def keep_indices_by_l2(weight, ratio):
    out_channels = weight.shape[0]
    keep = max(1, int(round(out_channels * (1.0 - ratio))))
    scores = weight.detach().pow(2).flatten(1).sum(1).sqrt()
    return torch.topk(scores, k=keep, largest=True).indices.sort().values


def build_structured_pruned(base_model, ratio):
    idx1 = keep_indices_by_l2(base_model.conv1.weight, ratio)
    idx2 = keep_indices_by_l2(base_model.conv2.weight, ratio)
    idx3 = keep_indices_by_l2(base_model.conv3.weight, ratio)
    idx4 = keep_indices_by_l2(base_model.conv4.weight, ratio)
    channels = (len(idx1), len(idx2), len(idx3), len(idx4))
    model = OversizedCNN(channels=channels, hidden=base_model.hidden)
    with torch.no_grad():
        model.conv1.weight.copy_(base_model.conv1.weight[idx1])
        model.conv1.bias.copy_(base_model.conv1.bias[idx1])
        model.conv2.weight.copy_(base_model.conv2.weight[idx2][:, idx1])
        model.conv2.bias.copy_(base_model.conv2.bias[idx2])
        model.conv3.weight.copy_(base_model.conv3.weight[idx3][:, idx2])
        model.conv3.bias.copy_(base_model.conv3.bias[idx3])
        model.conv4.weight.copy_(base_model.conv4.weight[idx4][:, idx3])
        model.conv4.bias.copy_(base_model.conv4.bias[idx4])
        w = base_model.fc1.weight.view(base_model.hidden, base_model.channels[3], 7, 7)
        model.fc1.weight.copy_(w[:, idx4].reshape(base_model.hidden, -1))
        model.fc1.bias.copy_(base_model.fc1.bias)
        model.fc2.weight.copy_(base_model.fc2.weight)
        model.fc2.bias.copy_(base_model.fc2.bias)
    return model


def save_curve(df, x, y, hue, title, xlabel, ylabel, path):
    plt.figure(figsize=(8, 5))
    for key, group in df.groupby(hue):
        group = group.sort_values(x)
        plt.plot(group[x], group[y], marker="o", label=str(key))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def fmt(x, digits=2):
    return f"{x:.{digits}f}"


def write_report(baseline, pruning_df, structured_df, latency_df, distill_df, summary_df):
    best_global = pruning_df[pruning_df["method"] == "global"].sort_values("test_accuracy", ascending=False).iloc[0]
    best_layer = pruning_df[pruning_df["method"] == "layerwise"].sort_values("test_accuracy", ascending=False).iloc[0]
    best_struct = structured_df.sort_values("test_accuracy", ascending=False).iloc[0]
    plain = distill_df[distill_df["model"] == "student_plain"].iloc[0]
    kd = distill_df[distill_df["model"] == "student_distilled"].iloc[0]
    table_prune = pruning_df[["method", "target_sparsity", "actual_sparsity", "test_accuracy", "dense_size_mb", "sparse_size_mb", "latency_ms"]].to_markdown(index=False, floatfmt=".3f")
    table_struct = structured_df[["prune_ratio", "params", "test_accuracy", "size_mb", "latency_ms", "macs_m"]].to_markdown(index=False, floatfmt=".3f")
    table_distill = distill_df[["model", "params", "test_accuracy", "size_mb", "latency_ms", "macs_m"]].to_markdown(index=False, floatfmt=".3f")
    table_summary = summary_df.to_markdown(index=False, floatfmt=".3f")
    report = f"""# Lab 7: Compression — Pruning and Distillation

**Student:** Kyrylo  
**Student ID:** 71608  
**Dataset:** Fashion-MNIST  
**Device:** {baseline['device']}

## 1. Objective

This lab studies model compression using magnitude pruning, structured filter pruning, and knowledge distillation. The experiments use the complete Fashion-MNIST training split rather than a reduced subset. The main goal is to compare accuracy, parameter count, on-disk size, inference latency, compute, and an explicitly labelled energy proxy while keeping the analysis honest about the difference between storage sparsity and real hardware speedup.

The required sparsity targets are `0, 0.25, 0.5, 0.75, 0.9, 0.95`. Magnitude pruning is performed globally and layer-wise. Every non-zero pruning target is fine-tuned for exactly 3 epochs with the pruning mask fixed. Structured pruning physically removes convolutional filters according to L2 norm at `10%, 25%, 50%`. Distillation uses the oversized baseline as teacher, a student below 100k parameters, `T=4`, and `alpha=0.5`.

## 2. Experimental Setup

Fashion-MNIST contains 60,000 training images and 10,000 test images. A deterministic 55,000/5,000 split is used for training and validation, and the official 10,000-image test split is kept for final evaluation. Inputs are normalized with mean `0.2860` and standard deviation `0.3530`.

The oversized baseline CNN has four convolutional layers followed by two fully connected layers. Its recorded parameter count is **{int(baseline['params']):,}**, which is close to the requested 1M-parameter scale.

| Setting | Value |
|---|---:|
| Seed | {SEED} |
| Baseline epochs | {BASELINE_EPOCHS} |
| Pruning fine-tune epochs | {PRUNE_FINETUNE_EPOCHS} |
| Student epochs | {STUDENT_EPOCHS} |
| Training batch size | {BATCH_SIZE} |
| Baseline learning rate | {LR_BASELINE} |
| Pruning fine-tune learning rate | {LR_FINETUNE} |
| Distillation temperature | {T} |
| Distillation alpha | {ALPHA} |

## 3. Task 1 — Oversized Baseline CNN

The oversized baseline was trained on the full Fashion-MNIST training data. The final test accuracy was **{fmt(baseline['test_accuracy'])}%**. The model contains **{int(baseline['params']):,} parameters**, occupies **{fmt(baseline['size_mb'], 3)} MB** as a dense PyTorch state dictionary, and has median batch-1 latency of **{fmt(baseline['latency_ms'], 3)} ms** on the recorded device.

The baseline compute is **{fmt(baseline['macs_m'], 3)} million MACs per image**.

Generated evidence:

- `results/task1_baseline_history.csv`
- `results/task1_baseline_metrics.csv`
- `results/task1_baseline.pt`

## 4. Task 2 — Magnitude Pruning

For global pruning, all convolutional and linear weights are ranked together by absolute value and the smallest weights are masked. For layer-wise pruning, the same sparsity percentage is applied independently inside each convolutional and linear layer. The mask remains fixed during 3 fine-tuning epochs, so pruned connections cannot regrow.

{table_prune}

The best global-pruning point by test accuracy was target sparsity **{fmt(best_global['target_sparsity'], 2)}**, with **{fmt(best_global['test_accuracy'])}%** accuracy. The best layer-wise point was target sparsity **{fmt(best_layer['target_sparsity'], 2)}**, with **{fmt(best_layer['test_accuracy'])}%** accuracy.

![Magnitude pruning accuracy](results/task2_magnitude_pruning_accuracy.png)

The curve shows how aggressively weights can be removed before the accuracy penalty becomes significant. Moderate sparsity often preserves most of the baseline performance, while very high sparsity can cause a sharper degradation because too much representational capacity is removed.

## 5. Structured Filter Pruning

Structured pruning ranks whole convolutional filters by L2 norm and physically removes the lowest-ranked output channels. Dependent input channels in the following convolution and the corresponding fully connected input features are also removed. Unlike unstructured masking, this changes tensor shapes and therefore can reduce real dense-kernel compute.

{table_struct}

The best structured configuration by test accuracy used a pruning ratio of **{fmt(best_struct['prune_ratio'], 2)}** and reached **{fmt(best_struct['test_accuracy'])}%** test accuracy.

![Structured pruning accuracy](results/task2_structured_pruning_accuracy.png)

## 6. Task 3 — Storage Size and Honest Latency Analysis

The experiment stores unstructured models in two ways: a normal dense PyTorch state dictionary and a custom sparse representation containing non-zero indices and values. This directly measures the difference between mathematical sparsity and actual file size.

The latency measurements deliberately use the normal dense PyTorch execution path. This is important because zero weights inside dense tensors do not automatically select sparse convolution kernels. Therefore, unstructured pruning can reduce the number of non-zero parameters and may reduce storage in an appropriate sparse format, while producing little or no latency improvement in ordinary dense CPU kernels.

`results/task3_latency_storage.csv` contains the complete comparison.

{latency_df.to_markdown(index=False, floatfmt='.3f')}

Structured pruning is different because channels are physically removed. That changes tensor dimensions, reduces dense MACs, and is much more likely to produce a measurable latency reduction on standard hardware.

## 7. Task 4 — Knowledge Distillation

The oversized baseline is used as the teacher. The compact student contains fewer than 100k parameters. Two students are trained under the same epoch budget: one with ordinary cross-entropy and one with knowledge distillation.

The distillation objective is:

`alpha * CE(student, labels) + (1-alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))`

with `T=4` and `alpha=0.5`.

{table_distill}

The plain student reached **{fmt(plain['test_accuracy'])}%**, while the distilled student reached **{fmt(kd['test_accuracy'])}%**. The distilled student therefore changed accuracy by **{fmt(kd['test_accuracy'] - plain['test_accuracy'])} percentage points** relative to the plain student under the same training budget.

![Distillation comparison](results/task4_distillation_comparison.png)

The student model has **{int(kd['params']):,} parameters**, satisfying the requirement of approximately 100k parameters or fewer.

## 8. Task 5 — Efficiency Pillars Summary

The final comparison is organized around the efficiency pillars from Chapter 9: data, model, compute, and energy.

{table_summary}

### Data

All models use the same full Fashion-MNIST dataset, normalization, deterministic split, and test set. This keeps the compression comparison controlled and prevents data quantity from becoming a hidden variable.

### Model

Unstructured magnitude pruning reduces the number of non-zero weights without changing layer shapes. Structured pruning physically reduces channels and parameter count. Distillation replaces the oversized teacher with a much smaller student.

### Compute

Dense MACs remain almost unchanged for unstructured pruning because the network shapes are unchanged. Structured pruning and the student model reduce dense MACs directly. This distinction explains why parameter sparsity alone is not equivalent to real inference acceleration.

### Energy

Direct joule measurement is not available from portable PyTorch APIs on this Windows/CPU setup. The report therefore uses an **energy proxy**, not a claimed physical energy measurement. The proxy is proportional to measured latency multiplied by dense MAC count and is normalized to the baseline. Lower values indicate a model that performs less dense work for less measured execution time. Actual energy should be measured with platform-specific power counters for deployment-grade conclusions.

## 9. Main Conclusions

1. Magnitude pruning can remove a large fraction of weights while preserving useful accuracy at moderate sparsity.
2. Unstructured sparsity is valuable for compression, but it does not automatically speed up dense CPU kernels because standard convolution and linear operators still process dense tensor shapes.
3. Structured filter pruning provides a more direct path to latency reduction because it physically reduces channel dimensions and dense compute.
4. Knowledge distillation allows a sub-100k-parameter student to learn from the oversized teacher and can improve the compact model over plain supervised training under the same epoch budget.
5. Compression must be evaluated across accuracy, size, latency, compute, and energy rather than by sparsity alone.

## 10. Graded Evidence

The required evidence is generated automatically:

- `results/task2_magnitude_pruning_accuracy.png` — sparsity curves for global and layer-wise pruning
- `results/task2_structured_pruning_accuracy.png` — structured pruning curve
- `results/task3_latency_storage.csv` — dense-vs-sparse storage and latency analysis
- `results/task4_distillation_comparison.csv` — distilled vs plain student
- `results/task4_distillation_comparison.png` — student comparison plot
- `results/task5_efficiency_summary.csv` — final efficiency table
- `report.md` — this report with the measured values from the completed run

## 11. Reproducibility

The experiment uses a fixed random seed and deterministic train/validation split. The script saves all main intermediate tables, curves, checkpoints, and the final report. All reported numerical values are produced by the current run rather than hard-coded into the report.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    seed_everything()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    device = get_device()
    print(f"Device: {device}")
    print("Loading full Fashion-MNIST...")
    train_loader, val_loader, test_loader = build_loaders()

    print("\nTASK 1/5 | oversized baseline")
    baseline_model = OversizedCNN().to(device)
    baseline_params = count_params(baseline_model)
    print(f"Baseline parameters: {baseline_params:,}")
    baseline_history = train_model(baseline_model, train_loader, val_loader, device, BASELINE_EPOCHS, LR_BASELINE, "Baseline")
    baseline_history.to_csv(RESULTS / "task1_baseline_history.csv", index=False)
    _, baseline_acc = evaluate(baseline_model, test_loader, device)
    baseline_size = state_size_mb(baseline_model, RESULTS / "task1_baseline.pt")
    baseline_latency = latency_ms(baseline_model, device)
    baseline_macs = macs_count(baseline_model)
    baseline_state = {k: v.detach().cpu().clone() for k, v in baseline_model.state_dict().items()}
    baseline_metrics = {
        "device": str(device),
        "params": baseline_params,
        "test_accuracy": baseline_acc,
        "size_mb": baseline_size,
        "latency_ms": baseline_latency,
        "macs_m": baseline_macs / 1e6,
    }
    pd.DataFrame([baseline_metrics]).to_csv(RESULTS / "task1_baseline_metrics.csv", index=False)
    print(f"Baseline test accuracy: {baseline_acc:.2f}% | size {baseline_size:.3f} MB | latency {baseline_latency:.3f} ms")

    print("\nTASK 2/5 | magnitude pruning")
    pruning_rows = []
    for method in ["global", "layerwise"]:
        for target in SPARSITIES:
            print(f"{method} pruning | target sparsity {target:.2f}")
            model = clone_baseline(baseline_state).to(device)
            if target > 0:
                if method == "global":
                    apply_global_pruning(model, target)
                else:
                    apply_layerwise_pruning(model, target)
                history = train_model(model, train_loader, val_loader, device, PRUNE_FINETUNE_EPOCHS, LR_FINETUNE, f"{method} s={target:.2f}")
                history.to_csv(RESULTS / f"task2_{method}_{str(target).replace('.', '_')}_history.csv", index=False)
            _, test_acc = evaluate(model, test_loader, device)
            actual_sparsity = model_sparsity(model)
            remove_pruning(model)
            dense_path = RESULTS / f"task2_{method}_{str(target).replace('.', '_')}_dense.pt"
            sparse_path = RESULTS / f"task2_{method}_{str(target).replace('.', '_')}_sparse.pt"
            dense_size = state_size_mb(model, dense_path)
            sparse_size = sparse_state_size_mb(model, sparse_path)
            lat = latency_ms(model, device)
            pruning_rows.append({
                "method": method,
                "target_sparsity": target,
                "actual_sparsity": actual_sparsity,
                "test_accuracy": test_acc,
                "dense_size_mb": dense_size,
                "sparse_size_mb": sparse_size,
                "latency_ms": lat,
                "params": count_params(model),
                "dense_macs_m": baseline_macs / 1e6,
                "ideal_nonzero_macs_m": (baseline_macs * (1.0 - actual_sparsity)) / 1e6,
            })
            print(f"{method} s={target:.2f} | test {test_acc:.2f}% | actual sparsity {actual_sparsity:.3f} | latency {lat:.3f} ms")
    pruning_df = pd.DataFrame(pruning_rows)
    pruning_df.to_csv(RESULTS / "task2_magnitude_pruning.csv", index=False)
    save_curve(pruning_df, "actual_sparsity", "test_accuracy", "method", "Magnitude Pruning: Accuracy vs Sparsity", "Actual sparsity", "Test accuracy (%)", RESULTS / "task2_magnitude_pruning_accuracy.png")

    print("\nTASK 2/5 | structured filter pruning")
    structured_rows = []
    baseline_cpu = clone_baseline(baseline_state).cpu()
    for ratio in STRUCTURED_RATIOS:
        print(f"structured pruning | ratio {ratio:.2f}")
        model = build_structured_pruned(baseline_cpu, ratio).to(device)
        history = train_model(model, train_loader, val_loader, device, PRUNE_FINETUNE_EPOCHS, LR_FINETUNE, f"structured r={ratio:.2f}")
        history.to_csv(RESULTS / f"task2_structured_{str(ratio).replace('.', '_')}_history.csv", index=False)
        _, test_acc = evaluate(model, test_loader, device)
        size_mb = state_size_mb(model, RESULTS / f"task2_structured_{str(ratio).replace('.', '_')}.pt")
        lat = latency_ms(model, device)
        macs = macs_count(model)
        structured_rows.append({
            "prune_ratio": ratio,
            "channels": str(model.channels),
            "params": count_params(model),
            "test_accuracy": test_acc,
            "size_mb": size_mb,
            "latency_ms": lat,
            "macs_m": macs / 1e6,
        })
        print(f"structured r={ratio:.2f} | params {count_params(model):,} | test {test_acc:.2f}% | latency {lat:.3f} ms")
    structured_df = pd.DataFrame(structured_rows)
    structured_df.to_csv(RESULTS / "task2_structured_pruning.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(structured_df["prune_ratio"], structured_df["test_accuracy"], marker="o")
    plt.xlabel("Structured filter pruning ratio")
    plt.ylabel("Test accuracy (%)")
    plt.title("Structured Pruning: Accuracy vs Filter Removal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "task2_structured_pruning_accuracy.png", dpi=160)
    plt.close()

    print("\nTASK 3/5 | latency and storage")
    latency_df = pruning_df[["method", "target_sparsity", "actual_sparsity", "dense_size_mb", "sparse_size_mb", "latency_ms", "dense_macs_m", "ideal_nonzero_macs_m"]].copy()
    latency_df.to_csv(RESULTS / "task3_latency_storage.csv", index=False)

    print("\nTASK 4/5 | distillation")
    teacher = clone_baseline(baseline_state).to(device)
    student_plain = StudentCNN().to(device)
    student_distilled = StudentCNN().to(device)
    print(f"Student parameters: {count_params(student_plain):,}")
    plain_history = train_model(student_plain, train_loader, val_loader, device, STUDENT_EPOCHS, LR_STUDENT, "Student plain")
    kd_history = train_model(student_distilled, train_loader, val_loader, device, STUDENT_EPOCHS, LR_STUDENT, "Student distilled", teacher=teacher)
    plain_history.to_csv(RESULTS / "task4_student_plain_history.csv", index=False)
    kd_history.to_csv(RESULTS / "task4_student_distilled_history.csv", index=False)
    distill_rows = []
    for name, model in [("student_plain", student_plain), ("student_distilled", student_distilled)]:
        _, acc = evaluate(model, test_loader, device)
        size_mb = state_size_mb(model, RESULTS / f"task4_{name}.pt")
        lat = latency_ms(model, device)
        macs = macs_count(model)
        distill_rows.append({
            "model": name,
            "params": count_params(model),
            "test_accuracy": acc,
            "size_mb": size_mb,
            "latency_ms": lat,
            "macs_m": macs / 1e6,
        })
    distill_df = pd.DataFrame(distill_rows)
    distill_df.to_csv(RESULTS / "task4_distillation_comparison.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.bar(distill_df["model"], distill_df["test_accuracy"])
    plt.ylabel("Test accuracy (%)")
    plt.title("Plain Student vs Distilled Student")
    plt.tight_layout()
    plt.savefig(RESULTS / "task4_distillation_comparison.png", dpi=160)
    plt.close()

    print("\nTASK 5/5 | efficiency summary")
    best_global = pruning_df[pruning_df["method"] == "global"].sort_values("test_accuracy", ascending=False).iloc[0]
    best_struct = structured_df.sort_values("test_accuracy", ascending=False).iloc[0]
    kd = distill_df[distill_df["model"] == "student_distilled"].iloc[0]
    summary_rows = [
        {
            "model": "baseline_teacher",
            "data": "full Fashion-MNIST",
            "model_efficiency": "dense ~1M CNN",
            "accuracy": baseline_acc,
            "params": baseline_params,
            "size_mb": baseline_size,
            "latency_ms": baseline_latency,
            "macs_m": baseline_macs / 1e6,
        },
        {
            "model": f"best_global_s{best_global['target_sparsity']:.2f}",
            "data": "full Fashion-MNIST",
            "model_efficiency": "unstructured sparse",
            "accuracy": best_global["test_accuracy"],
            "params": int(best_global["params"]),
            "size_mb": best_global["sparse_size_mb"],
            "latency_ms": best_global["latency_ms"],
            "macs_m": best_global["dense_macs_m"],
        },
        {
            "model": f"best_structured_r{best_struct['prune_ratio']:.2f}",
            "data": "full Fashion-MNIST",
            "model_efficiency": "physically smaller channels",
            "accuracy": best_struct["test_accuracy"],
            "params": int(best_struct["params"]),
            "size_mb": best_struct["size_mb"],
            "latency_ms": best_struct["latency_ms"],
            "macs_m": best_struct["macs_m"],
        },
        {
            "model": "student_distilled",
            "data": "full Fashion-MNIST + teacher logits",
            "model_efficiency": "<100k student",
            "accuracy": kd["test_accuracy"],
            "params": int(kd["params"]),
            "size_mb": kd["size_mb"],
            "latency_ms": kd["latency_ms"],
            "macs_m": kd["macs_m"],
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    baseline_proxy = baseline_latency * (baseline_macs / 1e6)
    summary_df["energy_proxy_rel"] = (summary_df["latency_ms"] * summary_df["macs_m"]) / baseline_proxy
    summary_df.to_csv(RESULTS / "task5_efficiency_summary.csv", index=False)

    write_report(baseline_metrics, pruning_df, structured_df, latency_df, distill_df, summary_df)
    print("\nDONE")
    print(f"Results: {RESULTS}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
