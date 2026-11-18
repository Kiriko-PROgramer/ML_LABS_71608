import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*")

import os
import time
import json
import random
import subprocess
from pathlib import Path

import yaml
import torch
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.profiler import profile, record_function, ProfilerActivity

try:
    import psutil
except ImportError:
    psutil = None

CONFIG_PATH = Path("../lab6_config.yaml")


def default_config():
    return {
        "project": {
            "name": "lab6_training_pipeline",
            "seed": 42,
            "output_dir": "lab6_outputs_fast",
            "data_dir": "data",
            "student_name": "Kyrylo",
            "student_id": "71608"
        },
        "data": {
            "dataset": "FashionMNIST",
            "download": True,
            "batch_size": 128,
            "num_workers": 2,
            "fast_mode": True,
            "fast_train_size": 6144,
            "fast_test_size": 1024
        },
        "training": {
            "epochs": 5,
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer": "Adam",
            "momentum": 0.9,
            "scheduler": "CosineAnnealingLR",
            "step_size": 3,
            "gamma": 0.1,
            "accumulation_steps": 1
        },
        "studies": {
            "num_workers": [0, 1, 2, 4, 8],
            "worker_benchmark_batches": 32,
            "batch_sizes": [32, 64, 128, 256, 512],
            "schedulers": ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"],
            "optimizers": ["SGD", "Adam"],
            "batch_study_epochs": 5,
            "schedule_study_epochs": 10,
            "accumulation_epochs": 5,
            "resume_total_epochs": 6
        },
        "proposal": {
            "problem": "Fashion-MNIST image classification using the best CNN architecture from Lab 3 for compact deployment.",
            "dataset": "Fashion-MNIST: 60,000 training and 10,000 test grayscale images, 28x28 pixels, 10 clothing classes.",
            "success_metric": "Test accuracy, using the Lab 3 CNN result of 89.11% after 5 epochs as the reference baseline.",
            "latency_constraint_ms": 50,
            "size_constraint_mb": 5,
            "planned_optimization": "Post-training INT8 quantization, with structured pruning as a fallback if further compression is needed."
        }
    }


def merge_dict(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config():
    cfg = default_config()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        cfg = merge_dict(cfg, existing)
    cfg["project"]["student_name"] = "Kyrylo"
    cfg["project"]["student_id"] = "71608"
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % 2**32)


class Lab3CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.net(x)


def build_model(device):
    return Lab3CNN().to(device)


def build_datasets(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    root = cfg["project"]["data_dir"]
    train_full = datasets.FashionMNIST(root=root, train=True, download=cfg["data"]["download"], transform=transform)
    test_full = datasets.FashionMNIST(root=root, train=False, download=cfg["data"]["download"], transform=transform)
    if not cfg["data"]["fast_mode"]:
        return train_full, test_full
    g = torch.Generator().manual_seed(cfg["project"]["seed"])
    train_size = min(cfg["data"]["fast_train_size"], len(train_full))
    test_size = min(cfg["data"]["fast_test_size"], len(test_full))
    train_idx = torch.randperm(len(train_full), generator=g)[:train_size].tolist()
    test_idx = torch.randperm(len(test_full), generator=g)[:test_size].tolist()
    return Subset(train_full, train_idx), Subset(test_full, test_idx)


def training_workers(cfg):
    return 0 if cfg["data"]["fast_mode"] else cfg["data"]["num_workers"]


def build_loaders(cfg, train_set, test_set, batch_size=None, num_workers=None, persistent=None):
    batch_size = batch_size or cfg["data"]["batch_size"]
    if num_workers is None:
        num_workers = training_workers(cfg)
    if persistent is None:
        persistent = num_workers > 0
    g = torch.Generator().manual_seed(cfg["project"]["seed"])
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": g
    }
    if num_workers > 0:
        common["persistent_workers"] = persistent
        common["prefetch_factor"] = 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, **common)
    test_loader = DataLoader(test_set, batch_size=min(512, max(128, batch_size)), shuffle=False, drop_last=False, **common)
    return train_loader, test_loader


def build_optimizer(model, cfg, name=None):
    name = name or cfg["training"]["optimizer"]
    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg["training"]["lr"],
            momentum=cfg["training"]["momentum"],
            weight_decay=cfg["training"]["weight_decay"]
        )
    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"]
        )
    raise ValueError(name)


def build_scheduler(optimizer, cfg, name=None, epochs=None):
    name = name or cfg["training"]["scheduler"]
    epochs = epochs or cfg["training"]["epochs"]
    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["training"]["step_size"],
            gamma=cfg["training"]["gamma"]
        )
    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    raise ValueError(name)


def flatten_dict(d, parent=""):
    result = {}
    for key, value in d.items():
        name = f"{parent}.{key}" if parent else str(key)
        if isinstance(value, dict):
            result.update(flatten_dict(value, name))
        else:
            result[name] = json.dumps(value) if isinstance(value, (list, tuple)) else value
    return result


def process_memory_mb():
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train_one_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    peak_cpu = process_memory_mb()
    start = time.perf_counter()
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        raw_loss = criterion(logits, y)
        (raw_loss / accumulation_steps).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += raw_loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += y.size(0)
        if device.type == "cpu" and step % 8 == 0:
            current = process_memory_mb()
            if current is not None:
                peak_cpu = current if peak_cpu is None else max(peak_cpu, current)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_memory = peak_cpu
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "throughput": total_samples / elapsed,
        "elapsed": elapsed,
        "peak_memory_mb": peak_memory
    }


def save_checkpoint(path, epoch, model, optimizer, scheduler, history, cfg):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "config": cfg
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    return state["epoch"], state.get("history", [])


def run_training(cfg, train_loader, test_loader, device, epochs, label, optimizer_name=None, scheduler_name=None, accumulation_steps=1, checkpoint_path=None, resume_path=None, scheduler_epochs=None):
    seed_everything(cfg["project"]["seed"])
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg, optimizer_name)
    scheduler = build_scheduler(optimizer, cfg, scheduler_name, scheduler_epochs or epochs)
    start_epoch = 0
    history = []
    if resume_path and Path(resume_path).exists():
        start_epoch, history = load_checkpoint(resume_path, model, optimizer, scheduler, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak_memory = None
    for epoch in range(start_epoch, epochs):
        stats = train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        current_scheduler = scheduler_name or cfg["training"]["scheduler"]
        if current_scheduler == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": stats["loss"],
            "train_accuracy": stats["accuracy"],
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "throughput": stats["throughput"],
            "epoch_seconds": stats["elapsed"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(row)
        if stats["peak_memory_mb"] is not None:
            peak_memory = stats["peak_memory_mb"] if peak_memory is None else max(peak_memory, stats["peak_memory_mb"])
        mlflow.log_metrics({
            "train_loss": row["train_loss"],
            "train_accuracy": row["train_accuracy"],
            "val_loss": row["val_loss"],
            "val_accuracy": row["val_accuracy"],
            "throughput": row["throughput"],
            "lr": row["lr"]
        }, step=epoch + 1)
        if checkpoint_path:
            save_checkpoint(checkpoint_path, epoch + 1, model, optimizer, scheduler, history, cfg)
        print(f"{label} | epoch {epoch + 1}/{epochs} | loss {row['train_loss']:.4f} | train {row['train_accuracy'] * 100:.2f}% | test {row['val_accuracy'] * 100:.2f}% | {row['epoch_seconds']:.1f}s", flush=True)
    return model, history, peak_memory


def save_history(history, path):
    pd.DataFrame(history).to_csv(path, index=False)


def task1_pipeline(cfg, train_set, test_set, device, out):
    print("\nTASK 1/8 | YAML pipeline + MLflow + checkpoint", flush=True)
    train_loader, test_loader = build_loaders(cfg, train_set, test_set)
    checkpoint = out / "task1_checkpoint.pt"
    history_path = out / "task1_history.csv"
    with mlflow.start_run(run_name="task1_pipeline"):
        mlflow.log_params(flatten_dict(cfg))
        model, history, peak = run_training(
            cfg,
            train_loader,
            test_loader,
            device,
            cfg["training"]["epochs"],
            "Task 1",
            accumulation_steps=cfg["training"]["accumulation_steps"],
            checkpoint_path=checkpoint
        )
        save_history(history, history_path)
        model_path = out / "task1_model_state.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(CONFIG_PATH))
        mlflow.log_artifact(str(checkpoint))
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(model_path))
        if peak is not None:
            mlflow.log_metric("peak_memory_mb", peak)


def task2_profile(cfg, train_set, device, out):
    print("\nTASK 2/8 | torch.profiler one epoch", flush=True)
    loader, _ = build_loaders(cfg, train_set, train_set, batch_size=cfg["data"]["batch_size"], num_workers=training_workers(cfg))
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    iterator = iter(loader)
    with profile(activities=activities, record_shapes=False, profile_memory=True) as prof:
        for _ in range(len(loader)):
            with record_function("data_loading"):
                x, y = next(iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with record_function("forward_pass"):
                logits = model(x)
                loss = criterion(logits, y)
            with record_function("backward_pass"):
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()
    names = ["data_loading", "forward_pass", "backward_pass", "optimizer_step"]
    values = {name: 0.0 for name in names}
    for event in prof.key_averages():
        if event.key in values:
            values[event.key] += event.cpu_time_total / 1000.0
    total = sum(values.values()) or 1.0
    rows = [{"stage": name, "time_ms": values[name], "share_percent": 100 * values[name] / total} for name in names]
    df = pd.DataFrame(rows)
    csv_path = out / "task2_profile_breakdown.csv"
    trace_path = out / "task2_profiler_trace.json"
    plot_path = out / "task2_profile_breakdown.png"
    df.to_csv(csv_path, index=False)
    prof.export_chrome_trace(str(trace_path))
    plt.figure(figsize=(8, 5))
    plt.bar(df["stage"], df["share_percent"])
    plt.ylabel("Share of measured time (%)")
    plt.title("Task 2 Profiler Breakdown")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(df.to_string(index=False), flush=True)
    with mlflow.start_run(run_name="task2_profile"):
        for row in rows:
            mlflow.log_metric(f"share_{row['stage']}", row["share_percent"])
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(trace_path))
        mlflow.log_artifact(str(plot_path))


def task3_workers(cfg, train_set, out):
    print("\nTASK 3/8 | num_workers study", flush=True)
    rows = []
    batch_size = cfg["data"]["batch_size"]
    max_batches = cfg["studies"]["worker_benchmark_batches"]
    for workers in cfg["studies"]["num_workers"]:
        g = torch.Generator().manual_seed(cfg["project"]["seed"])
        kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": seed_worker,
            "generator": g
        }
        if workers > 0:
            kwargs["persistent_workers"] = False
            kwargs["prefetch_factor"] = 2
        loader = DataLoader(train_set, **kwargs)
        iterator = iter(loader)
        warmup = min(2, len(loader))
        for _ in range(warmup):
            next(iterator)
        start = time.perf_counter()
        samples = 0
        measured = 0
        for _ in range(min(max_batches, max(0, len(loader) - warmup))):
            x, _ = next(iterator)
            samples += x.size(0)
            measured += 1
        elapsed = time.perf_counter() - start
        throughput = samples / elapsed if elapsed > 0 else 0.0
        rows.append({
            "num_workers": workers,
            "batches": measured,
            "samples": samples,
            "seconds": elapsed,
            "samples_per_second": throughput
        })
        print(f"Task 3 | workers={workers} | {throughput:.1f} samples/s", flush=True)
        del iterator
        del loader
    df = pd.DataFrame(rows)
    csv_path = out / "task3_num_workers.csv"
    plot_path = out / "task3_num_workers.png"
    df.to_csv(csv_path, index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(df["num_workers"], df["samples_per_second"], marker="o")
    plt.xlabel("num_workers")
    plt.ylabel("Data throughput (samples/s)")
    plt.title("Task 3 DataLoader Worker Study")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    best = df.loc[df["samples_per_second"].idxmax()]
    conclusion_path = out / "task3_bottleneck.txt"
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write(f"Best num_workers: {int(best['num_workers'])}\n")
        f.write(f"Best data throughput: {best['samples_per_second']:.2f} samples/s\n")
        f.write("If throughput stops improving or decreases as workers increase, worker startup, IPC, transform overhead, or CPU contention is the bottleneck.\n")
    with mlflow.start_run(run_name="task3_workers"):
        mlflow.log_metric("best_num_workers", float(best["num_workers"]))
        mlflow.log_metric("best_samples_per_second", float(best["samples_per_second"]))
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(conclusion_path))


def task4_batch_size(cfg, train_set, test_set, device, out):
    print("\nTASK 4/8 | batch-size study", flush=True)
    rows = []
    epochs = cfg["studies"]["batch_study_epochs"]
    for batch_size in cfg["studies"]["batch_sizes"]:
        train_loader, test_loader = build_loaders(cfg, train_set, test_set, batch_size=batch_size, num_workers=training_workers(cfg))
        run_name = f"task4_batch_{batch_size}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"batch_size": batch_size, "epochs": epochs})
            _, history, peak = run_training(cfg, train_loader, test_loader, device, epochs, f"Task 4 batch={batch_size}")
            hist_path = out / f"{run_name}.csv"
            save_history(history, hist_path)
            mlflow.log_artifact(str(hist_path))
            rows.append({
                "batch_size": batch_size,
                "throughput_samples_s": sum(x["throughput"] for x in history) / len(history),
                "peak_memory_mb": peak,
                "final_accuracy": history[-1]["val_accuracy"]
            })
    df = pd.DataFrame(rows)
    csv_path = out / "task4_batch_size_study.csv"
    acc_plot = out / "task4_batch_accuracy.png"
    speed_plot = out / "task4_batch_throughput.png"
    df.to_csv(csv_path, index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(df["batch_size"], df["final_accuracy"], marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Final test accuracy")
    plt.title("Task 4 Batch Size vs Accuracy")
    plt.tight_layout()
    plt.savefig(acc_plot, dpi=150)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(df["batch_size"], df["throughput_samples_s"], marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (samples/s)")
    plt.title("Task 4 Batch Size vs Throughput")
    plt.tight_layout()
    plt.savefig(speed_plot, dpi=150)
    plt.close()
    with mlflow.start_run(run_name="task4_summary"):
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(acc_plot))
        mlflow.log_artifact(str(speed_plot))


def task5_optimizers_schedulers(cfg, train_set, test_set, device, out):
    print("\nTASK 5/8 | optimizer + scheduler study", flush=True)
    rows = []
    epochs = cfg["studies"]["schedule_study_epochs"]
    for optimizer_name in cfg["studies"]["optimizers"]:
        for scheduler_name in cfg["studies"]["schedulers"]:
            train_loader, test_loader = build_loaders(cfg, train_set, test_set, num_workers=training_workers(cfg))
            run_name = f"task5_{optimizer_name}_{scheduler_name}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "epochs": epochs,
                    "batch_size": cfg["data"]["batch_size"]
                })
                _, history, peak = run_training(
                    cfg,
                    train_loader,
                    test_loader,
                    device,
                    epochs,
                    f"Task 5 {optimizer_name}+{scheduler_name}",
                    optimizer_name=optimizer_name,
                    scheduler_name=scheduler_name
                )
                hist_path = out / f"{run_name}.csv"
                save_history(history, hist_path)
                mlflow.log_artifact(str(hist_path))
                rows.append({
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "final_accuracy": history[-1]["val_accuracy"],
                    "final_loss": history[-1]["val_loss"],
                    "avg_throughput": sum(x["throughput"] for x in history) / len(history),
                    "peak_memory_mb": peak
                })
    df = pd.DataFrame(rows)
    csv_path = out / "task5_optimizer_scheduler.csv"
    plot_path = out / "task5_optimizer_scheduler.png"
    df.to_csv(csv_path, index=False)
    labels = df["optimizer"] + " + " + df["scheduler"]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, df["final_accuracy"])
    plt.ylabel("Final test accuracy")
    plt.title("Task 5 Optimizer and Scheduler Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    with mlflow.start_run(run_name="task5_summary"):
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(plot_path))


def task6_accumulation(cfg, train_set, test_set, device, out):
    print("\nTASK 6/8 | gradient accumulation", flush=True)
    epochs = cfg["studies"]["accumulation_epochs"]
    results = {}
    for name, batch_size, steps in [("batch256_acc1", 256, 1), ("batch64_acc4", 64, 4)]:
        train_loader, test_loader = build_loaders(cfg, train_set, test_set, batch_size=batch_size, num_workers=training_workers(cfg))
        with mlflow.start_run(run_name=f"task6_{name}"):
            mlflow.log_params({
                "batch_size": batch_size,
                "accumulation_steps": steps,
                "effective_batch_size": batch_size * steps,
                "epochs": epochs
            })
            _, history, peak = run_training(cfg, train_loader, test_loader, device, epochs, f"Task 6 {name}", accumulation_steps=steps)
            results[name] = history
            hist_path = out / f"task6_{name}.csv"
            save_history(history, hist_path)
            mlflow.log_artifact(str(hist_path))
            if peak is not None:
                mlflow.log_metric("peak_memory_mb", peak)
    rows = []
    for epoch in range(epochs):
        a = results["batch256_acc1"][epoch]
        b = results["batch64_acc4"][epoch]
        rows.append({
            "epoch": epoch + 1,
            "batch256_accuracy": a["val_accuracy"],
            "batch64_acc4_accuracy": b["val_accuracy"],
            "accuracy_difference": abs(a["val_accuracy"] - b["val_accuracy"]),
            "batch256_loss": a["val_loss"],
            "batch64_acc4_loss": b["val_loss"]
        })
    df = pd.DataFrame(rows)
    csv_path = out / "task6_accumulation_comparison.csv"
    plot_path = out / "task6_accumulation_comparison.png"
    df.to_csv(csv_path, index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["batch256_accuracy"], marker="o", label="batch 256")
    plt.plot(df["epoch"], df["batch64_acc4_accuracy"], marker="o", label="batch 64 x accumulation 4")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Task 6 Gradient Accumulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    with mlflow.start_run(run_name="task6_summary"):
        mlflow.log_metric("mean_accuracy_difference", float(df["accuracy_difference"].mean()))
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(plot_path))


def task7_resume(cfg, train_set, test_set, device, out):
    print("\nTASK 7/8 | interrupt at epoch 3 + resume", flush=True)
    checkpoint = out / "task7_resume_checkpoint.pt"
    total_epochs = cfg["studies"]["resume_total_epochs"]
    train_loader, test_loader = build_loaders(cfg, train_set, test_set, num_workers=training_workers(cfg))
    with mlflow.start_run(run_name="task7_interrupt_epoch3"):
        mlflow.log_params({"stop_epoch": 3, "planned_total_epochs": total_epochs})
        _, first_history, _ = run_training(
            cfg,
            train_loader,
            test_loader,
            device,
            3,
            "Task 7 before interrupt",
            checkpoint_path=checkpoint,
            scheduler_epochs=total_epochs
        )
        first_path = out / "task7_before_resume.csv"
        save_history(first_history, first_path)
        mlflow.log_artifact(str(first_path))
        mlflow.log_artifact(str(checkpoint))
    train_loader, test_loader = build_loaders(cfg, train_set, test_set, num_workers=training_workers(cfg))
    with mlflow.start_run(run_name="task7_resume_training"):
        mlflow.log_params({"resume_from_epoch": 3, "total_epochs": total_epochs})
        _, resumed_history, _ = run_training(
            cfg,
            train_loader,
            test_loader,
            device,
            total_epochs,
            "Task 7 resumed",
            checkpoint_path=checkpoint,
            resume_path=checkpoint,
            scheduler_epochs=total_epochs
        )
        resumed_path = out / "task7_resumed_history.csv"
        plot_path = out / "task7_resume_curve.png"
        save_history(resumed_history, resumed_path)
        df = pd.DataFrame(resumed_history)
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["val_accuracy"], marker="o", label="test accuracy")
        plt.axvline(3, linestyle="--", label="resume point")
        plt.xlabel("Epoch")
        plt.ylabel("Test accuracy")
        plt.title("Task 7 Resume From Checkpoint")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(resumed_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(checkpoint))


def task8_proposal(cfg, out):
    print("\nTASK 8/8 | project proposal", flush=True)
    p = cfg["proposal"]
    project = cfg["project"]
    proposal = f"""# Individual Project Proposal

Student name: {project['student_name']}

Student ID: {project['student_id']}

## Problem

{p['problem']}

## Dataset

{p['dataset']}

## Success metric

{p['success_metric']}

## Deployment constraints

Latency must be at most {p['latency_constraint_ms']} ms per sample and model size must be at most {p['size_constraint_mb']} MB.

## Planned optimization

{p['planned_optimization']}

## Reproducibility and evaluation plan

Training uses a YAML-driven configuration, a fixed random seed, checkpointing, MLflow experiment tracking, throughput and memory measurements, and controlled comparisons of batch size, worker count, optimizer, and learning-rate scheduling. Fast mode uses one deterministic subset for every controlled comparison so the experiments remain directly comparable while finishing faster on a CPU. The final project version can disable fast mode to repeat the same pipeline on the complete Fashion-MNIST training and test sets.
"""
    path = out / "task8_project_proposal.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(proposal)
    with mlflow.start_run(run_name="task8_project_proposal"):
        mlflow.log_artifact(str(path))
    try:
        if Path(".git").exists():
            subprocess.run(["git", "add", str(path), str(CONFIG_PATH)], check=False)
            subprocess.run(["git", "commit", "-m", "Add Lab 6 project proposal and training pipeline config"], check=False)
    except Exception:
        pass


def make_summary(cfg, train_set, test_set, out):
    scope_path = out / "experiment_scope.txt"
    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(f"Fast mode: {cfg['data']['fast_mode']}\n")
        f.write(f"Training samples used per experiment: {len(train_set)}\n")
        f.write(f"Test samples used per experiment: {len(test_set)}\n")
        f.write("Required epoch budgets and study value lists are unchanged.\n")
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    path = out / "graded_evidence.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("Generated Lab 6 evidence:\n")
        for name in files:
            f.write(f"{name}\n")


def main():
    cfg = load_config()
    seed_everything(cfg["project"]["seed"])
    out = Path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    db_path = (out / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(cfg["project"]["name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"MLflow database: {db_path}", flush=True)
    train_set, test_set = build_datasets(cfg)
    print(f"Fast mode: {cfg['data']['fast_mode']} | train samples: {len(train_set)} | test samples: {len(test_set)}", flush=True)
    task1_pipeline(cfg, train_set, test_set, device, out)
    task2_profile(cfg, train_set, device, out)
    task3_workers(cfg, train_set, out)
    task4_batch_size(cfg, train_set, test_set, device, out)
    task5_optimizers_schedulers(cfg, train_set, test_set, device, out)
    task6_accumulation(cfg, train_set, test_set, device, out)
    task7_resume(cfg, train_set, test_set, device, out)
    task8_proposal(cfg, out)
    make_summary(cfg, train_set, test_set, out)
    print("\nDONE", flush=True)
    print(f"Results: {out.resolve()}", flush=True)
    print(f"MLflow UI: mlflow ui --backend-store-uri sqlite:///{db_path.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
