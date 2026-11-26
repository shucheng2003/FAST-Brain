import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .config import merged_config, project_root
from .data import WindowDataset, load_data
from .metrics import bootstrap_mean_ci, conditional_metrics, conditional_state, corr_mae, ec_binary_metrics, ec_jacobian, rollout, safe_corrcoef, sddes_reference_side_effects, sddes_reference_state
from .model import FastBrainModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rnn", "sddes", "hcp"], required=True)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--history-len", type=int)
    p.add_argument("--pred-len", type=int)
    p.add_argument("--euler-steps", type=int)
    p.add_argument("--use-rope", dest="use_rope", action="store_true", default=None)
    p.add_argument("--no-rope", dest="use_rope", action="store_false")
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-steps", type=int)
    p.add_argument("--num-prob-samples", type=int)
    p.add_argument("--conditional-blocks", type=int)
    p.add_argument("--num-runs", type=int)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, data, cfg, device):
    tr = cfg["training"]
    split = int(float(cfg["train_fraction"]) * data.all_x.shape[0])
    train_dataset = WindowDataset(data.all_x[:split], data.all_y[:split])
    test_dataset = WindowDataset(data.all_x[split:], data.all_y[split:])
    loader = DataLoader(train_dataset, batch_size=int(tr["batch_size"]), shuffle=bool(tr["shuffle"]))
    test_loader = DataLoader(test_dataset, batch_size=int(tr["batch_size"]), shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    scheduler = None
    if bool(tr["scheduler"]):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(tr["epochs"]), 1), eta_min=float(tr["eta_min"]))
    losses = []
    for _ in range(int(tr["epochs"])):
        model.train()
        total = 0.0
        seen = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss = model.loss(x, y, float(tr["mu_t"]), float(tr["sigma_t"]))
            opt.zero_grad()
            loss.backward()
            if bool(tr["grad_clip"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * x.shape[0]
            seen += x.shape[0]
        if scheduler is not None:
            scheduler.step()
        losses.append(total / max(seen, 1))
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                b = x.shape[0]
                eps = torch.randn_like(y)
                t = torch.rand(b, device=device)
                tau = t.view(b, 1, 1)
                z = tau * y + (1.0 - tau) * eps
                out = model.network_output(z, t, x)
                clean = model.clean_from_output(out, z, t)
                v = (clean - z) / (1.0 - tau).clamp(min=0.01)
                F.mse_loss(v, y - eps)
    return losses


def evaluation_steps(data, cfg, args):
    history_len = int(cfg["history_len"])
    pred_len = int(cfg["pred_len"])
    split_index = data.split_index
    configured = cfg["evaluation"].get("pred_steps")
    if configured is None:
        total_steps = len(data.eval_signal) - split_index
    else:
        total_steps = int(configured)
    if args.max_eval_steps is not None:
        total_steps = min(total_steps, args.max_eval_steps)
    elif args.max_train_samples is not None:
        total_steps = min(total_steps, pred_len * 2)
    return max(pred_len, total_steps)


def evaluate_once(model, data, cfg, args, device, cond_state=None, sddes_state=None):
    history_len = int(cfg["history_len"])
    pred_len = int(cfg["pred_len"])
    split_index = data.split_index
    total_steps = evaluation_steps(data, cfg, args)
    initial = data.eval_signal[split_index - history_len:split_index]
    generated = rollout(model, initial, total_steps, device)
    target_fc = safe_corrcoef(data.signals)
    model_fc = safe_corrcoef(generated)
    fc_corr, fc_mae = corr_mae(model_fc, target_fc)
    out = {"fc_corr": fc_corr, "fc_mae": fc_mae, "eval_steps": int(total_steps)}
    if data.name == "sddes" and sddes_state is not None:
        sddes_reference_side_effects(
            model,
            generated,
            sddes_state,
            int(cfg["evaluation"]["conditional_samples"]),
            int(cfg["training"]["batch_size"]),
            pred_len,
            int(cfg["roi_count"]),
            device
        )
    if data.name == "rnn":
        max_ec = int(cfg["evaluation"]["ec_samples"])
        if args.max_train_samples is not None:
            max_ec = min(max_ec, args.max_train_samples)
        model.train()
        raw_ec = ec_jacobian(model, data.all_x, device, max_ec)
        ec_auroc, ec_auprc = ec_binary_metrics(raw_ec, data.graph_matrices[0], float(cfg["evaluation"]["ec_sc_threshold"]))
        ec = np.asarray(raw_ec, dtype=np.float64).copy()
        percentile = cfg["evaluation"].get("ec_sparsity_percentile")
        if percentile is not None:
            threshold = np.percentile(np.abs(ec), float(percentile))
            ec[np.abs(ec) < threshold] = 0.0
            np.fill_diagonal(ec, 0.0)
        ec_corr, ec_mae = corr_mae(ec, data.real_ec)
        out["ec_corr"] = ec_corr
        out["ec_mae"] = ec_mae
        out["ec_auroc"] = ec_auroc
        out["ec_auprc"] = ec_auprc
        samples = int(cfg["evaluation"]["conditional_samples"])
        out.update(conditional_metrics(model, samples, device, cond_state, min(int(cfg["training"]["batch_size"]), len(cond_state["contexts"]))))
    return out


def evaluate(model, data, cfg, args, device):
    runs = int(cfg["evaluation"].get("num_runs", 1))
    if args.max_train_samples is not None and args.num_runs is None:
        runs = 1
    cond_state = None
    sddes_state = None
    if data.name == "rnn":
        history_len = int(cfg["history_len"])
        pred_len = int(cfg["pred_len"])
        total_steps = evaluation_steps(data, cfg, args)
        blocks = int(cfg["evaluation"].get("conditional_blocks") or (total_steps // pred_len))
        if args.max_train_samples is not None:
            blocks = min(blocks, 2)
        cond_state = conditional_state(data.signals, data.split_index, history_len, pred_len, blocks, int(cfg["roi_count"]), device)
    if data.name == "sddes":
        history_len = int(cfg["history_len"])
        pred_len = int(cfg["pred_len"])
        total_steps = evaluation_steps(data, cfg, args)
        sddes_state = sddes_reference_state(data.signals, data.split_index, history_len, pred_len, total_steps, int(cfg["roi_count"]), device)
    values = [evaluate_once(model, data, cfg, args, device, cond_state, sddes_state) for _ in range(runs)]
    out = {}
    ci_bootstrap = int(cfg["evaluation"].get("ci_bootstrap", 1000))
    ci_level = float(cfg["evaluation"].get("ci_level", 0.95))
    seed = int(cfg.get("seed", 0))
    for idx, key in enumerate(sorted({k for value in values for k in value})):
        arr = np.array([value[key] for value in values if key in value], dtype=float)
        out[key] = float(np.nanmean(arr))
        if runs > 1 and key != "eval_steps":
            out[f"{key}_std"] = float(np.nanstd(arr, ddof=0))
            ci_low, ci_high = bootstrap_mean_ci(arr, ci_bootstrap, ci_level, seed + idx)
            out[f"{key}_ci_low"] = ci_low
            out[f"{key}_ci_high"] = ci_high
    out["eval_steps"] = int(round(out["eval_steps"]))
    out["num_runs"] = runs
    return out


def write_records(records, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    keys = sorted({key for record in records for key in record})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def run_one(seed, cfg, args, device):
    set_seed(seed)
    data = load_data(cfg, args.max_train_samples)
    model = FastBrainModel(
        roi_count=int(cfg["roi_count"]),
        history_len=int(cfg["history_len"]),
        pred_len=int(cfg["pred_len"]),
        matrices=data.graph_matrices,
        cfg=cfg["model"]
    ).to(device)
    losses = train_model(model, data, cfg, device)
    metrics = evaluate(model, data, cfg, args, device)
    record = {
        "dataset": args.dataset,
        "variant": "fast",
        "seed": seed,
        "train_loss": losses[-1] if losses else None
    }
    record.update(metrics)
    return record, model


def main():
    args = parse_args()
    cfg = merged_config(args.dataset, args)
    device = torch.device(args.device)
    out_dir = project_root() / args.results_dir / args.dataset / "fast"
    records = []
    last_model = None
    seeds = args.seeds if args.seeds is not None else [int(cfg.get("seed", 0))]
    for seed in seeds:
        record, last_model = run_one(seed, cfg, args, device)
        records.append(record)
        print(json.dumps(record, sort_keys=True))
    write_records(records, out_dir)
    if args.save_checkpoint and last_model is not None:
        torch.save(last_model.state_dict(), out_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
