from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .config import project_root


@dataclass
class BrainData:
    name: str
    signals: np.ndarray
    all_x: np.ndarray
    all_y: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    eval_signal: np.ndarray
    graph_matrices: list
    real_ec: Optional[np.ndarray]
    split_index: int


class WindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def read_array(path, kind):
    full = project_root() / path
    if kind == "csv_header":
        return pd.read_csv(full).values.astype(np.float32)
    if kind == "csv":
        return pd.read_csv(full, header=None).values.astype(np.float32)
    return np.loadtxt(full, dtype=np.float32)


def standardize(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32)


def minmax_scale(x):
    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    return (2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)


def normalize_matrix(x, transform):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi > lo:
        x = (x - lo) / (hi - lo)
    if transform == "inverse_normalize":
        x = 1.0 - x
    np.fill_diagonal(x, 0.0)
    return x.astype(np.float32)


def make_windows(signals, history_len, pred_len):
    total = signals.shape[0] - history_len - pred_len + 1
    x = np.empty((total, history_len, signals.shape[1]), dtype=np.float32)
    y = np.empty((total, pred_len, signals.shape[1]), dtype=np.float32)
    for i in range(total):
        x[i] = signals[i:i + history_len]
        y[i] = signals[i + history_len:i + history_len + pred_len]
    return x, y


def load_graphs(cfg):
    graphs = []
    for item in cfg["data"]["matrices"]:
        mat = read_array(item["path"], item["format"])
        graphs.append(np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    return graphs


def load_hcp_sessions(cfg):
    roi_count = int(cfg["roi_count"])
    sessions = []
    drop_first = int(cfg["data"].get("drop_first", 0))
    for item in cfg["data"]["sessions"]:
        arr = read_array(item["path"], item["format"])
        arr = arr[drop_first:, :roi_count]
        sessions.append(standardize(arr))
    return sessions


def load_signals(cfg):
    name = cfg["dataset"]
    roi_count = int(cfg["roi_count"])
    if name == "hcp":
        return np.vstack(load_hcp_sessions(cfg)).astype(np.float32)
    item = cfg["data"]["signals"]
    arr = read_array(item["path"], item["format"])
    if name == "sddes" and arr.shape[1] > roi_count:
        arr = arr[:, -roi_count:]
    else:
        arr = arr[:, :roi_count]
    return minmax_scale(arr.astype(np.float32))


def load_data(cfg, max_train_samples=None):
    history_len = int(cfg["history_len"])
    pred_len = int(cfg["pred_len"])
    if cfg["dataset"] == "hcp":
        sessions = load_hcp_sessions(cfg)
        windows = [make_windows(item, history_len, pred_len) for item in sessions]
        x = np.concatenate([item[0] for item in windows], axis=0)
        y = np.concatenate([item[1] for item in windows], axis=0)
        signals = np.vstack(sessions).astype(np.float32)
    else:
        signals = load_signals(cfg)
        x, y = make_windows(signals, history_len, pred_len)
    if max_train_samples is not None:
        x = x[:max_train_samples]
        y = y[:max_train_samples]
    n_train = int(float(cfg["train_fraction"]) * len(x))
    train_x = x[:n_train]
    train_y = y[:n_train]
    real_ec = None
    if "real_ec" in cfg["data"]:
        item = cfg["data"]["real_ec"]
        real_ec = read_array(item["path"], item["format"])
    split_index = int(float(cfg["train_fraction"]) * len(x)) + history_len
    split_index = max(split_index, history_len)
    split_index = min(split_index, len(signals) - pred_len)
    return BrainData(
        name=cfg["dataset"],
        signals=signals,
        all_x=x,
        all_y=y,
        train_x=train_x,
        train_y=train_y,
        eval_signal=signals,
        graph_matrices=load_graphs(cfg),
        real_ec=real_ec,
        split_index=split_index
    )
