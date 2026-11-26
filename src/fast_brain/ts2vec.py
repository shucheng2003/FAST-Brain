import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def generate_continuous_mask(b, t, n=5, l=0.1):
    out = torch.full((b, t), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * t)
    n = max(min(n, t // 2), 1)
    if isinstance(l, float):
        l = int(l * t)
    l = max(l, 1)
    for i in range(b):
        for _ in range(n):
            start = np.random.randint(t - l + 1)
            out[i, start:start + l] = False
    return out


def generate_binomial_mask(b, t, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(b, t))).to(torch.bool)


def take_per_row(x, indices, count):
    all_indices = indices[:, None] + np.arange(count)
    return x[torch.arange(all_indices.shape[0])[:, None], all_indices]


def pad_nan_to_target(x, target_length, axis=0):
    pad_size = target_length - x.shape[axis]
    if pad_size <= 0:
        return x
    pads = [(0, 0)] * x.ndim
    pads[axis] = (0, pad_size)
    return np.pad(x, pad_width=pads, mode="constant", constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    parts = np.array_split(x, sections, axis=axis)
    target_length = parts[0].shape[axis]
    return [pad_nan_to_target(item, target_length, axis=axis) for item in parts]


def centerize_vary_length_series(x):
    left = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    right = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (left + right) // 2 - left
    rows, cols = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    cols = cols - offset[:, np.newaxis]
    return x[rows, cols]


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        field = (kernel_size - 1) * dilation + 1
        self.remove = 1 if field % 2 == 0 else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=field // 2, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(channels):
            layers.append(ConvBlock(channels[i - 1] if i > 0 else in_channels, out_channels, kernel_size, 2 ** i, i == len(channels) - 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode="binomial"):
        super().__init__()
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3)
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)
        if mask is None:
            mask = self.mask_mode if self.training else "all_true"
        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        mask &= nan_mask
        x[~mask] = 0
        x = x.transpose(1, 2)
        x = self.repr_dropout(self.feature_extractor(x))
        return x.transpose(1, 2)


def instance_contrastive_loss(z1, z2):
    b = z1.size(0)
    if b == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    idx = torch.arange(b, device=z1.device)
    return (logits[:, idx, b + idx - 1].mean() + logits[:, b + idx, idx].mean()) / 2


def temporal_contrastive_loss(z1, z2):
    t = z1.size(1)
    if t == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    idx = torch.arange(t, device=z1.device)
    return (logits[:, idx, t + idx - 1].mean() + logits[:, t + idx, idx].mean()) / 2


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0.0, device=z1.device)
    depth = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if depth >= temporal_unit and 1 - alpha != 0:
            loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        depth += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        depth += 1
    return loss / depth


class TS2Vec:
    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10, device="cuda", lr=0.001, batch_size=16, max_train_length=None, temporal_unit=0):
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
        missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if missing[0] or missing[-1]:
            train_data = centerize_vary_length_series(train_data)
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        loader = DataLoader(TensorDataset(torch.from_numpy(train_data).to(torch.float)), batch_size=min(self.batch_size, len(train_data)), shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        losses = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            total = 0.0
            seen = 0
            interrupted = False
            for batch in loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    start = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, start:start + self.max_train_length]
                x = x.to(self.device)
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                optimizer.zero_grad()
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                loss = hierarchical_contrastive_loss(out1, out2, temporal_unit=self.temporal_unit)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                total += loss.item()
                seen += 1
                self.n_iters += 1
            if interrupted:
                break
            losses.append(total / seen)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={losses[-1]}")
            self.n_epochs += 1
        return losses

    def encode(self, data, mask=None, encoding_window=None, batch_size=None):
        loader = DataLoader(TensorDataset(torch.from_numpy(data).to(torch.float)), batch_size=batch_size or self.batch_size)
        outputs = []
        with torch.no_grad():
            self.net.eval()
            for batch in loader:
                x = batch[0].to(self.device)
                out = self.net(x, mask)
                if encoding_window == "full_series":
                    out = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).transpose(1, 2)
                outputs.append(out.cpu())
        return torch.cat(outputs, dim=0).numpy()
