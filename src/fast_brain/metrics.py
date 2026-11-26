import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from scipy.stats import ks_2samp, pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score
from .ts2vec import TS2Vec


def safe_corrcoef(x):
    c = np.corrcoef(np.asarray(x), rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def offdiag(x):
    mask = ~np.eye(x.shape[0], dtype=bool)
    return x[mask]


def corr_mae(pred, target):
    a = offdiag(pred)
    b = offdiag(target)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        r = 0.0
    else:
        r = float(pearsonr(a, b)[0])
    return r, float(np.mean(np.abs(a - b)))


def ec_binary_metrics(pred, sc, threshold):
    a = np.asarray(pred, dtype=np.float64).copy()
    b = np.asarray(sc, dtype=np.float64).copy()
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(b, 0.0)
    mask = ~np.eye(a.shape[0], dtype=bool)
    labels = (b[mask] > threshold).astype(np.int32)
    scores = np.abs(a[mask])
    if np.unique(labels).size < 2:
        return np.nan, np.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def bootstrap_mean_ci(values, n_bootstrap, ci_level, seed):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, arr.size, size=arr.size)
        vals.append(np.mean(arr[idx]))
    low = 100.0 * (1.0 - ci_level) / 2.0
    high = 100.0 - low
    return tuple(float(x) for x in np.percentile(vals, [low, high]))


def rollout(model, initial_history, total_steps, device):
    was_training = model.training
    model.train()
    history = initial_history.astype(np.float32).copy()
    h = history.shape[0]
    out = []
    try:
        with torch.no_grad():
            while len(out) < total_steps:
                x = torch.tensor(history[None], dtype=torch.float32, device=device)
                y = model.sample(x)[0].detach().cpu().numpy()
                out.extend(list(y))
                history = np.concatenate([history, y], axis=0)[-h:]
    finally:
        model.train(was_training)
    return np.asarray(out[:total_steps], dtype=np.float32)


def ec_jacobian(model, contexts, device, max_samples):
    count = min(max_samples, contexts.shape[0])
    indices = np.random.choice(contexts.shape[0], count, replace=False)
    x = torch.tensor(contexts[indices], dtype=torch.float32, device=device, requires_grad=True)
    pred = model.clean_prediction_for_ec(x)[:, 0, :]
    n = pred.shape[1]
    ec = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        grad = torch.autograd.grad(pred[:, i], x, grad_outputs=torch.ones_like(pred[:, i]), retain_graph=True)[0]
        ec[i] = grad[:, -1, :].mean(dim=0).detach().cpu().numpy()
    np.fill_diagonal(ec, 0.0)
    return ec


def empirical_crps(pred_samples, true_values):
    preds = np.asarray(pred_samples, dtype=np.float64).reshape(-1, pred_samples.shape[-1])
    targets = np.asarray(true_values, dtype=np.float64).reshape(-1, 1)
    term1 = np.mean(np.abs(preds - targets), axis=1)
    sorted_preds = np.sort(preds, axis=1)
    s = sorted_preds.shape[1]
    coeff = (2.0 * np.arange(1, s + 1) - s - 1.0)[None, :]
    term2 = np.sum(sorted_preds * coeff, axis=1) / float(s * s)
    return float(np.mean(term1 - term2))


def qice(pred_samples, true_values, bins=10):
    preds = np.asarray(pred_samples, dtype=np.float64).reshape(-1, pred_samples.shape[-1])
    targets = np.asarray(true_values, dtype=np.float64).reshape(-1)
    qs = np.arange(bins + 1) * (100.0 / bins)
    quantiles = np.percentile(preds, q=qs, axis=1)
    membership = ((targets[None, :] - quantiles) > 0).astype(np.int32).sum(axis=0)
    counts = np.array([(membership == v).sum() for v in np.arange(bins + 2)], dtype=np.float64)
    counts[1] += counts[0]
    counts[-2] += counts[-1]
    ratio = counts[1:-1] / max(len(targets), 1)
    return float(np.mean(np.abs(ratio - 1.0 / bins)))


def probcorr(pred_samples, true_values):
    preds = np.asarray(pred_samples, dtype=np.float64)
    truth = np.asarray(true_values, dtype=np.float64)
    scores = []
    for b in range(truth.shape[0]):
        target_corr = safe_corrcoef(truth[b])
        sample_scores = []
        for s in range(preds.shape[-1]):
            sample_scores.append(np.mean(np.abs(safe_corrcoef(preds[b, :, :, s]) - target_corr)))
        scores.append(np.mean(sample_scores))
    return float(np.mean(scores))


class MetricEvaluator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)
        self.pred_out = nn.Linear(hidden_dim, input_dim)

    def forward_disc(self, x):
        _, h = self.rnn(x)
        return torch.sigmoid(self.out(h.squeeze(0)))

    def forward_pred(self, x):
        out, _ = self.rnn(x)
        return self.pred_out(out[:, -1, :])

    def get_embedding(self, x):
        _, h = self.rnn(x)
        return h.squeeze(0)


def split_sequences(data_seq, window_len):
    samples = []
    for i in range(0, len(data_seq) - window_len + 1, window_len):
        samples.append(data_seq[i:i + window_len])
    return np.asarray(samples)


def calculate_cross_correlation(data_tensor):
    vals = []
    data_np = data_tensor.detach().cpu().numpy() if torch.is_tensor(data_tensor) else data_tensor
    for i in range(len(data_np)):
        c = np.corrcoef(data_np[i], rowvar=False)
        if not np.isnan(c).any():
            vals.append(c)
    return np.mean(vals, axis=0)


def compute_fcd(time_series, window_size=40, step=1):
    t, n = time_series.shape
    count = (t - window_size) // step + 1
    values = []
    triu = np.triu_indices(n, k=1)
    for w in range(count):
        start = w * step
        end = start + window_size
        fc = np.corrcoef(time_series[start:end], rowvar=False)
        fc = np.nan_to_num(fc, nan=0.0)
        values.append(fc[triu])
    out = np.corrcoef(np.asarray(values))
    return np.nan_to_num(out, nan=0.0)


def sequence_features(x):
    x = np.asarray(x, dtype=np.float64)
    means = x.mean(axis=1)
    stds = x.std(axis=1)
    mins = x.min(axis=1)
    maxs = x.max(axis=1)
    return np.concatenate([means, stds, mins, maxs], axis=1)


def fid(a, b):
    def cov(x):
        if x.shape[0] < 2:
            return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        return np.cov(x, rowvar=False)
    mu1, sig1 = a.mean(axis=0), cov(a)
    mu2, sig2 = b.mean(axis=0), cov(b)
    sig1 = np.atleast_2d(np.nan_to_num(sig1, nan=0.0, posinf=0.0, neginf=0.0))
    sig2 = np.atleast_2d(np.nan_to_num(sig2, nan=0.0, posinf=0.0, neginf=0.0))
    eps = 1e-6
    sig1 = sig1 + np.eye(sig1.shape[0]) * eps
    sig2 = sig2 + np.eye(sig2.shape[0]) * eps
    covmean = linalg.sqrtm(sig1 @ sig2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        covmean = np.zeros_like(sig1)
    val = np.sum((mu1 - mu2) ** 2) + np.trace(sig1 + sig2 - 2.0 * covmean)
    return float(np.real(val))


def conditional_blocks(signals, split_index, history_len, pred_len, max_blocks):
    contexts = []
    futures = []
    available = len(signals) - split_index
    blocks = available // pred_len
    if max_blocks:
        blocks = min(blocks, max_blocks)
    for i in range(blocks):
        start = split_index + i * pred_len
        c = signals[start - history_len:start]
        f = signals[start:start + pred_len]
        if len(c) == history_len and len(f) == pred_len:
            contexts.append(c)
            futures.append(f)
    return np.asarray(contexts, dtype=np.float32), np.asarray(futures, dtype=np.float32)


def conditional_samples(model, contexts, samples, device, eval_batch_size=None):
    outputs = []
    was_training = model.training
    model.train()
    eval_batch_size = eval_batch_size or len(contexts)
    try:
        with torch.no_grad():
            for start in range(0, len(contexts), eval_batch_size):
                end = min(start + eval_batch_size, len(contexts))
                x = torch.tensor(contexts[start:end], dtype=torch.float32, device=device)
                flat = x.reshape(end - start, -1).repeat_interleave(samples, dim=0)
                y = model(flat).view(end - start, samples, model.pred_len, model.roi_count)
                outputs.append(y.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(outputs, axis=0)


def conditional_state(signals, split_index, history_len, pred_len, blocks, input_dims, device):
    contexts, futures = conditional_blocks(signals, split_index, history_len, pred_len, blocks)
    real_seq = np.concatenate([contexts, futures], axis=1).astype(np.float32)
    ts2vec = TS2Vec(input_dims=input_dims, device=device, batch_size=min(128, len(real_seq)), lr=0.001, output_dims=320, max_train_length=3000)
    ts2vec.fit(real_seq, verbose=False)
    real_repr = ts2vec.encode(real_seq, encoding_window="full_series", batch_size=min(128, len(real_seq))).reshape(len(real_seq), -1)
    return {"contexts": contexts, "futures": futures, "ts2vec": ts2vec, "real_repr": real_repr}


def conditional_metrics(model, samples, device, state, eval_batch_size=None):
    contexts = state["contexts"]
    futures = state["futures"]
    preds = conditional_samples(model, contexts, samples, device, eval_batch_size)
    b, p, n, s = preds.shape
    repeated_context = np.repeat(contexts, s, axis=0)
    sampled_future = np.transpose(preds, (0, 3, 1, 2)).reshape(b * s, p, n)
    generated = np.concatenate([repeated_context, sampled_future], axis=1).astype(np.float32)
    gen_repr = state["ts2vec"].encode(generated, encoding_window="full_series", batch_size=min(128, len(generated))).reshape(len(generated), -1)
    return {
        "crps": empirical_crps(preds, futures),
        "qice": qice(preds, futures),
        "probcorr": probcorr(preds, futures),
        "conditional_fid": fid(state["real_repr"], gen_repr)
    }


def sddes_reference_state(signals, split_index, history_len, pred_len, total_steps, input_dims, device):
    blocks = total_steps // pred_len
    contexts, futures = conditional_blocks(signals, split_index, history_len, pred_len, blocks)
    real_seq = np.concatenate([contexts, futures], axis=1).astype(np.float32)
    ts2vec = TS2Vec(input_dims=input_dims, device=device, batch_size=min(128, len(real_seq)), lr=0.001, output_dims=320, max_train_length=3000)
    ts2vec.fit(real_seq, verbose=False)
    real_repr = ts2vec.encode(real_seq, encoding_window="full_series", batch_size=min(128, len(real_seq))).reshape(len(real_seq), -1)
    ground_truth = signals[split_index:split_index + total_steps]
    fcd_real = compute_fcd(ground_truth, window_size=40, step=1)
    return {"contexts": contexts, "futures": futures, "ts2vec": ts2vec, "real_repr": real_repr, "ground_truth": ground_truth, "fcd_real": fcd_real}


def sddes_reference_side_effects(model, generated, state, samples, batch_size, pred_len, roi_count, device):
    ground_truth = state["ground_truth"]
    np.mean((ground_truth - generated) ** 2)
    fcd_fake = compute_fcd(generated, window_size=40, step=1)
    k = state["fcd_real"].shape[0]
    triu = np.triu_indices(k, k=1)
    ks_2samp(state["fcd_real"][triu], fcd_fake[triu])
    real_samples = split_sequences(ground_truth, pred_len)
    fake_samples = split_sequences(generated, pred_len)
    min_samples = min(len(real_samples), len(fake_samples))
    real_tensor = torch.tensor(real_samples[:min_samples], dtype=torch.float32, device=device)
    fake_tensor = torch.tensor(fake_samples[:min_samples], dtype=torch.float32, device=device)
    disc_x = torch.cat([real_tensor, fake_tensor], dim=0)
    disc_y = torch.cat([torch.zeros(len(real_tensor)), torch.ones(len(fake_tensor))], dim=0).to(device)
    idx = torch.randperm(len(disc_x), device=device)
    disc_x = disc_x[idx]
    disc_y = disc_y[idx]
    split_d = int(0.8 * len(disc_x))
    discriminator = MetricEvaluator(input_dim=roi_count).to(device)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    for _ in range(20):
        discriminator.train()
        opt_d.zero_grad()
        preds = discriminator.forward_disc(disc_x[:split_d]).squeeze()
        loss = nn.BCELoss()(preds, disc_y[:split_d])
        loss.backward()
        opt_d.step()
    discriminator.eval()
    with torch.no_grad():
        test_preds = discriminator.forward_disc(disc_x[split_d:]).squeeze()
        np.mean(disc_y[split_d:].detach().cpu().numpy() == (test_preds.detach().cpu().numpy() > 0.5).astype(int))
    predictor = MetricEvaluator(input_dim=roi_count).to(device)
    opt_p = torch.optim.Adam(predictor.parameters(), lr=0.001)
    for _ in range(20):
        predictor.train()
        opt_p.zero_grad()
        preds = predictor.forward_pred(fake_tensor[:, :-1, :])
        loss = nn.L1Loss()(preds, fake_tensor[:, -1, :])
        loss.backward()
        opt_p.step()
    predictor.eval()
    with torch.no_grad():
        predictor.forward_pred(real_tensor[:, :-1, :])
        emb_real = discriminator.get_embedding(real_tensor).detach().cpu().numpy()
        emb_fake = discriminator.get_embedding(fake_tensor).detach().cpu().numpy()
    fid(emb_real, emb_fake)
    calculate_cross_correlation(real_tensor)
    calculate_cross_correlation(fake_tensor)
    preds = conditional_samples(model, state["contexts"], samples, device, min(batch_size, len(state["contexts"])))
    empirical_crps(preds, state["futures"])
    qice(preds, state["futures"])
    probcorr(preds, state["futures"])
    b, p, n, s = preds.shape
    repeated_context = np.repeat(state["contexts"], s, axis=0)
    sampled_future = np.transpose(preds, (0, 3, 1, 2)).reshape(b * s, p, n)
    generated_seq = np.concatenate([repeated_context, sampled_future], axis=1).astype(np.float32)
    gen_repr = state["ts2vec"].encode(generated_seq, encoding_window="full_series", batch_size=min(128, len(generated_seq))).reshape(len(generated_seq), -1)
    fid(state["real_repr"], gen_repr)
