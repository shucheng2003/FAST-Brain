import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adjacency(adj, add_loop):
    x = np.array(adj, dtype=np.float32, copy=True)
    np.fill_diagonal(x, 0.0)
    if add_loop:
        x = x + np.eye(x.shape[0], dtype=np.float32)
    degree = x.sum(axis=1)
    inv = np.zeros_like(degree, dtype=np.float32)
    mask = degree > 0
    inv[mask] = np.power(degree[mask], -0.5)
    d = np.diag(inv)
    out = d @ x @ d
    return torch.tensor(np.nan_to_num(out), dtype=torch.float32)


def adjacency_powers(adj, order):
    vals = [torch.eye(adj.shape[0], dtype=torch.float32)]
    cur = adj
    for k in range(1, order):
        if k == 1:
            vals.append(cur)
        else:
            cur = cur @ adj
            vals.append(cur)
    return torch.stack(vals, dim=0)


class MultiGraphConvolution(nn.Module):
    def __init__(self, matrices, in_channels, out_channels, order):
        super().__init__()
        powers = []
        for i, mat in enumerate(matrices):
            powers.append(adjacency_powers(normalized_adjacency(mat, i == 0), order))
        self.register_buffer("powers", torch.stack(powers, dim=0))
        self.alpha = nn.Parameter(torch.ones(len(matrices), order) / float(order))
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        adj = torch.einsum("mk,mknj->nj", self.alpha, self.powers)
        support = torch.einsum("nm,bmc->bnc", adj, x)
        return F.relu(support @ self.weight + self.bias)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=50000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("rotary dimension must be even")
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv", inv)

    def forward(self, positions):
        freqs = torch.outer(positions.float(), self.inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    a, b = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-b, a], dim=-1)


def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return x * cos + rotate_half(x) * sin


class RotaryMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("head dimension must be even")
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value=None, q_cos=None, q_sin=None, k_cos=None, k_sin=None):
        if key_value is None:
            key_value = query
        b, q_len, _ = query.shape
        kv_len = key_value.shape[1]
        q = self.q_proj(query).view(b, q_len, self.n_heads, self.head_dim)
        k = self.k_proj(key_value).view(b, kv_len, self.n_heads, self.head_dim)
        v = self.v_proj(key_value).view(b, kv_len, self.n_heads, self.head_dim)
        if q_cos is not None and q_sin is not None:
            q = apply_rope(q, q_cos, q_sin)
        if k_cos is not None and k_sin is not None:
            k = apply_rope(k, k_cos, k_sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.d_model)
        return self.out_proj(out)


class StandardEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.self_attn(x, x, x)[0]
        x = self.norm1(x + self.dropout(y))
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout(y))


class RotaryEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = RotaryMultiheadAttention(d_model, n_heads, dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, cos, sin):
        y = self.self_attn(x, q_cos=cos, q_sin=sin, k_cos=cos, k_sin=sin)
        x = self.norm1(x + self.dropout(y))
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout(y))


class FastBrainNetwork(nn.Module):
    def __init__(self, roi_count, history_len, pred_len, matrices, cfg, use_rope=True):
        super().__init__()
        d_model = int(cfg["d_model"])
        n_heads = int(cfg["n_heads"])
        d_ff = int(cfg.get("d_ff", d_model * 4))
        dropout = float(cfg["dropout"])
        bottleneck = int(cfg.get("bottleneck_dim", 256))
        n_layers = int(cfg.get("n_layers", 2))
        self.roi_count = roi_count
        self.history_len = history_len
        self.pred_len = pred_len
        self.use_rope = use_rope
        self.input_proj = nn.Sequential(nn.Linear(roi_count, bottleneck), nn.SiLU(), nn.Linear(bottleneck, d_model))
        self.hist_proj = nn.Sequential(nn.Linear(roi_count, bottleneck), nn.SiLU(), nn.Linear(bottleneck, d_model))
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if use_rope:
            self.rope = RotaryEmbedding(d_model // n_heads)
            self.cross_attn = RotaryMultiheadAttention(d_model, n_heads, dropout)
            self.layers = nn.ModuleList([RotaryEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        else:
            self.hist_pos_emb = nn.Parameter(torch.randn(1, history_len, d_model) * 0.02)
            self.out_pos_emb = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.layers = nn.ModuleList([StandardEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm1 = nn.LayerNorm(d_model)
        self.out_trans = nn.Linear(d_model, roi_count)
        self.dropout = nn.Dropout(dropout)
        hidden = int(cfg.get("gcn_hidden", d_model // 2))
        order = int(cfg["gcn_order"])
        self.gcn_in = nn.Linear(1, hidden)
        self.gcn = nn.Sequential(
            MultiGraphConvolution(matrices, hidden, hidden, order),
            nn.SiLU(),
            MultiGraphConvolution(matrices, hidden, hidden, order),
        )
        self.gcn_out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gcn_out.weight)
        nn.init.zeros_(self.gcn_out.bias)

    def forward(self, z, t, history):
        b, p, n = z.shape
        x = self.input_proj(z)
        if t.dim() == 1:
            t = t.view(-1, 1)
        t_emb = self.time_mlp(t).unsqueeze(1)
        if self.use_rope:
            query = x + t_emb
            hist = self.hist_proj(history)
            hist_pos = torch.arange(history.shape[1], device=history.device)
            pred_pos = torch.arange(history.shape[1], history.shape[1] + p, device=history.device)
            q_cos, q_sin = self.rope(pred_pos)
            k_cos, k_sin = self.rope(hist_pos)
            y = self.cross_attn(query, hist, q_cos, q_sin, k_cos, k_sin)
            h = self.norm1(query + self.dropout(y))
            for layer in self.layers:
                h = layer(h, q_cos, q_sin)
        else:
            query = x + t_emb + self.out_pos_emb[:, :p, :]
            hist = self.hist_proj(history) + self.hist_pos_emb[:, :history.shape[1], :]
            y = self.cross_attn(query, hist, hist)[0]
            h = self.norm1(query + self.dropout(y))
            for layer in self.layers:
                h = layer(h)
        temporal = self.out_trans(h)
        g = self.gcn_in(z.reshape(b * p, n, 1))
        g = self.gcn(g)
        g = self.gcn_out(g).reshape(b, p, n)
        return temporal + g


class FastBrainModel(nn.Module):
    def __init__(self, roi_count, history_len, pred_len, matrices, cfg):
        super().__init__()
        self.pred_len = pred_len
        self.roi_count = roi_count
        self.euler_steps = int(cfg["euler_steps"])
        self.denom_clamp = float(cfg["denom_clamp"])
        self.net = FastBrainNetwork(
            roi_count,
            history_len,
            pred_len,
            matrices,
            cfg,
            use_rope=bool(cfg.get("use_rope", True)),
        )

    def network_output(self, z, t, history):
        return self.net(z, t, history)

    def clean_from_output(self, out, z, t):
        return out

    def loss(self, history, target, mu_t, sigma_t):
        b = history.shape[0]
        t = torch.sigmoid(mu_t + sigma_t * torch.randn(b, device=history.device))
        tau = t.view(b, 1, 1)
        eps = torch.randn_like(target)
        z = tau * target + (1.0 - tau) * eps
        out = self.network_output(z, t, history)
        v = (out - z) / (1.0 - tau).clamp(min=self.denom_clamp)
        return F.mse_loss(v, target - eps)

    def forward(self, history):
        history = history.view(history.shape[0], -1, self.roi_count)
        return self.sample(history).reshape(history.shape[0], -1)

    @torch.no_grad()
    def sample(self, history):
        b = history.shape[0]
        z = torch.randn(b, self.pred_len, self.roi_count, device=history.device)
        dt = 1.0 / float(self.euler_steps)
        for i in range(self.euler_steps):
            t = torch.full((b,), i / float(self.euler_steps), device=history.device)
            out = self.network_output(z, t, history)
            clean = self.clean_from_output(out, z, t)
            tau = t.view(b, 1, 1)
            v = (clean - z) / (1.0 - tau).clamp(min=self.denom_clamp)
            z = z + dt * v
        return z

    def clean_prediction_for_ec(self, history):
        b = history.shape[0]
        t = torch.zeros(b, device=history.device)
        z = history[:, -1:, :].expand(b, self.pred_len, self.roi_count).contiguous()
        out = self.network_output(z, t, history)
        return self.clean_from_output(out, z, t)
