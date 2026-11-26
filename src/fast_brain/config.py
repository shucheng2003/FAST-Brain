from pathlib import Path
import copy
import yaml


def project_root():
    assets = Path(__file__).resolve().parent / "assets"
    if (assets / "configs").is_dir():
        return assets
    return Path(__file__).resolve().parents[2]


def load_config(dataset):
    path = project_root() / "configs" / f"{dataset}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merged_config(dataset, args):
    cfg = copy.deepcopy(load_config(dataset))
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.history_len is not None:
        cfg["history_len"] = args.history_len
    if args.pred_len is not None:
        cfg["pred_len"] = args.pred_len
    if args.euler_steps is not None:
        cfg["model"]["euler_steps"] = args.euler_steps
    if args.use_rope is not None:
        cfg["model"]["use_rope"] = args.use_rope
    if args.conditional_blocks is not None:
        cfg["evaluation"]["conditional_blocks"] = args.conditional_blocks
    if args.num_prob_samples is not None:
        cfg["evaluation"]["conditional_samples"] = args.num_prob_samples
    if args.num_runs is not None:
        cfg["evaluation"]["num_runs"] = args.num_runs
    return cfg
