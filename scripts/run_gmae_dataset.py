#!/usr/bin/env python
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import csv
import itertools
import json
import logging
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss import orthogonal_loss
from models import GMAE_MVC


DATASETS = [
    "ORL",
    "YaleB",
    "flower17",
    "COIL20",
    "Caltech101-7",
    "100leaves",
    "Mfeat",
    "UCI_Digits",
    "NTU2012_mvcnn_gvcnn",
    "MNIST",
    "animal",
    "ALOI",
    "VGGFace2-50",
    "CIFAR10_llc_with_img_fea",
    "Food-101",
]


def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_labels(y):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    values = np.unique(y)
    mapping = {v: i for i, v in enumerate(values)}
    return np.array([mapping[v] for v in y], dtype=np.int64)


def _mat_label(data):
    for key in ("Y", "y", "gt", "labels"):
        if key in data:
            return data[key]
    raise KeyError("No label key found among Y/y/gt/labels")


def _cell_features(x, n_samples):
    features = []
    for item in x.ravel(order="C"):
        arr = np.asarray(item)
        if arr.ndim != 2:
            continue
        if arr.shape[0] != n_samples and arr.shape[1] == n_samples:
            arr = arr.T
        if arr.shape[0] == n_samples:
            features.append(arr.astype(np.float32, copy=False))
    return features


def _load_hdf5_mat(path):
    with h5py.File(path, "r") as f:
        y = normalize_labels(np.array(f["Y"]).reshape(-1))
        x_ref = f["X"]
        features = []
        for ref in x_ref[()].ravel(order="C"):
            arr = np.array(f[ref])
            if arr.shape[0] != len(y) and arr.shape[1] == len(y):
                arr = arr.T
            features.append(arr.astype(np.float32, copy=False))
    return features, y


def load_multiview_dataset(path):
    path = Path(path)
    load_started = time.perf_counter()
    if path.name == "Food-101.mat":
        data = sio.loadmat(path)
        y = normalize_labels(data["labels"])
        features = [
            data["image_features"].astype(np.float32, copy=False),
            data["class_text_features"].astype(np.float32, copy=False),
            data["desc_text_features"].astype(np.float32, copy=False),
        ]
    else:
        try:
            data = sio.loadmat(path)
            y = normalize_labels(_mat_label(data))
            if "X" in data:
                features = _cell_features(data["X"], len(y))
            elif "fea" in data:
                features = _cell_features(data["fea"], len(y))
            else:
                raise KeyError("No feature key found among X/fea")
        except NotImplementedError:
            features, y = _load_hdf5_mat(path)

    if not features:
        raise ValueError(f"No features loaded from {path}")
    for i, x in enumerate(features):
        if x.shape[0] != len(y):
            raise ValueError(f"View {i} has {x.shape[0]} rows but labels have {len(y)}")
        features[i] = MinMaxScaler().fit_transform(x).astype(np.float32, copy=False)
    return features, y, time.perf_counter() - load_started


def aligned_macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = max(y_true.max(), y_pred.max()) + 1
    conf = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(p, p) for p in y_pred], dtype=np.int64)
    return float(f1_score(y_true, aligned, average="macro"))


def clustering_metrics(y_true, y_pred):
    return {
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "f1_macro": aligned_macro_f1(y_true, y_pred),
    }


def prepare_neighbors(features, pos_num, neg_num, seed):
    rng = np.random.default_rng(seed)
    n = features[0].shape[0]
    all_pos = []
    all_neg = []
    for x in features:
        k = min(n, pos_num)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(x)
        idx = nbrs.kneighbors(x, return_distance=False)
        pos = idx[:, 1:k] if k > 1 else idx[:, :0]
        if pos.shape[1] < pos_num - 1:
            fill = rng.integers(0, n, size=(n, pos_num - 1 - pos.shape[1]))
            pos = np.concatenate([pos, fill], axis=1)
        neg = rng.integers(0, n, size=(n, neg_num))
        rows = np.arange(n)[:, None]
        neg = np.where(neg == rows, (neg + 1) % n, neg)
        all_pos.append(torch.LongTensor(pos[:, : pos_num - 1]))
        all_neg.append(torch.LongTensor(neg))
    return torch.cat(all_pos, dim=1), torch.cat(all_neg, dim=1)


def contrastive_loss_vectorized(args, hidden, nbr_idx, neg_idx):
    if not args.do_contrast:
        return hidden.new_tensor(0.0)
    device = hidden.device
    nbr_idx = nbr_idx.to(device)
    neg_idx = neg_idx.to(device)
    total = hidden.new_tensor(0.0)
    n = hidden.shape[0]
    chunk_size = max(1, int(args.contrast_chunk_size))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        anchor = hidden[start:end].unsqueeze(1)
        positive = torch.exp(torch.cosine_similarity(anchor, hidden[nbr_idx[start:end]].detach(), dim=-1))
        negative = torch.exp(
            torch.cosine_similarity(anchor, hidden[neg_idx[start:end]].detach(), dim=-1)
        ).sum(dim=1, keepdim=True)
        chunk_loss = -torch.log((positive / negative.clamp_min(1e-12)).clamp_min(1e-12)).sum(dim=1).sum()
        total = total + chunk_loss
    return total / n


def make_logger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def train_and_eval(args, dataset_name, features_np, labels_np, alpha, beta, seed, run_dir, save_model=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(run_dir / "run.log")
    seed_setting(seed)
    started = time.perf_counter()
    device = torch.device(args.device)
    features = [torch.from_numpy(x).to(device) for x in features_np]
    input_dims = [x.shape[1] for x in features_np]
    view_num = len(features_np)
    n_clusters = int(np.unique(labels_np).size)

    nbr_started = time.perf_counter()
    nbr_idx, neg_idx = prepare_neighbors(features_np, args.pos_num, args.neg_num, seed)
    neighbor_seconds = time.perf_counter() - nbr_started

    model = GMAE_MVC(input_dims, view_num, args.feature_dim, h_dims=[500, 200]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss_fn = nn.MSELoss()

    logger.info(
        json.dumps(
            {
                "event": "start",
                "dataset": dataset_name,
                "seed": seed,
                "alpha_lambda_ma": alpha,
                "beta_lambda_con": beta,
                "epochs": args.train_epoch,
                "feature_dim": args.feature_dim,
                "pos_num": args.pos_num,
                "neg_num": args.neg_num,
                "contrast_chunk_size": args.contrast_chunk_size,
                "input_dims": input_dims,
                "samples": len(labels_np),
                "views": view_num,
                "clusters": n_clusters,
            },
            sort_keys=True,
        )
    )

    for epoch in range(args.train_epoch):
        model.train()
        optimizer.zero_grad()
        hidden_share, hidden_specific, hidden, recs = model(features)
        loss_rec = hidden.new_tensor(0.0)
        loss_mi = hidden.new_tensor(0.0)
        loss_ad = hidden.new_tensor(0.0)
        for v in range(view_num):
            loss_rec = loss_rec + mse_loss_fn(recs[v], features[v])
            loss_mi = loss_mi + orthogonal_loss(hidden_share, hidden_specific[v])
            loss_ad = loss_ad + model.discriminators_loss(hidden_specific, v)
        loss_con = contrastive_loss_vectorized(args, hidden, nbr_idx, neg_idx)
        total_loss = loss_rec + alpha * (loss_mi + loss_ad) + beta * loss_con
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % args.log_interval == 0 or epoch + 1 == args.train_epoch:
            logger.info(
                json.dumps(
                    {
                        "event": "epoch",
                        "epoch": epoch + 1,
                        "loss": float(total_loss.detach().cpu()),
                        "loss_rec": float(loss_rec.detach().cpu()),
                        "loss_mi": float(loss_mi.detach().cpu()),
                        "loss_ad": float(loss_ad.detach().cpu()),
                        "loss_con": float(loss_con.detach().cpu()),
                    },
                    sort_keys=True,
                )
            )

    model.eval()
    with torch.no_grad():
        _, _, hidden, _ = model(features)
        labels_pred = KMeans(n_clusters=n_clusters, n_init=args.kmeans_n_init, random_state=seed).fit_predict(
            hidden.detach().cpu().numpy()
        )
    metrics = clustering_metrics(labels_np, labels_pred)
    total_seconds = time.perf_counter() - started
    metrics.update(
        {
            "dataset": dataset_name,
            "seed": seed,
            "alpha_lambda_ma": alpha,
            "beta_lambda_con": beta,
            "train_epoch": args.train_epoch,
            "feature_dim": args.feature_dim,
            "neighbor_seconds": neighbor_seconds,
            "total_seconds": total_seconds,
        }
    )
    np.save(run_dir / "labels.npy", labels_pred)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    if save_model:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dims": input_dims,
                "view_num": view_num,
                "feature_dim": args.feature_dim,
                "h_dims": [500, 200],
                "seed": seed,
                "alpha_lambda_ma": alpha,
                "beta_lambda_con": beta,
            },
            run_dir / "model.pt",
        )
    logger.info(json.dumps({"event": "final", **metrics}, sort_keys=True))
    return metrics


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_repeats(dataset_name, dataset_info, best_params, repeat_rows):
    summary = {
        "dataset": dataset_name,
        "samples": dataset_info["samples"],
        "views": dataset_info["views"],
        "dims": json.dumps(dataset_info["dims"]),
        "clusters": dataset_info["clusters"],
        "best_alpha_lambda_ma": best_params["alpha_lambda_ma"],
        "best_beta_lambda_con": best_params["beta_lambda_con"],
    }
    for key in ("nmi", "ari", "f1_macro", "total_seconds"):
        values = np.array([row[key] for row in repeat_rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def run_dataset(args, dataset_name):
    dataset_path = Path(args.data_root) / f"{dataset_name}.mat"
    dataset_dir = Path(args.results_root) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    features, labels, load_seconds = load_multiview_dataset(dataset_path)
    dataset_info = {
        "dataset": dataset_name,
        "samples": int(len(labels)),
        "views": int(len(features)),
        "dims": [int(x.shape[1]) for x in features],
        "clusters": int(np.unique(labels).size),
        "load_seconds": load_seconds,
        "data_path": str(dataset_path),
    }
    with open(dataset_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, sort_keys=True)

    alphas = [round(x, 2) for x in np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step)]
    betas = [round(x, 2) for x in np.arange(args.beta_min, args.beta_max + 1e-9, args.beta_step)]
    search_rows = []
    search_dir = dataset_dir / "search_seed42"
    for alpha, beta in itertools.product(alphas, betas):
        run_name = f"alpha_{alpha:.2f}_beta_{beta:.2f}"
        metrics_path = search_dir / run_name / "metrics.json"
        if args.resume and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            metrics = train_and_eval(
                args,
                dataset_name,
                features,
                labels,
                alpha,
                beta,
                args.search_seed,
                search_dir / run_name,
                save_model=False,
            )
        search_rows.append(metrics)
        write_csv(
            search_dir / "search_metrics.csv",
            search_rows,
            [
                "dataset",
                "seed",
                "alpha_lambda_ma",
                "beta_lambda_con",
                "nmi",
                "ari",
                "f1_macro",
                "train_epoch",
                "feature_dim",
                "neighbor_seconds",
                "total_seconds",
            ],
        )

    best = max(search_rows, key=lambda row: row["nmi"])
    with open(dataset_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, sort_keys=True)

    best_dir = dataset_dir / "best_seed42"
    if not (args.resume and (best_dir / "metrics.json").exists()):
        train_and_eval(
            args,
            dataset_name,
            features,
            labels,
            best["alpha_lambda_ma"],
            best["beta_lambda_con"],
            args.search_seed,
            best_dir,
            save_model=True,
        )

    repeat_rows = []
    repeats_dir = dataset_dir / "five_seed_repeats"
    for seed in args.repeat_seeds:
        run_dir = repeats_dir / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        if args.resume and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            metrics = train_and_eval(
                args,
                dataset_name,
                features,
                labels,
                best["alpha_lambda_ma"],
                best["beta_lambda_con"],
                seed,
                run_dir,
                save_model=True,
            )
        repeat_rows.append(metrics)
    write_csv(
        repeats_dir / "repeat_metrics.csv",
        repeat_rows,
        [
            "dataset",
            "seed",
            "alpha_lambda_ma",
            "beta_lambda_con",
            "nmi",
            "ari",
            "f1_macro",
            "train_epoch",
            "feature_dim",
            "neighbor_seconds",
            "total_seconds",
        ],
    )
    summary = summarize_repeats(dataset_name, dataset_info, best, repeat_rows)
    summary["wall_seconds_including_load"] = time.perf_counter() - load_started
    with open(dataset_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data_root", default="/home/disk2/zhangh/research/clustering/mvcdatasets")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_epoch", type=int, default=500)
    parser.add_argument("--search_seed", type=int, default=42)
    parser.add_argument("--repeat_seeds", type=int, nargs="+", default=[42, 3407, 4079, 2024, 0])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--pos_num", type=int, default=21)
    parser.add_argument("--neg_num", type=int, default=128)
    parser.add_argument("--contrast_chunk_size", type=int, default=2048)
    parser.add_argument("--do_contrast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kmeans_n_init", type=int, default=50)
    parser.add_argument("--alpha_min", type=float, default=0.01)
    parser.add_argument("--alpha_max", type=float, default=0.07)
    parser.add_argument("--alpha_step", type=float, default=0.01)
    parser.add_argument("--beta_min", type=float, default=0.01)
    parser.add_argument("--beta_max", type=float, default=0.07)
    parser.add_argument("--beta_step", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_dataset(args, args.dataset)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
