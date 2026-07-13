#!/usr/bin/env python
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5"])
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--data_root", default="/home/disk2/zhangh/research/clustering/mvcdatasets")
    parser.add_argument("--train_epoch", type=int, default=500)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--neg_num", type=int, default=128)
    parser.add_argument("--contrast_chunk_size", type=int, default=2048)
    parser.add_argument("--kmeans_n_init", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def command_for(args, dataset):
    cmd = [
        sys.executable,
        "scripts/run_gmae_dataset.py",
        "--dataset",
        dataset,
        "--data_root",
        args.data_root,
        "--results_root",
        args.results_root,
        "--device",
        "cuda:0",
        "--train_epoch",
        str(args.train_epoch),
        "--feature_dim",
        str(args.feature_dim),
        "--neg_num",
        str(args.neg_num),
        "--contrast_chunk_size",
        str(args.contrast_chunk_size),
        "--kmeans_n_init",
        str(args.kmeans_n_init),
    ]
    if args.resume:
        cmd.append("--resume")
    return cmd


def collect_summaries(results_root):
    rows = []
    for path in sorted(Path(results_root).glob("*/summary.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        return
    fieldnames = [
        "dataset",
        "samples",
        "views",
        "dims",
        "clusters",
        "nmi_mean",
        "nmi_std",
        "ari_mean",
        "ari_std",
        "f1_macro_mean",
        "f1_macro_std",
        "total_seconds_mean",
        "total_seconds_std",
        "best_alpha_lambda_ma",
        "best_beta_lambda_con",
        "wall_seconds_including_load",
    ]
    out = Path(results_root) / "summary_all.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    logs_dir = results_root / "_launcher_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = list(args.datasets)
    running = {}
    failures = []

    while pending or running:
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            dataset = pending.pop(0)
            log_path = logs_dir / f"{dataset}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["OPENBLAS_NUM_THREADS"] = "4"
            with open(log_path, "ab", buffering=0) as log:
                proc = subprocess.Popen(command_for(args, dataset), stdout=log, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (dataset, proc, log_path, time.time())
            print(f"START gpu={gpu} dataset={dataset} log={log_path}", flush=True)

        time.sleep(20)
        for gpu, (dataset, proc, log_path, started) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            elapsed = time.time() - started
            print(f"DONE gpu={gpu} dataset={dataset} code={code} elapsed={elapsed:.1f}s", flush=True)
            if code != 0:
                failures.append({"dataset": dataset, "gpu": gpu, "code": code, "log": str(log_path)})
            del running[gpu]
            collect_summaries(args.results_root)

    collect_summaries(args.results_root)
    if failures:
        fail_path = results_root / "_launcher_failures.json"
        fail_path.write_text(json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8")
        raise SystemExit(f"Failures occurred; see {fail_path}")


if __name__ == "__main__":
    main()
