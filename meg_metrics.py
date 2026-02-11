from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None



def safe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def list_sample_dirs(base_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(base_dir):
        return out
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "data.json")):
            out.append(p)
    def key_fn(x):
        try:
            return int(os.path.basename(x))
        except Exception:
            return os.path.basename(x)
    return sorted(out, key=key_fn)


def load_meg_run(sample_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(sample_dir, "data.json"), "r") as f:
        return json.load(f)


def is_tox21_record(rec: Dict[str, Any]) -> bool:
    return safe_get(rec, "prediction.type") == "bin_classification"


def probs_and_class(rec: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    probs = np.array(safe_get(rec, "prediction.output"), dtype=float)
    pred_class = int(safe_get(rec, "prediction.class"))
    return probs, pred_class


def confidence_margin(probs: np.ndarray) -> float:
    """p_top1 - p_top2."""
    if probs.size < 2:
        return float("nan")
    s = np.sort(probs)[::-1]
    return float(s[0] - s[1])


@dataclass
class PairMetrics:
    sample_id: str
    og_smiles: Optional[str]
    cf_smiles: Optional[str]
    og_pred_class: int
    cf_pred_class: int
    flip: int

    target_class: int
    p_target_og: float
    p_target_cf: float
    delta_p_target: float
    cf_conf_margin: float
    similarity: float
    reward: float
    reward_pred: float
    reward_sim: float



def compute_metrics_for_sample(sample_dir: str, top_k: int = 10) -> Tuple[List[PairMetrics], Dict[str, Any]]:

    sample_id = os.path.basename(sample_dir)
    data = load_meg_run(sample_dir)

    og = None
    cfs = []
    for rec in data:
        if rec.get("marker") == "og":
            og = rec
        elif rec.get("marker") == "cf":
            cfs.append(rec)

    if og is None:
        raise ValueError(f"No original (marker=='og') found in {sample_dir}")
    if not is_tox21_record(og):
        raise ValueError(f"Non-tox21 record in {sample_dir}. This script is for TOX21 (bin_classification).")

    og_probs, og_class = probs_and_class(og)
    target_class = 1 - og_class
    p_target_og = float(og_probs[target_class])
    og_smiles = og.get("smiles")


    cfs_sorted = sorted(cfs, key=lambda r: float(r.get("reward", -1e9)), reverse=True)

    pairs: List[PairMetrics] = []
    flips_in_topk = 0
    flips_total = 0

    for rank, cf in enumerate(cfs_sorted):
        cf_probs, cf_class = probs_and_class(cf)
        cf_smiles = cf.get("smiles")
        flip = int(cf_class != og_class)

        if flip:
            flips_total += 1
            if rank < top_k:
                flips_in_topk += 1


        sim = cf.get("reward_sim", None)
        if sim is None:
            sim = cf.get("similarity", None)
        similarity = float(sim) if sim is not None else float("nan")

        p_target_cf = float(cf_probs[target_class])
        delta_p = p_target_cf - p_target_og

        pm = PairMetrics(
            sample_id=str(sample_id),
            og_smiles=og_smiles,
            cf_smiles=cf_smiles,
            og_pred_class=og_class,
            cf_pred_class=cf_class,
            flip=flip,
            target_class=target_class,
            p_target_og=p_target_og,
            p_target_cf=p_target_cf,
            delta_p_target=delta_p,
            cf_conf_margin=confidence_margin(cf_probs),
            similarity=similarity,
            reward=float(cf.get("reward", float("nan"))),
            reward_pred=float(cf.get("reward_pred", float("nan"))),
            reward_sim=float(cf.get("reward_sim", float("nan"))),
        )
        pairs.append(pm)

    sample_summary = {
        "sample_id": str(sample_id),
        "og_pred_class": og_class,
        "target_class": target_class,
        "num_cfs": len(cfs_sorted),
        "num_flips_total": flips_total,
        "num_flips_topk": flips_in_topk,
        "success_at_k": int(flips_in_topk > 0),
    }
    return pairs, sample_summary


def aggregate_metrics(
    all_pairs: List[PairMetrics],
    sample_summaries: List[Dict[str, Any]],
    sim_thresholds: List[float],
    top_k: int,
) -> Dict[str, Any]:
    """Compute global metrics."""
    flips = np.array([p.flip for p in all_pairs], dtype=int)
    sim = np.array([p.similarity for p in all_pairs], dtype=float)
    dpt = np.array([p.delta_p_target for p in all_pairs], dtype=float)

    og_cls = np.array([p.og_pred_class for p in all_pairs], dtype=int)
    cf_cls = np.array([p.cf_pred_class for p in all_pairs], dtype=int)

    overall_flip_rate = float(np.mean(flips)) if flips.size else float("nan")


    tox_to_notox = (og_cls == 1) & (cf_cls == 0)
    notox_to_tox = (og_cls == 0) & (cf_cls == 1)
    tox_pairs = (og_cls == 1)
    notox_pairs = (og_cls == 0)

    tox_to_notox_rate = float(np.sum(tox_to_notox) / np.sum(tox_pairs)) if np.sum(tox_pairs) else float("nan")
    notox_to_tox_rate = float(np.sum(notox_to_tox) / np.sum(notox_pairs)) if np.sum(notox_pairs) else float("nan")


    success_at_k = float(np.mean([s["success_at_k"] for s in sample_summaries])) if sample_summaries else float("nan")
    avg_flips_per_seed = float(np.mean([s["num_flips_total"] for s in sample_summaries])) if sample_summaries else float("nan")

    flip_mask = flips == 1
    nonflip_mask = flips == 0

    def stats(x: np.ndarray) -> Dict[str, float]:
        x = x[~np.isnan(x)]
        if x.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "min": float("nan"),
                    "q25": float("nan"), "q75": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "min": float(np.min(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "max": float(np.max(x)),
        }

    sim_stats_flip = stats(sim[flip_mask])
    sim_stats_nonflip = stats(sim[nonflip_mask])

    flip_rate_by_threshold = {}
    for t in sim_thresholds:
        eligible = (~np.isnan(sim)) & (sim >= t)
        if np.sum(eligible) == 0:
            flip_rate_by_threshold[f"sim>={t}"] = float("nan")
        else:
            flip_rate_by_threshold[f"sim>={t}"] = float(np.mean(flips[eligible]))

    corr = float("nan")
    finite_mask = (~np.isnan(sim)) & (~np.isnan(dpt))
    if np.sum(finite_mask) >= 2:
        corr = float(np.corrcoef(sim[finite_mask], dpt[finite_mask])[0, 1])


    dpt_stats_flip = stats(dpt[flip_mask])
    dpt_stats_nonflip = stats(dpt[nonflip_mask])

    return {
        "pair_level": {
            "overall_flip_rate": overall_flip_rate,
            "tox_to_notox_rate": tox_to_notox_rate,
            "notox_to_tox_rate": notox_to_tox_rate,
            "num_pairs": int(len(all_pairs)),
        },
        "seed_level": {
            f"success_at_{top_k}": success_at_k,
            "avg_flips_per_seed": avg_flips_per_seed,
            "num_seeds": int(len(sample_summaries)),
        },
        "similarity": {
            "flip": sim_stats_flip,
            "non_flip": sim_stats_nonflip,
            "flip_rate_by_threshold": flip_rate_by_threshold,
        },
        "prediction_contrast": {
            "delta_p_target_flip": dpt_stats_flip,
            "delta_p_target_nonflip": dpt_stats_nonflip,
            "corr_similarity_delta_p": corr,
        },
    }



def make_pareto_tables(all_pairs: List[PairMetrics], top_n: int = 20):

    flips = [p for p in all_pairs if p.flip == 1]
    by_reward = sorted(flips, key=lambda p: p.reward, reverse=True)[:top_n]
    by_sim = sorted(flips, key=lambda p: (p.similarity if not np.isnan(p.similarity) else -1e9), reverse=True)[:top_n]
    by_dpt = sorted(flips, key=lambda p: p.delta_p_target, reverse=True)[:top_n]
    return by_reward, by_sim, by_dpt


def pairs_to_dataframe(pairs: List[PairMetrics]):
    if pd is None:
        return None
    return pd.DataFrame([p.__dict__ for p in pairs])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True,
                    help="Path to meg_output directory, e.g. runs/tox21/<exp>/meg_output")
    ap.add_argument("--k", type=int, default=10, help="Top-k for Success@k and flips in top-k")
    ap.add_argument("--sim-thresholds", type=float, nargs="*", default=[0.9, 0.8, 0.7],
                    help="Similarity thresholds to report flip rate at/above")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N for Pareto-style tables")
    ap.add_argument("--save-csv", action="store_true", help="Save pair-level table + pareto tables as CSVs")
    args = ap.parse_args()

    sample_dirs = list_sample_dirs(args.base)
    if not sample_dirs:
        raise SystemExit(f"No sample dirs with data.json found under: {args.base}")

    all_pairs: List[PairMetrics] = []
    sample_summaries: List[Dict[str, Any]] = []

    for sd in sample_dirs:
        try:
            pairs, summ = compute_metrics_for_sample(sd, top_k=args.k)
            all_pairs.extend(pairs)
            sample_summaries.append(summ)
        except Exception as e:
            print(f"[WARN] Skipping {sd}: {e}")

    agg = aggregate_metrics(all_pairs, sample_summaries, args.sim_thresholds, args.k)

    
    print("\n====================")
    print("MEG RL METRICS (TOX21)")
    print("====================")
    print(json.dumps(agg, indent=2))


    by_reward, by_sim, by_dpt = make_pareto_tables(all_pairs, top_n=args.top_n)

    def print_top(title: str, lst: List[PairMetrics], n: int = 10):
        print(f"\n--- {title} (top {min(n, len(lst))}) ---")
        for p in lst[:n]:
            print(
                f"seed={p.sample_id} flip={p.flip} "
                f"og_cls={p.og_pred_class} cf_cls={p.cf_pred_class} "
                f"pT(og)={p.p_target_og:.3f} pT(cf)={p.p_target_cf:.3f} Δp={p.delta_p_target:.3f} "
                f"sim={p.similarity:.3f} reward={p.reward:.3f}"
            )

    print_top("TOP FLIPS BY REWARD", by_reward)
    print_top("TOP FLIPS BY SIMILARITY", by_sim)
    print_top("TOP FLIPS BY Δp(target)", by_dpt)

    if args.save_csv and pd is not None:
        out_dir = os.path.join(args.base, "_metrics")
        os.makedirs(out_dir, exist_ok=True)

        df_all = pairs_to_dataframe(all_pairs)
        df_all.to_csv(os.path.join(out_dir, "pairs_all.csv"), index=False)

        pairs_to_dataframe(by_reward).to_csv(os.path.join(out_dir, "pareto_flips_by_reward.csv"), index=False)
        pairs_to_dataframe(by_sim).to_csv(os.path.join(out_dir, "pareto_flips_by_similarity.csv"), index=False)
        pairs_to_dataframe(by_dpt).to_csv(os.path.join(out_dir, "pareto_flips_by_delta_p.csv"), index=False)

        if pd is not None:
            pd.DataFrame(sample_summaries).to_csv(os.path.join(out_dir, "seed_level_summary.csv"), index=False)

        with open(os.path.join(out_dir, "aggregate_summary.json"), "w") as f:
            json.dump(agg, f, indent=2)

        print(f"\n[OK] Saved CSV/JSON reports to: {out_dir}")
    elif args.save_csv and pd is None:
        print("\n[WARN] pandas not installed -> cannot save CSV. Install pandas or remove --save-csv.")


if __name__ == "__main__":
    main()
