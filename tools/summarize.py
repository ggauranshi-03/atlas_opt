"""Paper tables: full-precision vs INT4-AWQ (g128) test metrics, mean +- std over training seeds.

Per training seed, INT4 results are first averaged over calibration seeds; the table then reports the
mean +- sample std across training seeds. LM metric: held-out FineWeb-Edu and WikiText-2 perplexity;
CIFAR-10 metric: official test-set accuracy.

  python tools/summarize.py --config configs/tuned/nanogpt_fineweb.yaml
"""
import os
import csv
import math
import argparse
import statistics as st
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mean_std(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0) if xs else (math.nan, math.nan)


def fmt(ms, digits=2):
    m, s = ms
    return "—" if math.isnan(m) else f"{m:.{digits}f} ± {s:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--wbits", default="4")
    ap.add_argument("--group-size", default="128")
    ap.add_argument("--method", default="awq")
    a = ap.parse_args()
    os.chdir(ROOT)

    cfg = yaml.safe_load(open(a.config))
    cid = cfg["experiment"]["experiment_id"]
    cifar = cfg["experiment"].get("task_type") == "image_classification"
    metrics = ["test_acc"] if cifar else ["fineweb_ppl", "wikitext2_ppl"]

    per = {}  # (optimizer, seed) -> {"fp": {m: v}, "q": {m: [v over calib seeds]}}
    with open(f"ptq_results/{cid}.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                if r["method"] == "fp":
                    per.setdefault((r["optimizer"], r["seed"]), {"fp": {}, "q": {}})["diverged"] = True
                continue
            d = per.setdefault((r["optimizer"], r["seed"]), {"fp": {}, "q": {}})
            if r["method"] == "fp":
                d["fp"] = {m: float(r[m]) for m in metrics}
            elif r["method"] == a.method and r["wbits"] == a.wbits and r["group_size"] == a.group_size:
                for m in metrics:
                    d["q"].setdefault(m, []).append(float(r[m]))

    opts = {}
    for (opt, seed), d in per.items():
        o = opts.setdefault(opt, {"seeds": 0, "diverged": 0, "fp": {m: [] for m in metrics},
                                  "q": {m: [] for m in metrics}, "rel": {m: [] for m in metrics}})
        o["seeds"] += 1
        if d.get("diverged") or not d["fp"]:
            o["diverged"] += 1
            continue
        for m in metrics:
            fp = d["fp"][m]
            q = st.mean(d["q"][m]) if d["q"].get(m) else math.nan
            o["fp"][m].append(fp)
            o["q"][m].append(q)
            # accuracy: drop in points; perplexity: relative increase in %
            o["rel"][m].append(fp - q if cifar else 100.0 * (q / fp - 1.0))

    key = metrics[0]
    order = sorted(opts, key=lambda k: (-1 if cifar else 1) * (mean_std(opts[k]["fp"][key])[0]
                                                               if opts[k]["fp"][key] else (-math.inf if cifar else math.inf)))
    tag = f"INT{a.wbits}-{a.method.upper()} g{a.group_size}"
    lines = [f"## {cid}: full precision vs {tag} (mean ± std over training seeds)", ""]
    if cifar:
        lines += ["| Optimizer | seeds | FP test acc (%) | INT4 test acc (%) | Drop (pts) |", "|---|---|---|---|---|"]
        for o in order:
            d = opts[o]
            lines.append(f"| {o} | {d['seeds'] - d['diverged']}/{d['seeds']} | {fmt(mean_std(d['fp']['test_acc']))} | "
                         f"{fmt(mean_std(d['q']['test_acc']))} | {fmt(mean_std(d['rel']['test_acc']))} |")
    else:
        lines += ["| Optimizer | seeds | FineWeb PPL FP | FineWeb PPL INT4 | Δ% | WikiText-2 PPL FP | WikiText-2 PPL INT4 | Δ% |",
                  "|---|---|---|---|---|---|---|---|"]
        for o in order:
            d = opts[o]
            lines.append(f"| {o} | {d['seeds'] - d['diverged']}/{d['seeds']} | "
                         f"{fmt(mean_std(d['fp']['fineweb_ppl']))} | {fmt(mean_std(d['q']['fineweb_ppl']))} | {fmt(mean_std(d['rel']['fineweb_ppl']))} | "
                         f"{fmt(mean_std(d['fp']['wikitext2_ppl']))} | {fmt(mean_std(d['q']['wikitext2_ppl']))} | {fmt(mean_std(d['rel']['wikitext2_ppl']))} |")
    lines += ["", "seeds = training seeds with finite weights / total. Compare INT4 degradation only between "
                  "optimizers with similar full-precision quality: under-trained models trivially degrade less."]
    out = f"results/{cid}_summary.md"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
