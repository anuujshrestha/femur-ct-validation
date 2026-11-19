"""Final analysis: exclusions, paired statistics, femoral-head agreement, and the two summary figures.

Reads the CSVs written by evaluate.py and quality_checks.py and writes
  results/analysis.md                     - all tables quoted in the README and the report
  results/femoral_head_with_qc.csv        - head results for native hips, with the fit-check flag
  results/figures/head_error_vs_dice.png
  results/figures/bland_altman_head_diameter_qc.png

Exclusions from head sizing: hips with an implant or a cut-off head (quality_checks.py).
Fit check: a predicted head fit supported by fewer than --min_fit_points surface points is
counted as a fit failure. This threshold was chosen AFTER seeing the data (exploratory);
it must be validated on new scans before use.

Usage: python scripts/analyse.py results [--cases cases.csv]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"full": "#1f77b4", "fast": "#d62728"}


def med_iqr(x, nd=3):
    return f"{np.median(x):.{nd}f} [{np.percentile(x, 25):.{nd}f}–{np.percentile(x, 75):.{nd}f}]"


def wilcoxon_p(a, b):
    return stats.wilcoxon(a, b).pvalue if (np.asarray(a) != np.asarray(b)).any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--min_fit_points", type=int, default=2500)
    a = ap.parse_args()
    R = a.results
    (R / "figures").mkdir(parents=True, exist_ok=True)

    seg = pd.read_csv(R / "segmentation_metrics.csv")
    head = pd.read_csv(R / "femoral_head.csv")
    fl = pd.read_csv(R / "implant_flags.csv").merge(pd.read_csv(R / "head_fov_flags.csv"), on=["image_id", "side"])
    fl["exclude"] = fl.implant | fl.head_truncated
    fit = pd.read_csv(R / "pred_fit_qc.csv")
    seg["side"] = seg.bone.str.split("_").str[1]
    seg = seg.merge(fl[["image_id", "side", "exclude"]], on=["image_id", "side"])
    out = []
    w = out.append

    n_cases = seg.image_id.nunique()
    splits = seg.drop_duplicates("image_id").split.value_counts().to_dict()
    w(f"# Analysis\n\nCases: {n_cases} (splits: {splits}). Medians [IQR]; p = paired Wilcoxon signed-rank, normal vs fast.\n")
    ex = fl[fl.exclude]
    w(f"Excluded from head sizing: {len(ex)} of {len(fl)} hips "
      f"({int(fl.implant.sum())} implant, {int((fl.head_truncated & ~fl.implant).sum())} cut-off head only):\n")
    w(ex[["image_id", "side", "metal_voxels", "head_below_mm"]].to_markdown(index=False) + "\n")

    # Segmentation
    for label, s in [("all sides", seg), ("native sides only (implant and cut-off sides removed)", seg[~seg.exclude])]:
        w(f"\n## Segmentation, {label}\n")
        rows = []
        for bone, d in s.groupby("bone"):
            p = d.pivot(index="image_id", columns="mode", values=["dice", "assd", "hd95"])
            row = {"bone": bone, "n": len(p)}
            for m, nd in [("dice", 3), ("assd", 2), ("hd95", 2)]:
                row[f"{m} normal"] = med_iqr(p[(m, "full")], nd)
                row[f"{m} fast"] = med_iqr(p[(m, "fast")], nd)
                row[f"p {m}"] = f"{wilcoxon_p(p[(m, 'full')], p[(m, 'fast')]):.1g}"
            rows.append(row)
        w(pd.DataFrame(rows).to_markdown(index=False) + "\n")
        fem = s[s.bone.str.startswith("femur")].groupby("mode")[["dice", "assd"]].median()
        w(f"\nPooled femur medians: Dice {fem.loc['full', 'dice']:.3f} normal vs {fem.loc['fast', 'dice']:.3f} fast; "
          f"ASSD {fem.loc['full', 'assd']:.2f} vs {fem.loc['fast', 'assd']:.2f} mm "
          f"(fast/normal = {fem.loc['fast', 'assd'] / fem.loc['full', 'assd']:.1f}x).\n")

    # Femoral head, native complete heads
    h = head.merge(fl[["image_id", "side", "exclude"]], on=["image_id", "side"])
    h = h.merge(fit[fit.source != "gt"].rename(columns={"source": "mode"})[["image_id", "side", "mode", "n_points", "fit_rmse_mm"]],
                on=["image_id", "side", "mode"])
    h = h.merge(seg[seg.bone.str.startswith("femur")][["image_id", "side", "mode", "dice", "assd"]], on=["image_id", "side", "mode"])
    nat = h[~h.exclude].copy()
    nat["fit_fail"] = nat.n_points < a.min_fit_points
    nat.to_csv(R / "femoral_head_with_qc.csv", index=False)
    g = nat[nat["mode"] == nat["mode"].iloc[0]]
    w(f"\n## Femoral head (native, complete heads: n = {len(g)} per model)\n")
    w(f"GT diameter mean {g.gt_diam_mm.mean():.1f} mm (SD {g.gt_diam_mm.std():.1f}, range {g.gt_diam_mm.min():.1f}–{g.gt_diam_mm.max():.1f}); "
      f"GT sphere-fit RMSE median {g.gt_fit_rmse_mm.median():.2f} mm.\n")
    rows = []
    for mode, d in nat.groupby("mode"):
        for lab, e in [("all native", d), (f"fit check passed (>= {a.min_fit_points} points)", d[~d.fit_fail])]:
            x, c = e.diam_err_mm, e.centre_err_mm
            rows.append({"model": "normal" if mode == "full" else mode, "subset": lab, "n": len(x),
                         "median abs diam err (mm)": round(x.abs().median(), 2), "max abs diam err (mm)": round(x.abs().max(), 2),
                         "bias (mm)": round(x.mean(), 2), "LoA ± (mm)": round(1.96 * x.std(), 2),
                         "median centre err (mm)": round(c.median(), 2), "diam err > 1 mm": int((x.abs() > 1).sum())})
    w(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    pw = nat.pivot_table(index=["image_id", "side"], columns="mode", values=["diam_err_mm", "centre_err_mm"])
    w(f"\nPaired normal vs fast (all native heads): |diameter error| p = {stats.wilcoxon(pw[('diam_err_mm', 'fast')].abs(), pw[('diam_err_mm', 'full')].abs()).pvalue:.1g}; "
      f"centre error p = {stats.wilcoxon(pw[('centre_err_mm', 'fast')], pw[('centre_err_mm', 'full')]).pvalue:.1g}.\n")
    for mode, d in nat.groupby("mode"):
        rho, p = stats.spearmanr(d.diam_err_mm.abs(), d.dice)
        w(f"Spearman(|diameter error|, femur Dice), {mode}: rho = {rho:+.2f}, p = {p:.2f}.\n")
    w("\n### Fit failures (fewer than the minimum number of fit points)\n")
    w(nat[nat.fit_fail][["image_id", "side", "mode", "dice", "gt_diam_mm", "pred_diam_mm", "diam_err_mm", "centre_err_mm", "n_points", "fit_rmse_mm"]].round(3).to_markdown(index=False) + "\n")
    ok = nat[~nat.fit_fail]
    w(f"\nPassing fits use at least {int(ok.n_points.min())} points (max fit RMSE {ok.fit_rmse_mm.max():.2f} mm).\n")
    w("\n### Excluded hips (for reference)\n")
    w(h[h.exclude][["image_id", "side", "mode", "gt_diam_mm", "pred_diam_mm", "diam_err_mm", "centre_err_mm"]].round(2).to_markdown(index=False) + "\n")
    (R / "analysis.md").write_text("\n".join(out))
    print("\n".join(out))

    # Figures
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in ["full", "fast"]:
        d = nat[nat["mode"] == m]
        ax.scatter(d.dice, d.diam_err_mm.abs(), s=22, c=COL[m], alpha=.8,
                   label=f"{'normal' if m == 'full' else 'fast'} ({'1.5' if m == 'full' else '3'} mm model)")
        for _, r in d[d.fit_fail].iterrows():
            ax.annotate(f"{r.image_id} {r.side[0].upper()}", (r.dice, abs(r.diam_err_mm)), xytext=(4, 2), textcoords="offset points", fontsize=7)
    ax.set(xlabel="femur Dice", ylabel="|femoral-head diameter error| (mm)",
           title=f"Dice does not predict head-sizing error ({len(g)} native hips)")
    ax.axhline(1, ls=":", c="grey", lw=1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "figures" / "head_error_vs_dice.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, m in zip(axes, ["full", "fast"]):
        d = ok[ok["mode"] == m]
        mean, diff = (d.gt_diam_mm + d.pred_diam_mm) / 2, d.pred_diam_mm - d.gt_diam_mm
        b, s = diff.mean(), diff.std()
        ax.scatter(mean, diff, s=18, c=COL[m])
        for v, ls in [(b, "-"), (b + 1.96 * s, "--"), (b - 1.96 * s, "--")]:
            ax.axhline(v, ls=ls, c="k", lw=1)
        ax.set(title=f"{'normal' if m == 'full' else 'fast'}: bias {b:+.2f} mm, LoA ±{1.96 * s:.2f} mm (n={len(d)})",
               xlabel="mean head diameter (mm)")
    axes[0].set_ylabel("pred − GT (mm)")
    fig.suptitle("Femoral-head diameter agreement: native, complete heads that pass the fit check", fontsize=10)
    fig.tight_layout()
    fig.savefig(R / "figures" / "bland_altman_head_diameter_qc.png", dpi=150)


if __name__ == "__main__":
    main()
