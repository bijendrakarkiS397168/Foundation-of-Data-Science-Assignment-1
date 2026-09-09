"""Task 4: do teams score more group-stage goals than their xG expects?

Draws a simple random sample of 100 from the 144 group-stage team-match
observations, then reports descriptives, a 95% interval for the population mean
finishing differential, and a one-sample t-test against zero. See
docs/METHOD.md.

    python src/analysis.py
"""

import json
import platform
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import build_team_match, integrity_checks, file_sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
SEED = 7
SAMPLE_N = 100
BENCHMARK = 0.0
N_TEAM_TASKS = 4


def rule(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def describe(s):
    """Returns the standard descriptive summary of a series."""
    s = pd.Series(s).dropna()
    return {"n": len(s), "mean": s.mean(), "median": s.median(),
            "sd": s.std(ddof=1), "se": s.std(ddof=1) / np.sqrt(len(s)),
            "min": s.min(), "max": s.max()}


def ci_mean(s, conf=0.95):
    """Returns a t interval for the mean of a series."""
    s = pd.Series(s).dropna()
    se = s.std(ddof=1) / np.sqrt(len(s))
    return stats.t.interval(conf, len(s) - 1, loc=s.mean(), scale=se)


def one_sample(s, mu0):
    """Runs a two-sided one-sample t-test against mu0."""
    s = pd.Series(s).dropna()
    t, p = stats.ttest_1samp(s, mu0)
    lo, hi = ci_mean(s)
    return {"n": len(s), "mean": s.mean(), "sd": s.std(ddof=1),
            "ci_low": lo, "ci_high": hi, "t": t, "df": len(s) - 1, "p": p,
            "d": (s.mean() - mu0) / s.std(ddof=1)}


def main():
    rng = np.random.default_rng(SEED)

    # Step 1: wrangle the export and check it
    rule("1. DATA WRANGLING AND INTEGRITY CHECKS")
    tm = build_team_match()
    checks = integrity_checks(tm)
    for _, c in checks.iterrows():
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.check:<50} {c.detail}")
    if not checks.passed.all():
        raise SystemExit("integrity checks failed, analysis halted")

    # Step 2: define the population and draw the sample
    rule("2. DATA PREPARATION AND SAMPLING")
    print("  POPULATION")
    print("    All team-match observations from the FIFA World Cup 2026 group stage.")
    print(f"    N = {len(tm)}  (72 fixtures x 2 teams)")
    print("    Variable: finishing differential = goals scored minus xG.")
    print("    A positive value means a team scored more than its chances expected.")
    print(f"    Population mean goals {tm.goals.mean():.4f}, "
          f"mean xG {tm.xg.mean():.4f}")

    print("\n  SAMPLING TECHNIQUE")
    print("    Simple random sampling without replacement.")
    print("    This question estimates one overall mean rather than comparing")
    print("    groups, so the frame has no strata to preserve and a simple")
    print("    random draw is the appropriate technique.")
    print(f"    Sample size n = {SAMPLE_N}, seed = {SEED}.")

    idx = rng.choice(tm.index.to_numpy(), size=SAMPLE_N, replace=False)
    sample = tm.loc[idx].sort_index()
    sample.to_csv(ROOT / "data" / "task4_sample.csv", index=False)
    print(f"    achieved sample size n = {len(sample)} "
          f"({100 * len(sample) / len(tm):.1f}% of the frame)")

    fin = sample["finishing"]

    # Step 3: describe the sample
    rule("3. DESCRIPTIVE STATISTICS (SAMPLE)")
    d = describe(fin)
    print("  Finishing differential (goals minus xG)")
    print(f"    n={d['n']}  mean={d['mean']:.4f}  median={d['median']:.4f}  "
          f"sd={d['sd']:.4f}  se={d['se']:.4f}")
    print(f"    range {d['min']:.3f} to {d['max']:.3f}")

    over = int((fin > 0).sum())
    under = int((fin < 0).sum())
    print(f"\n    outscored their xG : {over} observations "
          f"({100 * over / len(fin):.1f}%)")
    print(f"    fell short of xG   : {under} observations "
          f"({100 * under / len(fin):.1f}%)")

    print("\n  Components")
    for name, col in (("goals", "goals"), ("xG", "xg")):
        c = describe(sample[col])
        print(f"    {name:<6} mean={c['mean']:.4f}  sd={c['sd']:.4f}  "
              f"range {c['min']:.2f} to {c['max']:.2f}")

    # Step 4: estimate the population mean with a 95% interval
    rule("4. INFERENTIAL STATISTICS - CONFIDENCE INTERVAL")
    lo, hi = ci_mean(fin)
    print("  95% confidence interval for the POPULATION mean finishing")
    print("  differential, estimated from the sample:")
    print(f"    point estimate   {d['mean']:.4f} goals")
    print(f"    95% CI           [{lo:.4f}, {hi:.4f}]")
    print(f"    df = {d['n'] - 1}, standard error = {d['se']:.4f}")

    pop_mean = tm.finishing.mean()
    covered = lo <= pop_mean <= hi
    print(f"\n    Actual population mean (this frame is complete): {pop_mean:.4f}")
    print(f"    The interval {'contains' if covered else 'does NOT contain'} it.")

    # Step 5: test the sample mean against the benchmark
    rule("5. INFERENTIAL STATISTICS - ONE-SAMPLE t-TEST")
    W, sw_p = stats.shapiro(fin)
    print(f"  Shapiro-Wilk: W={W:.4f}, p={sw_p:.4f}, "
          f"skew={stats.skew(fin):+.3f}, excess kurtosis={stats.kurtosis(fin):+.3f}")
    print("  The differential is a count minus a continuous expectation, so it")
    print("  is lumpy rather than smooth. Step 6 answers that with a test that")
    print("  assumes no distribution.")

    r = one_sample(fin, BENCHMARK)
    print(f"\n  H0: mu = {BENCHMARK:.0f}   (teams score exactly what their chances expect)")
    print(f"  H1: mu != {BENCHMARK:.0f}                              alpha = {ALPHA}")
    print(f"\n  n = {r['n']}, mean = {r['mean']:.4f}, sd = {r['sd']:.4f}")
    print(f"  95% CI for the mean: [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print(f"  t({r['df']}) = {r['t']:.4f}, p = {r['p']:.6f}")
    print(f"  Cohen's d = {r['d']:.4f}")
    print(f"\n  Decision: {'reject H0' if r['p'] < ALPHA else 'fail to reject H0'}")

    # Step 6: check the conclusion against weaker assumptions
    rule("6. ROBUSTNESS AND SENSITIVITY")
    w_stat, w_p = stats.wilcoxon(fin - BENCHMARK)
    print(f"  Wilcoxon signed-rank (no normality assumption): W={w_stat:.1f}, "
          f"p={w_p:.6f} -> "
          f"{'same' if (w_p < ALPHA) == (r['p'] < ALPHA) else 'DIFFERENT'} conclusion")

    boot = np.array([rng.choice(fin, len(fin), replace=True).mean()
                     for _ in range(10000)])
    b_lo, b_hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Bootstrap 95% CI (10,000 resamples): [{b_lo:.4f}, {b_hi:.4f}]")

    cen = one_sample(tm.finishing, BENCHMARK)
    print(f"\n  SENSITIVITY - full population (N={len(tm)}, no sampling):")
    print(f"    mean {cen['mean']:.4f}, 95% CI [{cen['ci_low']:.4f}, "
          f"{cen['ci_high']:.4f}], t={cen['t']:.4f}, p={cen['p']:.6f}, "
          f"d={cen['d']:.4f}")
    print(f"    -> {'same' if (cen['p'] < ALPHA) == (r['p'] < ALPHA) else 'DIFFERENT'} "
          "conclusion as the sample")

    r2 = np.random.default_rng(SEED + 1)
    draws = np.array([stats.ttest_1samp(
        tm.finishing.to_numpy()[r2.choice(len(tm), SAMPLE_N, replace=False)],
        BENCHMARK).pvalue for _ in range(2000)])
    reject_rate = float(np.mean(draws < ALPHA))
    print(f"  Repeating the sampling design 2,000 times: "
          f"{100 * reject_rate:.1f}% of draws reject H0")
    if 0.3 < reject_rate < 0.7:
        print("    This design sits near the decision boundary. The sample result")
        print("    is therefore fragile: a different draw could land the other way.")
        print("    The seed was fixed in advance and has not been changed to")
        print("    obtain a preferred p-value.")

    # Step 7: report what this design could have detected
    rule("7. WHAT THIS DESIGN COULD DETECT")
    crit = stats.t.ppf(1 - ALPHA / 2, r["df"])

    def power_at(delta):
        # Guard the noncentral t: it overflows to NaN for large noncentrality,
        # where power is 1 to within floating-point precision anyway.
        ncp = delta * np.sqrt(r["n"])
        if ncp > 8:
            return 1.0
        return 1 - stats.nct.cdf(crit, r["df"], ncp) + stats.nct.cdf(-crit, r["df"], ncp)

    achieved = power_at(abs(r["d"]))
    mde_d = brentq(lambda x: power_at(x) - 0.80, 0.001, 0.8)
    print(f"  Achieved power against the observed effect (d={abs(r['d']):.3f}): "
          f"{achieved:.3f}")
    print(f"  Smallest effect detectable at 80% power: d={mde_d:.3f} "
          f"= {mde_d * r['sd']:.3f} goals per team-match")

    bonf = min(1.0, r["p"] * N_TEAM_TASKS)
    print(f"\n  Bonferroni-adjusted p across {N_TEAM_TASKS} team tasks: {bonf:.6f}")
    print(f"  The result {'survives' if bonf < ALPHA else 'does not survive'} "
          "that adjustment.")
    print("  This task does not recompute the other members' tests or apply a")
    print("  Holm adjustment across them.")

    # Step 8: draw the figure
    rule("8. FIGURE")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.3))

    ax1.hist(fin, bins=18, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax1.axvline(0, color="black", lw=1.5, label="xG expectation")
    ax1.axvline(fin.mean(), color="#C0504D", lw=2, ls="--",
                label=f"sample mean {fin.mean():.3f}")
    ax1.set_xlabel("Goals minus xG")
    ax1.set_ylabel("Observations")
    ax1.set_title("Finishing differential")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    ax2.scatter(sample.xg, sample.goals, s=30, alpha=0.65, color="#4C78A8")
    lim = [0, max(sample.xg.max(), sample.goals.max()) + 0.4]
    ax2.plot(lim, lim, color="black", lw=1.5, ls="--", label="goals = xG")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("Expected goals (xG)")
    ax2.set_ylabel("Goals scored")
    ax2.set_title("Points above the line outscored their chances")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    ax3.axvline(0, color="black", lw=1)
    for i, (lab, res, col) in enumerate([("sample", r, "#4C78A8"),
                                         ("population", cen, "#7F7F7F")]):
        ax3.errorbar(res["mean"], i,
                     xerr=[[res["mean"] - res["ci_low"]],
                           [res["ci_high"] - res["mean"]]],
                     fmt="o", ms=9, capsize=8, color=col, lw=2)
        ax3.text(res["mean"], i + 0.16, f"{res['mean']:+.3f}",
                 ha="center", fontsize=9)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels([f"Sample\n(n={len(sample)})", f"Population\n(N={len(tm)})"])
    ax3.set_ylim(-0.6, 1.6)
    ax3.set_xlabel("Mean goals minus xG")
    ax3.set_title("Estimate with 95% CI")
    ax3.grid(axis="x", alpha=0.25)

    fig.suptitle("Task 4: finishing against expectation (FIFA World Cup 2026 "
                 "group stage)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "task4.png", dpi=300)
    plt.close(fig)
    print("  saved figures/task4.png")

    # Step 9: write the outputs
    tm.to_csv(ROOT / "data" / "task4_population.csv", index=False)
    checks.to_csv(ROOT / "data" / "task4_integrity_checks.csv", index=False)

    num = lambda v: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
    out = {
        "task": "Task 4 - finishing against expected goals",
        "author": "Kalyan Acharya",
        "focal_point": "finishing relative to chance quality (goals against xG)",
        "run_date": date.today().isoformat(),
        "alpha": ALPHA, "seed": SEED, "benchmark": BENCHMARK,
        "population_N": len(tm),
        "sampling": "simple random sampling without replacement",
        "sample_n": len(sample),
        "sample_mean_goals": num(sample.goals.mean()),
        "sample_mean_xg": num(sample.xg.mean()),
        "test": {k: num(v) for k, v in r.items()},
        "ci_95_population_mean": [num(lo), num(hi)],
        "population_mean_actual": num(pop_mean),
        "ci_covers_population_mean": bool(covered),
        "shapiro_p": num(sw_p),
        "wilcoxon_p": float(w_p),
        "bootstrap_ci_95": [num(b_lo), num(b_hi)],
        "census_sensitivity": {k: num(v) for k, v in cen.items()},
        "repeat_sampling_reject_rate": num(np.mean(draws < ALPHA)),
        "achieved_power": num(achieved),
        "min_detectable_d_80pct": num(mde_d),
        "min_detectable_goals_80pct": num(mde_d * r["sd"]),
        "bonferroni_p_4_tasks": round(float(bonf), 6),
        "decision": "reject H0" if r["p"] < ALPHA else "fail to reject H0",
        "data_sha256": file_sha256(),
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
    }
    (ROOT / "data" / "task4_results.json").write_text(json.dumps(out, indent=2))

    rule("9. SUMMARY")
    print(f"  Population : {len(tm)} group-stage team-match observations")
    print(f"  Sample     : {len(sample)}, simple random, seed {SEED}")
    print(f"  Mean finishing differential {r['mean']:+.4f} goals, "
          f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    print(f"  t({r['df']}) = {r['t']:.4f}, p = {r['p']:.4f}, d = {r['d']:+.4f}")
    print(f"  {out['decision'].upper()} at alpha={ALPHA}.")
    print("\n  Outputs written to data/ and figures/.")


if __name__ == "__main__":
    main()
