"""Task 3: do teams that qualify take fewer yellow cards in the group stage?

Draws a stratified sample of 100 from the 144 group-stage team-match
observations, then reports descriptives, a 95% interval for the population mean,
and a two-sample t-test. See docs/METHOD.md.

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
from data import build_team_table, integrity_checks, file_sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
SEED = 42
SAMPLE_N = 100
N_TEAM_TASKS = 4


def rule(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def describe(s):
    s = pd.Series(s).dropna()
    return {"n": len(s), "mean": s.mean(), "median": s.median(),
            "sd": s.std(ddof=1), "se": s.std(ddof=1) / np.sqrt(len(s)),
            "min": s.min(), "max": s.max()}


def ci_mean(s, conf=0.95):
    s = pd.Series(s).dropna()
    se = s.std(ddof=1) / np.sqrt(len(s))
    return stats.t.interval(conf, len(s) - 1, loc=s.mean(), scale=se)


def two_sample(a, b, alpha=ALPHA):
    a, b = pd.Series(a), pd.Series(b)
    lev_W, lev_p = stats.levene(a, b)
    equal_var = lev_p >= alpha
    t, p = stats.ttest_ind(a, b, equal_var=equal_var)
    na, nb = len(a), len(b)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    df = se**4 / ((a.var(ddof=1)/na)**2 / (na-1) + (b.var(ddof=1)/nb)**2 / (nb-1))
    lo, hi = stats.t.interval(1 - alpha, df, loc=diff, scale=se)
    sp = np.sqrt(((na-1) * a.var(ddof=1) + (nb-1) * b.var(ddof=1)) / (na + nb - 2))
    return {"n_a": na, "n_b": nb, "mean_a": a.mean(), "mean_b": b.mean(),
            "sd_a": a.std(ddof=1), "sd_b": b.std(ddof=1), "diff": diff,
            "ci_low": lo, "ci_high": hi, "t": t, "df": df, "p": p,
            "d": diff / sp, "sp": sp, "levene_p": lev_p, "welch": not equal_var}


def main():
    rng = np.random.default_rng(SEED)

    # Step 1: wrangle the export and check it
    rule("1. DATA WRANGLING AND INTEGRITY CHECKS")
    tm, teams = build_team_table()
    checks = integrity_checks(tm, teams)
    for _, c in checks.iterrows():
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.check:<52} {c.detail}")
    if not checks.passed.all():
        raise SystemExit("integrity checks failed - analysis halted")

    tm = tm.merge(teams[["team", "qualified", "group"]], on="team", how="left")
    tm["yellow_cards"] = tm["yellow_cards"].astype(float)

    # Step 2: define the population and draw the stratified sample
    rule("2. DATA PREPARATION AND SAMPLING")
    print("  POPULATION")
    print("    All team-match observations from the FIFA World Cup 2026 group stage.")
    print(f"    N = {len(tm)}  (72 fixtures x 2 teams)")
    print(f"    Strata: qualified N = {int(tm.qualified.sum())}, "
          f"eliminated N = {int((~tm.qualified).sum())}")
    print("    Variable: yellow cards received by one team in one match.")
    print("    Every team played exactly 3 group matches, so exposure is equal.")

    print("\n  SAMPLING TECHNIQUE")
    print("    Proportionate stratified random sampling without replacement.")
    print("    Strata are the two comparison groups, unequal in the population")
    print("    (96 / 48). Proportionate allocation preserves that ratio and")
    print("    guarantees both strata are represented, which a simple random")
    print("    sample of this size does not.")
    print(f"    Target sample size n = {SAMPLE_N}, seed = {SEED}.")

    frac = SAMPLE_N / len(tm)
    parts = []
    for grp, g in tm.groupby("group", sort=True):
        k = int(round(len(g) * frac))
        idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
        parts.append(tm.loc[idx])
        print(f"    stratum {grp:<11} population {len(g):>3} -> sampled {k:>3} "
              f"({100 * k / len(g):.1f}%)")
    sample = pd.concat(parts).sort_index()
    print(f"    achieved sample size n = {len(sample)}")
    sample.to_csv(ROOT / "data" / "task3_sample.csv", index=False)

    sq = sample.loc[sample.qualified, "yellow_cards"]
    sel = sample.loc[~sample.qualified, "yellow_cards"]

    # Step 3: describe the sample
    rule("3. DESCRIPTIVE STATISTICS (SAMPLE)")
    overall = describe(sample.yellow_cards)
    print("  Overall sample")
    print(f"    n={overall['n']}  mean={overall['mean']:.4f}  "
          f"median={overall['median']:.1f}  sd={overall['sd']:.4f}  "
          f"se={overall['se']:.4f}  range {overall['min']:.0f}-{overall['max']:.0f}")

    print("\n  By stratum")
    for name, s in (("Qualified ", sq), ("Eliminated", sel)):
        d = describe(s)
        print(f"    {name}  n={d['n']:>2}  mean={d['mean']:.4f}  "
              f"median={d['median']:.1f}  sd={d['sd']:.4f}  se={d['se']:.4f}  "
              f"range {d['min']:.0f}-{d['max']:.0f}")

    print("\n  Distribution of the count (sample)")
    vc = sample.yellow_cards.value_counts().sort_index()
    print("    " + "   ".join(f"{int(k)} cards: {v}" for k, v in vc.items()))

    # Step 4: estimate the population mean with a 95% interval
    rule("4. INFERENTIAL STATISTICS - CONFIDENCE INTERVAL")
    lo, hi = ci_mean(sample.yellow_cards)
    print("  95% confidence interval for the POPULATION mean yellow cards per")
    print("  team-match, estimated from the sample:")
    print(f"    point estimate   {overall['mean']:.4f}")
    print(f"    95% CI           [{lo:.4f}, {hi:.4f}]")
    print(f"    df = {overall['n'] - 1}, standard error = {overall['se']:.4f}")

    pop_mean = tm.yellow_cards.mean()
    covered = lo <= pop_mean <= hi
    print(f"\n    Actual population mean (the frame is complete here): {pop_mean:.4f}")
    print(f"    The interval {'contains' if covered else 'does NOT contain'} it.")
    print("    Shown because this frame happens to be complete. In the usual")
    print("    case the value is unknown and is exactly what the interval estimates.")

    # Step 5: compare the two strata
    rule("5. INFERENTIAL STATISTICS - TWO-SAMPLE t-TEST")
    for name, s in (("Qualified ", sq), ("Eliminated", sel)):
        W, p = stats.shapiro(s)
        print(f"  Shapiro-Wilk {name}: W={W:.4f}, p={p:.4f}, skew={stats.skew(s):+.3f}")
    print("  Yellow cards are a small-integer count, so exact normality is not")
    print("  expected. The permutation check in section 6 answers that concern.")

    r = two_sample(sq, sel)
    print("\n  H0: mu_qualified = mu_eliminated")
    print(f"  H1: mu_qualified != mu_eliminated              alpha = {ALPHA}")
    print(f"\n  Levene: p={r['levene_p']:.4f} -> "
          f"{'Welch' if r['welch'] else 'pooled (Student)'} t-test")
    print(f"  Qualified : n={r['n_a']}, mean={r['mean_a']:.4f}, sd={r['sd_a']:.4f}")
    print(f"  Eliminated: n={r['n_b']}, mean={r['mean_b']:.4f}, sd={r['sd_b']:.4f}")
    print(f"\n  Difference in means  : {r['diff']:+.4f} yellow cards per team-match")
    print(f"  95% CI for difference: [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    print(f"  t = {r['t']:.4f}, df = {r['df']:.2f}, p = {r['p']:.6f}")
    print(f"  Cohen's d = {r['d']:+.4f}")
    print(f"\n  Decision: {'reject H0' if r['p'] < ALPHA else 'fail to reject H0'}")

    # Step 6: check the conclusion against weaker assumptions
    rule("6. ROBUSTNESS AND SENSITIVITY")
    u, up = stats.mannwhitneyu(sq, sel, alternative="two-sided")
    print(f"  Mann-Whitney U (no normality assumption): U={u:.1f}, p={up:.6f} "
          f"-> {'same' if (up < ALPHA) == (r['p'] < ALPHA) else 'DIFFERENT'} conclusion")

    pooled = np.concatenate([sq.to_numpy(), sel.to_numpy()])
    na = len(sq)
    perm = np.empty(10000)
    for i in range(10000):
        rng.shuffle(pooled)
        perm[i] = pooled[:na].mean() - pooled[na:].mean()
    perm_p = float(np.mean(np.abs(perm) >= abs(r["diff"])))
    print(f"  Permutation test (10,000 relabellings): p={perm_p:.6f} "
          f"-> {'same' if (perm_p < ALPHA) == (r['p'] < ALPHA) else 'DIFFERENT'} conclusion")

    r2 = np.random.default_rng(SEED + 1)
    draws = np.empty(2000)
    for j in range(2000):
        ps = []
        for _, g in tm.groupby("group", sort=True):
            k = int(round(len(g) * frac))
            ps.append(g.iloc[r2.choice(len(g), k, replace=False)])
        s2 = pd.concat(ps)
        draws[j] = stats.ttest_ind(s2.loc[s2.qualified, "yellow_cards"],
                                   s2.loc[~s2.qualified, "yellow_cards"],
                                   equal_var=True).pvalue
    print(f"  Repeating the sampling design 2,000 times: "
          f"{100 * np.mean(draws < ALPHA):.1f}% of samples would reject H0")

    cq = tm.loc[tm.qualified, "yellow_cards"]
    ce = tm.loc[~tm.qualified, "yellow_cards"]
    cen = two_sample(cq, ce)
    print(f"\n  SENSITIVITY - full population (N={len(tm)}, no sampling):")
    print(f"    qualified mean {cen['mean_a']:.4f} vs eliminated {cen['mean_b']:.4f}")
    print(f"    difference {cen['diff']:+.4f}, 95% CI "
          f"[{cen['ci_low']:+.4f}, {cen['ci_high']:+.4f}], "
          f"t={cen['t']:.4f}, p={cen['p']:.6f}, d={cen['d']:+.4f}")
    print(f"    -> {'same' if (cen['p'] < ALPHA) == (r['p'] < ALPHA) else 'DIFFERENT'} "
          "conclusion as the sample")

    # Step 7: report what this design could have detected
    rule("7. WHAT THIS DESIGN COULD DETECT")
    df_p = r["n_a"] + r["n_b"] - 2
    crit = stats.t.ppf(1 - ALPHA / 2, df_p)

    def power_at(delta):
        ncp = delta * np.sqrt(r["n_a"] * r["n_b"] / (r["n_a"] + r["n_b"]))
        return 1 - stats.nct.cdf(crit, df_p, ncp) + stats.nct.cdf(-crit, df_p, ncp)

    achieved = power_at(abs(r["d"]))
    mde_d = brentq(lambda x: power_at(x) - 0.80, 0.01, 2.0)
    print(f"  Achieved power against the observed effect (d={abs(r['d']):.3f}): "
          f"{achieved:.3f}")
    print(f"  Smallest effect detectable at 80% power: d={mde_d:.3f} "
          f"= {mde_d * r['sp']:.3f} cards per team-match")
    print("  The whole group stage holds 144 team-match observations. The ceiling")
    print("  on precision is the competition, not the analysis.")

    bonf = min(1.0, r["p"] * N_TEAM_TASKS)
    print(f"\n  Bonferroni-adjusted p across {N_TEAM_TASKS} team tasks: {bonf:.6f}")
    print("  Reported for team integration. This task does not recompute the")
    print("  other members' tests or apply a Holm adjustment across them.")

    # Step 8: draw the figure
    rule("8. FIGURE")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.3))

    width = 0.38
    levels = sorted(tm.yellow_cards.unique())
    for k, (lab, s, col) in enumerate([("Eliminated", sel, "#C0504D"),
                                       ("Qualified", sq, "#4C78A8")]):
        prop = [float(np.mean(s == v)) for v in levels]
        ax1.bar(np.arange(len(levels)) + (k - 0.5) * width, prop, width,
                label=f"{lab} (n={len(s)})", color=col, alpha=0.85)
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels([int(v) for v in levels])
    ax1.set_xlabel("Yellow cards in a match")
    ax1.set_ylabel("Proportion of observations")
    ax1.set_title("Sample distribution")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    bp = ax2.boxplot([sel, sq], widths=0.55, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#C0504D", "#4C78A8"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
    jit = np.random.default_rng(SEED)
    for i, s in enumerate([sel, sq], start=1):
        ax2.scatter(jit.normal(i, 0.06, len(s)), s + jit.normal(0, 0.05, len(s)),
                    s=16, alpha=0.6, color="#333333", zorder=3)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f"Eliminated\n(n={len(sel)})", f"Qualified\n(n={len(sq)})"])
    ax2.set_ylabel("Yellow cards per match")
    ax2.set_title("Sample by stratum")
    ax2.grid(axis="y", alpha=0.25)

    ax3.axhline(0, color="black", lw=1)
    for i, (lab, res, col) in enumerate([("sample", r, "#4C78A8"),
                                         ("population", cen, "#7F7F7F")]):
        ax3.errorbar(i, res["diff"],
                     yerr=[[res["diff"] - res["ci_low"]], [res["ci_high"] - res["diff"]]],
                     fmt="o", ms=9, capsize=8, color=col, lw=2)
        ax3.text(i + 0.13, res["diff"], f"{res['diff']:+.3f}", va="center", fontsize=9)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels([f"Sample\n(n={len(sample)})", f"Population\n(N={len(tm)})"])
    ax3.set_xlim(-0.5, 1.7)
    ax3.set_ylabel("Qualified − eliminated (cards/match)")
    ax3.set_title("Difference with 95% CI")
    ax3.grid(axis="y", alpha=0.25)

    fig.suptitle("Task 3: group-stage discipline and knockout qualification "
                 "(FIFA World Cup 2026)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "task3.png", dpi=300)
    plt.close(fig)
    print("  saved figures/task3.png")

    # Step 9: write the outputs
    teams.to_csv(ROOT / "data" / "task3_team_level.csv", index=False)
    tm.to_csv(ROOT / "data" / "task3_population.csv", index=False)
    checks.to_csv(ROOT / "data" / "task3_integrity_checks.csv", index=False)

    num = lambda v: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
    out = {
        "task": "Task 3 - disciplinary record and knockout qualification",
        "author": "Ritesh Panta",
        "focal_point": "disciplinary record (yellow cards)",
        "run_date": date.today().isoformat(),
        "alpha": ALPHA, "seed": SEED,
        "population_N": len(tm),
        "population_strata": {"qualified": int(tm.qualified.sum()),
                              "eliminated": int((~tm.qualified).sum())},
        "sampling": "proportionate stratified random sampling without replacement",
        "sample_n": len(sample),
        "sample_mean": num(overall["mean"]), "sample_sd": num(overall["sd"]),
        "ci_95_population_mean": [num(lo), num(hi)],
        "population_mean_actual": num(pop_mean),
        "ci_covers_population_mean": bool(covered),
        "test": {k: num(v) for k, v in r.items()},
        "mann_whitney_p": float(up), "permutation_p": perm_p,
        "repeat_sampling_reject_rate": num(np.mean(draws < ALPHA)),
        "census_sensitivity": {k: num(v) for k, v in cen.items()},
        "achieved_power": num(achieved),
        "min_detectable_d_80pct": num(mde_d),
        "min_detectable_cards_80pct": num(mde_d * r["sp"]),
        "bonferroni_p_4_tasks": round(float(bonf), 6),
        "decision": "reject H0" if r["p"] < ALPHA else "fail to reject H0",
        "data_sha256": file_sha256(),
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
    }
    (ROOT / "data" / "task3_results.json").write_text(json.dumps(out, indent=2))

    rule("9. SUMMARY")
    print(f"  Population : {len(tm)} group-stage team-match observations")
    print(f"  Sample     : {len(sample)}, proportionate stratified, seed {SEED}")
    print(f"  95% CI for population mean cards/match: [{lo:.3f}, {hi:.3f}]")
    print(f"  Qualified {r['mean_a']:.3f} vs eliminated {r['mean_b']:.3f} cards/match")
    print(f"  Difference {r['diff']:+.3f}, 95% CI "
          f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], p={r['p']:.4f}, d={r['d']:+.3f}")
    print(f"  {out['decision'].upper()} at alpha={ALPHA}.")
    print("\n  Outputs written to data/ and figures/.")


if __name__ == "__main__":
    main()
