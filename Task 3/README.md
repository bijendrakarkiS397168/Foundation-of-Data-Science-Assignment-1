# Task 3: discipline and knockout qualification

FIFA World Cup 2026, Objective 1. Ritesh Panta.
Focal point: **disciplinary record**. Task 1 covers corners; Task 2 covers shots
and shot accuracy.

> During the FIFA World Cup 2026 group stage, do teams that went on to qualify
> for the knockout stage take fewer yellow cards per match than teams knocked
> out in the group stage?

## Design

| Element | Value |
| --- | --- |
| Population | 144 team-match observations: 72 group-stage fixtures × 2 teams |
| Strata | 96 qualified, 48 eliminated |
| Sampling | Proportionate stratified random, without replacement |
| Sample | n = 100: 67 qualified, 33 eliminated. Seed 42 |
| Test | Two-sample *t*-test, α = 0.05, two-sided |

Every team played exactly three group matches, so exposure is equal and the
count needs no rate adjustment.

## Results

**Confidence interval.** Population mean yellow cards per team-match, estimated
from the sample: **1.160, 95% CI [0.957, 1.363]**, df = 99. The true population
mean is 1.236 and falls inside the interval.

**Two-sample t-test.**

| Measure | Qualified | Eliminated |
| --- | --- | --- |
| n | 67 | 33 |
| Mean cards per match | 1.045 | 1.394 |
| SD | 0.912 | 1.197 |
| Median | 1.0 | 1.0 |

Difference −0.349, 95% CI [−0.824, +0.125].
*t*(50.88) = −1.619, *p* = 0.109, Cohen's *d* = −0.344.
**The test fails to reject H₀.**

Qualified teams took fewer cards on average, but the interval crosses zero. The
finding is *no evidence of a difference*, a weaker claim than *evidence of no
difference*.

## Robustness

| Check | Result | Same conclusion |
| --- | --- | --- |
| Mann-Whitney U | p = 0.190 | yes |
| Permutation test, 10,000 relabellings | p = 0.120 | yes |
| Full population, no sampling | −0.302, p = 0.111 | yes |

Shapiro-Wilk rejects normality in both strata, which is what a small-integer
count does. The permutation test assumes no distribution and agrees, so the
conclusion does not rest on normality. The census check confirms sampling did
not bend the estimate.

## Why "no evidence" and not "no difference"

The smallest effect this design detects at 80% power is *d* = 0.602, about 0.61
cards per match. The observed effect is *d* = 0.344, and power against it is
0.361, so a real difference that size would slip past this study about two times
in three.

More data cannot fix it. A World Cup group stage holds 144 team-match
observations in total, so the competition sets the ceiling, not the analysis.

## Run the analysis

1.  Install the dependencies:

    ```bash
    pip install pandas numpy scipy matplotlib
    ```

2.  Run the script from the task root:

    ```bash
    python src/analysis.py
    ```

The script checks the source file's SHA-256 and runs eleven integrity checks
before computing any statistic. If one fails, it stops instead of returning a
number. All seeds are fixed, so the sample, the permutation test, and the figure
reproduce exactly.

## Contents

```
Task 3/
├── README.md
├── docs/
│   ├── METHOD.md                    six required skills, mapped and justified
│   ├── DATA_SOURCE.md               source, field mapping, verification record
│   └── PRESENTATION_SCRIPT.md       three-minute segment, five slides
├── src/
│   ├── data.py                      section parser and integrity checks
│   └── analysis.py                  sampling, descriptives, CI, t-test, figure
├── data/
│   ├── published_world_cup.csv      raw export, unedited
│   ├── task3_population.csv         144 team-match observations
│   ├── task3_sample.csv             the 100 sampled observations
│   ├── task3_team_level.csv         48 teams with the qualification flag
│   ├── task3_integrity_checks.csv   11 checks and their outcomes
│   └── task3_results.json           result, data hash, library versions
└── figures/
    └── task3.png
```

## Objective 1 coverage

Analytic question formulation, data wrangling, data preparation and sampling,
descriptive statistics, confidence interval, and a two-sample *t*-test.
[The method document](docs/METHOD.md) maps each one to its evidence.

The script also reports a four-task Bonferroni-adjusted *p*-value of 0.434 for
team integration. It does not recompute the other members' tests.

## Limitations

[The method document](docs/METHOD.md) carries the full list. The three that
matter most:

- The two observations from one fixture are not independent.
- Each team contributes three observations, so a team's traits enter the sample
  three times.
- A card count reflects the referee, the opponent, and the state of the game,
  not one team's discipline alone.

The result is an association. No causal claim is made in either direction.
