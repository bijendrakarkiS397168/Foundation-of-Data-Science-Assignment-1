# Task 4: finishing against expected goals

FIFA World Cup 2026, Objective 1. Kalyan Acharya.
Focal point: **finishing relative to chance quality**. Task 1 covers corners,
Task 2 shot volume and accuracy, Task 3 discipline.

> During the FIFA World Cup 2026 group stage, did teams score more goals than
> their chances expected?

The variable is the **finishing differential**: goals scored minus xG, for one
team in one match. A positive value means the team beat its chances.

## Design

| Element | Value |
| --- | --- |
| Population | 144 team-match observations: 72 group-stage fixtures × 2 teams |
| Sampling | Simple random, without replacement |
| Sample | n = 100, seed 7 |
| Test | One-sample *t*-test against zero, α = 0.05, two-sided |

Zero is the point at which a team converts exactly as the xG model predicts, so
the benchmark comes from the data rather than from an outside figure.

## Results

**Confidence interval.** Population mean finishing differential, estimated from
the sample: **+0.208 goals, 95% CI [+0.003, +0.414]**, df = 99. The true
population mean is +0.206 and falls inside the interval.

**One-sample t-test.**

| Measure | Value |
| --- | --- |
| n | 100 |
| Mean goals minus xG | +0.208 |
| SD | 1.037 |
| Median | +0.110 |
| Beat their xG | 53 of 100 |

*t*(99) = 2.009, *p* = 0.047, Cohen's *d* = 0.201.
**The test rejects H₀ at α = 0.05.**

Teams scored about a fifth of a goal per match more than their chances expected.

## Robustness

| Check | Result | Same conclusion |
| --- | --- | --- |
| Wilcoxon signed-rank | p = 0.082 | **no** |
| Bootstrap 95% CI, 10,000 resamples | [+0.010, +0.407] | yes |
| Full population, no sampling | +0.206, p = 0.020 | yes |
| Sampling design repeated 2,000 times | 47.7% of draws reject H₀ | n/a |
| Bonferroni across four tasks | p = 0.189 | **no** |

## The result sits on the decision boundary

The sampled test clears 0.05 at 0.047. The Wilcoxon check lands at 0.082 on the
other side. Only about half the samples this design could have drawn would
reject at all. The seed was fixed in advance and never changed to obtain a
preferred *p*-value.

So the two halves of the result behave differently:

- **The estimate is stable.** Sample +0.208, population +0.206, bootstrap
  interval clear of zero. Every route puts the effect near a fifth of a goal.
- **The verdict is not.** A yes-or-no decision at 0.05 flips on which 100 of the
  144 observations get drawn.

The full population uses all the evidence and rejects at p = 0.020, interval
[+0.033, +0.379]. That is the better guide to the substantive question. The
sampled test is the required exercise, and reporting its fragility is more
useful than presenting 0.047 as a clean win.

The power figures explain why. The smallest effect this design detects at 80%
power is 0.29 goals per match. The observed effect is 0.21, below that
threshold, so the test was always going to land near the line.

## Run the analysis

1.  Install the dependencies:

    ```bash
    pip install pandas numpy scipy matplotlib
    ```

2.  Run the script from the task root:

    ```bash
    python src/analysis.py
    ```

The script checks the source file's SHA-256 and runs twelve integrity checks
before computing any statistic. If one fails, it stops instead of returning a
number. All seeds are fixed, so the sample, the bootstrap, and the figure
reproduce exactly.

## Contents

```
Task 4/
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
│   ├── task4_population.csv         144 team-match observations
│   ├── task4_sample.csv             the 100 sampled observations
│   ├── task4_integrity_checks.csv   12 checks and their outcomes
│   └── task4_results.json           result, data hash, library versions
└── figures/
    └── task4.png
```

## Objective 1 coverage

Analytic question formulation, data wrangling, data preparation and sampling,
descriptive statistics, confidence interval, and a one-sample *t*-test.
[The method document](docs/METHOD.md) maps each one to its evidence.

The Bonferroni-adjusted *p*-value across four team tasks is 0.189, so this
result does not survive that correction. That belongs in the group conclusion.

## Limitations

[The method document](docs/METHOD.md) carries the full list. The three that
matter most:

- xG is a model output, not a measurement. The differential inherits every
  assumption in the provider's chance-quality model.
- The two observations from one fixture are not independent.
- Beating xG over 144 observations could reflect finishing quality, model
  calibration for this tournament, or ordinary variation. This task cannot
  separate them.

The result is an association. No causal claim is made.
