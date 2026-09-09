"""Parses the published World Cup 2026 export into the Task 4 tables.

Section headings, not row offsets, decide what belongs to the group stage.
Column positions are in COL.
"""

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "published_world_cup.csv"
EXPECTED_SHA256 = "5b3420416acaf1cd65b8beec80eef4f06e99ead9c89135a000ed38be8609cb1b"

GROUP_HEADING = "GROUP STAGE"
KNOCKOUT_HEADINGS = ["ROUND OF 32", "ROUND OF 16", "QUARTER FINALS",
                     "SEMI FINALS", "FINAL"]

COL = {"date": 0, "team1": 1, "goals1": 2, "goals2": 3, "team2": 4,
       "xg1": 20, "xg2": 21, "shots1": 22, "shots2": 23,
       "sot1": 24, "sot2": 25}


def file_sha256(path=RAW):
    """Returns the SHA-256 of the raw export."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_data_row(row):
    """Reports whether a row is a match row. Match rows start with a date."""
    return bool(row) and "/" in row[0] and row[0][:2].isdigit()


def _heading(row):
    """Returns the heading text of a row, or None if the row is not a heading."""
    joined = ",".join(row).strip(",").strip()
    return joined if joined and not _is_data_row(row) else None


def read_sections(path=RAW):
    """Splits the export into a dict of section name to match rows."""
    rows = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
    sections, current = {}, None
    for row in rows:
        head = _heading(row)
        if head and head.upper() in [GROUP_HEADING] + KNOCKOUT_HEADINGS:
            current = head.upper()
            sections[current] = []
        elif _is_data_row(row) and current:
            sections[current].append(row)
    return sections


def build_team_match(path=RAW):
    """Builds 144 team-match rows carrying goals, xG, and their difference."""
    sections = read_sections(path)
    records = []
    for src_row, r in enumerate(sections[GROUP_HEADING]):
        for side, t, o, g, ga, x, xo, sh, st in (
            (1, "team1", "team2", "goals1", "goals2", "xg1", "xg2", "shots1", "sot1"),
            (2, "team2", "team1", "goals2", "goals1", "xg2", "xg1", "shots2", "sot2"),
        ):
            goals = int(r[COL[g]])
            xg = float(r[COL[x]])
            records.append({
                "source_row": src_row,
                "date": r[COL["date"]],
                "team": r[COL[t]].strip(),
                "opponent": r[COL[o]].strip(),
                "listed_side": side,
                "goals": goals,
                "goals_against": int(r[COL[ga]]),
                "xg": xg,
                "xg_against": float(r[COL[xo]]),
                "finishing": round(goals - xg, 4),
                "shots": int(r[COL[sh]]),
                "shots_on_target": int(r[COL[st]]),
            })
    return pd.DataFrame(records)


def integrity_checks(tm):
    """Runs the integrity checks. Returns name, pass flag, and detail."""
    checks = []
    add = lambda n, ok, d: checks.append({"check": n, "passed": bool(ok), "detail": d})

    add("source file hash matches the recorded export",
        file_sha256() == EXPECTED_SHA256, file_sha256()[:16] + "...")
    add("72 group-stage fixtures", len(tm) // 2 == 72, f"{len(tm) // 2} fixtures")
    add("144 team-match observations", len(tm) == 144, f"{len(tm)} rows")
    add("48 distinct teams", tm.team.nunique() == 48, f"{tm.team.nunique()} teams")
    add("every team played exactly 3 group matches",
        set(tm.team.value_counts()) == {3},
        f"values: {sorted(set(tm.team.value_counts()))}")
    add("goals are non-negative integers",
        (tm.goals >= 0).all() and tm.goals.dtype.kind == "i",
        f"range {tm.goals.min()}-{tm.goals.max()}")
    add("xG values are positive and finite",
        tm.xg.gt(0).all() and tm.xg.notna().all(),
        f"range {tm.xg.min():.2f}-{tm.xg.max():.2f}")
    add("no missing values in the analysis columns",
        not tm[["goals", "xg", "finishing"]].isna().any().any(), "none found")
    add("finishing equals goals minus xG",
        ((tm.goals - tm.xg - tm.finishing).abs() < 1e-9).all(), "holds for all rows")
    add("shots on target never exceed shots",
        (tm.shots_on_target <= tm.shots).all(), "holds for all rows")
    add("no team appears twice in one fixture",
        (tm.team != tm.opponent).all(), "holds for all rows")
    add("every fixture contributes exactly two rows",
        tm.groupby("source_row").size().eq(2).all(),
        f"{tm.source_row.nunique()} fixtures")
    return pd.DataFrame(checks)
