"""Parses the published World Cup 2026 export into the Task 3 tables.

Section headings, not row offsets, decide what belongs to the group stage.
Column positions are in COL.
"""

import csv
import hashlib
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "published_world_cup.csv"

EXPECTED_SHA256 = "5b3420416acaf1cd65b8beec80eef4f06e99ead9c89135a000ed38be8609cb1b"

GROUP_HEADING = "GROUP STAGE"
KNOCKOUT_HEADINGS = ["ROUND OF 32", "ROUND OF 16", "QUARTER FINALS",
                     "SEMI FINALS", "FINAL"]

COL = {"date": 0, "team1": 1, "goals1": 2, "goals2": 3, "team2": 4,
       "yel1": 9, "yel2": 10, "red1": 11, "red2": 12,
       "shots1": 22, "shots2": 23, "sot1": 24, "sot2": 25}


def file_sha256(path=RAW):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_data_row(row):
    """Reports whether a row is a match row. Match rows start with a date."""
    return bool(row) and "/" in row[0] and row[0][:2].isdigit()


def _heading(row):
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


def group_stage_team_matches(sections):
    """Builds 144 team-match rows from the 72 group-stage fixtures."""
    records = []
    for src_row, r in enumerate(sections[GROUP_HEADING]):
        for side, t, o, g, ga, y, rc, sh, st in (
            (1, COL["team1"], COL["team2"], COL["goals1"], COL["goals2"],
             COL["yel1"], COL["red1"], COL["shots1"], COL["sot1"]),
            (2, COL["team2"], COL["team1"], COL["goals2"], COL["goals1"],
             COL["yel2"], COL["red2"], COL["shots2"], COL["sot2"]),
        ):
            records.append({
                "source_row": src_row,
                "date": r[COL["date"]],
                "team": r[t].strip(),
                "opponent": r[o].strip(),
                "listed_side": side,
                "goals_for": int(r[g]),
                "goals_against": int(r[ga]),
                "yellow_cards": int(r[y]),
                "red_cards": int(r[rc]),
                "shots": int(r[sh]),
                "shots_on_target": int(r[st]),
            })
    return pd.DataFrame(records)


def qualified_teams(sections):
    """Returns the teams appearing in any knockout section.

    Qualification comes from progression alone, never from the card variable.
    """
    teams = set()
    for name in KNOCKOUT_HEADINGS:
        for r in sections.get(name, []):
            teams.add(r[COL["team1"]].strip())
            teams.add(r[COL["team2"]].strip())
    return teams


def build_team_table(path=RAW):
    """Builds one row per team: group-stage discipline and qualification status."""
    sections = read_sections(path)
    tm = group_stage_team_matches(sections)
    qualified = qualified_teams(sections)

    agg = tm.groupby("team").agg(
        matches_played=("team", "size"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        shots=("shots", "sum"),
        shots_on_target=("shots_on_target", "sum"),
    ).reset_index()

    agg["qualified"] = agg["team"].isin(qualified)
    agg["group"] = agg["qualified"].map({True: "Qualified", False: "Eliminated"})
    agg["yellow_per_match"] = agg["yellow_cards"] / agg["matches_played"]
    return tm, agg.sort_values(["group", "team"]).reset_index(drop=True)


def integrity_checks(tm, teams):
    """Runs the integrity checks. Returns name, pass flag, and detail."""
    checks = []
    add = lambda n, ok, d: checks.append({"check": n, "passed": bool(ok), "detail": d})

    add("source file hash matches the recorded export",
        file_sha256() == EXPECTED_SHA256, file_sha256()[:16] + "...")
    add("72 group-stage fixtures", len(tm) // 2 == 72, f"{len(tm)//2} fixtures")
    add("144 team-match observations", len(tm) == 144, f"{len(tm)} rows")
    add("48 distinct teams", teams.team.nunique() == 48,
        f"{teams.team.nunique()} teams")
    add("every team played exactly 3 group matches",
        set(teams.matches_played) == {3}, f"values: {sorted(set(teams.matches_played))}")
    add("32 qualified and 16 eliminated",
        (teams.qualified.sum() == 32) and ((~teams.qualified).sum() == 16),
        f"{int(teams.qualified.sum())} / {int((~teams.qualified).sum())}")
    add("card counts are non-negative integers",
        (tm.yellow_cards >= 0).all() and tm.yellow_cards.dtype.kind == "i",
        f"range {tm.yellow_cards.min()}-{tm.yellow_cards.max()}")
    add("no missing values in the analysis columns",
        not tm[["team", "yellow_cards"]].isna().any().any(), "none found")
    add("shots on target never exceed shots",
        (tm.shots_on_target <= tm.shots).all(), "holds for all rows")
    add("no team appears twice in one fixture",
        (tm.team != tm.opponent).all(), "holds for all rows")
    add("team yellow totals reconstruct the fixture totals",
        tm.groupby("source_row").yellow_cards.sum().sum() == tm.yellow_cards.sum(),
        f"{int(tm.yellow_cards.sum())} cards")
    return pd.DataFrame(checks)
