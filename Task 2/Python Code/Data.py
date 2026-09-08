"""Read only the group-stage section of the original published CSV."""
import csv
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path

RAW_SHA256 = '5b3420416acaf1cd65b8beec80eef4f06e99ead9c89135a000ed38be8609cb1b'
PREVIOUS_SHA256 = 'e9c5ae8f09bd22f42379a4442ba883e73c664a3bb3fe0937b917c955b46ac2fc'
COLUMNS = ['date','team1','goals1','goals2','team2','goals_1h1','goals_1h2',
           'goals_2h1','goals_2h2','yellow1','yellow2','red1','red2','corners_1h1',
           'corners_1h2','corners_2h1','corners_2h2','corners1','corners2',
           'corners_total','xg1','xg2','shots1','shots2','sot1','sot2','fouls1','fouls2']
SHOT_COLUMNS = ['shots1','shots2','sot1','sot2']

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_source(path):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == RAW_SHA256,
            'Published source changed. Review the new data before updating the recorded hash.')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    require(rows[0][0].strip() == 'GROUP STAGE', 'Missing group-stage section.')
    require([rows[1][i].replace('\r','').replace('\n',' ') for i in [22,24]] == ['SHOTS','SHOTS ON TARGET'],
            'Shot headers changed; check column mapping.')
    matches = []
    for source_row, row in enumerate(rows[3:], start=4):
        if not row or not row[0].strip():
            if matches:
                break
            continue
        try:
            datetime.strptime(row[0], '%d/%m/%Y')
        except ValueError:
            break  # A later section is deliberately outside this task.
        require(len(row) == 28, f'Wrong column count at source row {source_row}.')
        record = dict(zip(COLUMNS, row))
        selected = {'source_row':source_row, **{k:record[k].strip() for k in ['date','team1','team2']}}
        for key in SHOT_COLUMNS:
            selected[key] = int(record[key])  # Missing, fractional and non-numeric counts fail.
        matches.append(selected)
    validate_matches(matches)
    return matches

def validate_matches(matches):
    require(len(matches)==72, f'Expected 72 group-stage matches; got {len(matches)}.')
    require(len({r['source_row'] for r in matches})==72, 'Duplicate source row.')
    teams = Counter()
    pairs = set()
    for r in matches:
        date = datetime.strptime(r['date'], '%d/%m/%Y')
        require(datetime(2026,6,11) <= date <= datetime(2026,6,28), 'Unexpected group-stage date.')
        require(r['team1'] and r['team2'] and r['team1']!=r['team2'], 'Invalid team names.')
        pair = tuple(sorted([r['team1'],r['team2']]))
        require(pair not in pairs, 'Duplicate fixture, including reversed team order.')
        pairs.add(pair)
        teams.update(pair)
        for side in ['1','2']:
            shots, sot = r['shots'+side], r['sot'+side]
            require(isinstance(shots,int) and isinstance(sot,int) and 0<=sot<=shots,
                    f'Invalid shot counts at source row {r["source_row"]}.')
    require(len(teams)==48 and set(teams.values())=={3}, 'Expected 48 teams, three matches each.')
    return group_mapping(matches)

def group_mapping(matches):
    graph = {}
    for r in matches:
        graph.setdefault(r['team1'],set()).add(r['team2'])
        graph.setdefault(r['team2'],set()).add(r['team1'])
    mapping = {}
    while len(mapping)<len(graph):
        pending = [min(set(graph)-set(mapping))]
        found = set()
        while pending:
            team = pending.pop()
            if team not in found:
                found.add(team)
                pending.extend(graph[team]-found)
        require(len(found)==4 and all(len(graph[t])==3 for t in found), 'Fixtures do not form complete four-team groups.')
        name = f'G{len(set(mapping.values()))+1:02d}'
        mapping.update({t:name for t in found})
    require(len(set(mapping.values()))==12, 'Expected twelve groups.')
    return mapping

def compare_previous(matches, path):
    require(hashlib.sha256(path.read_bytes()).hexdigest()==PREVIOUS_SHA256, 'Previous capture changed.')
    with path.open(encoding='utf-8',newline='') as stream:
        previous = list(csv.reader(stream, delimiter='\t'))
    require(len(previous)==len(matches), 'Previous capture count differs.')
    old = {int(row[0]):dict(zip(['source_row']+COLUMNS,row)) for row in previous}
    for record in matches:
        original = old[record['source_row']]
        for key in ['date','team1','team2']:
            require(record[key]==original[key], f'Previous capture disagrees on {key}.')
        for key in SHOT_COLUMNS:
            require(record[key]==int(original[key]), f'Previous capture disagrees on {key}.')
    return {'matches_compared':72,'shot_values_compared':288,'discrepancies':0,
            'scope':'Direct published CSV versus previous capture; not an independent provider audit.'}
