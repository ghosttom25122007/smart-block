"""
Core scheduling / conflict-resolution logic, kept separate from the API layer
so it can be unit-tested and reused.

This mirrors the logic already prototyped in frontend/rail-block-ai.html
(train-vs-block conflict detection + two resolution strategies), so the
frontend can eventually just call the API instead of computing this in JS.
"""

from typing import List, Dict, Any


def overlaps(a_start: float, a_dur: float, b_start: float, b_dur: float) -> bool:
    return a_start < b_start + b_dur and a_start + a_dur > b_start


def find_conflicts(trains: List[Dict[str, Any]], block: Dict[str, float],
                    priority_trains: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """A train conflicts if it overlaps the proposed block OR any max-priority train."""
    priority_trains = priority_trains or []
    block_dur = block["end"] - block["start"]
    conflicts = []
    for t in trains:
        hits_block = overlaps(t["start"], t["dur"], block["start"], block_dur)
        hits_priority = any(
            overlaps(t["start"], t["dur"], p["start"], p["dur"]) for p in priority_trains
        )
        if hits_block or hits_priority:
            conflicts.append(t)
    return conflicts


def _clear_by(train: Dict[str, Any], block: Dict[str, float],
              priority_trains: List[Dict[str, Any]]) -> float:
    block_dur = block["end"] - block["start"]
    end = block["end"] if overlaps(train["start"], train["dur"], block["start"], block_dur) else train["start"]
    for p in priority_trains:
        if overlaps(train["start"], train["dur"], p["start"], p["dur"]):
            end = max(end, p["start"] + p["dur"])
    return end


def resolve_conflicts(trains: List[Dict[str, Any]], block: Dict[str, float],
                       priority_trains: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns two resolution strategies:
      option_1 (Minimize Delay): push each conflicting train only just past
        whichever reservation (block or priority train) it clashes with.
      option_2 (Maximize Block): push conflicting trains further out with a
        buffer, guaranteeing the block/priority reservation is fully protected.
    """
    priority_trains = priority_trains or []
    conflicts = find_conflicts(trains, block, priority_trains)

    option_1, option_2 = [], []
    total_delay_1 = total_delay_2 = 0
    for t in conflicts:
        clear_at = _clear_by(t, block, priority_trains)

        new_start_1 = max(t["start"], clear_at)
        delay_1 = round((new_start_1 - t["start"]) * 60)
        option_1.append({**t, "new_start": new_start_1, "delay_min": delay_1})
        total_delay_1 += delay_1

        new_start_2 = clear_at + 0.5
        delay_2 = round((new_start_2 - t["start"]) * 60)
        option_2.append({**t, "new_start": new_start_2, "delay_min": delay_2})
        total_delay_2 += delay_2

    baseline = max(total_delay_1, total_delay_2, len(conflicts) * (block["end"] - block["start"]) * 60 * 1.4, 1)

    def gain(delay):
        return min(25, round(((baseline - delay) / baseline) * 30))

    return {
        "conflicts_count": len(conflicts),
        "option_1": {
            "label": "Minimize Delay", "trains": option_1,
            "total_delay_min": total_delay_1, "network_availability_gain_pct": gain(total_delay_1),
        },
        "option_2": {
            "label": "Maximize Block", "trains": option_2,
            "total_delay_min": total_delay_2, "network_availability_gain_pct": gain(total_delay_2),
        },
        "baseline_delay_min": round(baseline),
    }


def weekly_block_plan(tasks: List[Dict[str, Any]], base_capacity_min: int = 120,
                       days: int = 7) -> Dict[str, Any]:
    """
    Greedy weekly scheduler: sorts tasks by priority_score (already computed
    by the model), fits them into per-section daily block windows, bundling
    tasks that land on the same section+day across departments.
    """
    sections = sorted(set(t["section"] for t in tasks))
    remaining = {(s, d): base_capacity_min for s in sections for d in range(days)}
    sorted_tasks = sorted(tasks, key=lambda t: t.get("priority_score", 0), reverse=True)

    assignment = {}
    for t in sorted_tasks:
        placed = False
        deadline_day = t.get("deadline_day", days - 1)
        for d in range(min(days, deadline_day + 2)):
            key = (t["section"], d)
            if remaining[key] >= t["duration_min"]:
                remaining[key] -= t["duration_min"]
                assignment[t["task_id"]] = {"day": d, "breach": d > deadline_day}
                placed = True
                break
        if not placed:
            for d in range(days):
                key = (t["section"], d)
                if remaining[key] >= t["duration_min"]:
                    remaining[key] -= t["duration_min"]
                    assignment[t["task_id"]] = {"day": d, "breach": True}
                    placed = True
                    break
        if not placed:
            assignment[t["task_id"]] = {"day": None, "breach": True}

    blocks = {}
    for t in sorted_tasks:
        a = assignment[t["task_id"]]
        if a["day"] is None:
            continue
        key = f'{t["section"]}|day{a["day"]}'
        blocks.setdefault(key, {"section": t["section"], "day": a["day"], "tasks": [], "departments": set()})
        blocks[key]["tasks"].append(t["task_id"])
        blocks[key]["departments"].add(t["department"])

    block_list = [
        {"section": b["section"], "day": b["day"], "tasks": b["tasks"],
         "departments": sorted(b["departments"]), "bundled": len(b["departments"]) > 1}
        for b in blocks.values()
    ]
    breaches = sum(1 for a in assignment.values() if a["breach"])

    return {
        "assignment": assignment,
        "blocks": block_list,
        "kpis": {
            "tasks_scheduled": sum(1 for a in assignment.values() if a["day"] is not None),
            "total_tasks": len(tasks),
            "distinct_blocks": len(block_list),
            "bundled_blocks": sum(1 for b in block_list if b["bundled"]),
            "deadline_breaches": breaches,
        },
    }


# ======================================================================
# Planner-page logic (used by the "Generate Optimized Block Schedule" screen)
# ======================================================================

def _hash(s: str) -> int:
    """Same 32-bit string hash the original frontend used, so results stay stable."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def synth_timetable(names, shift_start: float, shift_end: float):
    """
    Deterministic SYNTHETIC timetable for the selected trains within a shift.
    This is the stand-in for real COA / train-timetable data: replace this one
    function with a lookup into the real timetable and nothing else changes.
    """
    span = shift_end - shift_start
    trains = []
    for i, name in enumerate(names):
        h = _hash(name + str(i))
        dur = 1 + (h % 3) * 0.5
        max_start = max(0.25, span - dur - 0.25)
        start = shift_start + ((h % 97) / 97) * max_start
        trains.append({"name": name, "start": start, "end": start + dur})
    return trains


def _windows(block):
    return block["windows"] if block.get("split") else [block]


def _slide(win, protected, shift_start, shift_end):
    """Move a block window forward (15-min steps) until it no longer overlaps any
    max-priority train. If no clean position exists in the shift, keep it as is."""
    dur = win["end"] - win["start"]
    s = win["start"]
    while s + dur <= shift_end + 1e-9:
        if not any(p["start"] < s + dur and s < p["end"] for p in protected):
            return {"start": s, "end": s + dur}
        s += 0.25
    return win


def block_window(option, shift_start, shift_end, dur, protected):
    if option == 1:      # Minimize Delay: single window early in the shift
        start = shift_start + 0.5
        blk = {"start": start, "end": start + dur}
    elif option == 2:    # Maximize Block: extended window for single-pass coverage
        start = shift_start + 1
        blk = {"start": start, "end": min(shift_end, start + dur * 1.75)}
    else:                # Split Window: two shorter windows
        half = dur / 2
        mid = (shift_start + shift_end) / 2
        s1 = shift_start + 0.75
        s2 = min(shift_end - half - 0.25, mid + 0.75)
        blk = {"split": True, "windows": [{"start": s1, "end": s1 + half},
                                          {"start": s2, "end": s2 + half}]}
    # Max-priority trains are never disturbed: slide the block clear of them.
    if blk.get("split"):
        blk["windows"] = [_slide(w, protected, shift_start, shift_end) for w in blk["windows"]]
    else:
        blk = _slide(blk, protected, shift_start, shift_end)
    return blk


def _delay_minutes(train, block):
    """Minutes a train must wait for the block to clear (0 if it doesn't clash)."""
    hits = [w for w in _windows(block) if train["start"] < w["end"] and w["start"] < train["end"]]
    if not hits:
        return 0
    return round((max(w["end"] for w in hits) - train["start"]) * 60)


def _baseline_delay(trains, shift_start, shift_end, dur):
    """Average total train delay if the block were dropped at ANY start time in the
    shift (15-min steps) - i.e. what an uninformed placement would cost."""
    movable = [t for t in trains if not t["protected"]]
    totals, s = [], shift_start
    while s + dur <= shift_end + 1e-9:
        blk = {"start": s, "end": s + dur}
        totals.append(sum(_delay_minutes(t, blk) for t in movable))
        s += 0.25
    return sum(totals) / len(totals) if totals else 0


def plan_options(shift_start, shift_end, dur_hours, names, emergency_names):
    trains = synth_timetable(names, shift_start, shift_end)
    emg = {e.lower() for e in emergency_names}
    for t in trains:
        t["protected"] = t["name"].lower() in emg
    protected = [t for t in trains if t["protected"]]
    baseline = _baseline_delay(trains, shift_start, shift_end, dur_hours)

    options = {}
    for o in (1, 2, 3):
        block = block_window(o, shift_start, shift_end, dur_hours, protected)
        tr = []
        for t in trains:
            d = 0 if t["protected"] else _delay_minutes(t, block)
            tr.append({**t, "conflict": d > 0, "delay_min": d})

        total = len(tr) or 1
        conflicts = sum(1 for t in tr if t["conflict"])
        delay = sum(t["delay_min"] for t in tr)
        gain = round(max(0, (baseline - delay) / baseline * 100)) if baseline > 0 else 0
        options[str(o)] = {
            "block": block,
            "trains": tr,
            "metrics": {
                "conflicts": conflicts,
                "total": len(tr),
                "protectedCount": len(protected),
                # COMPUTED: % of train-delay avoided vs an uninformed block placement
                "availability": gain,
                # COMPUTED: real minutes trains wait for the block to clear
                "delay": delay,
                # ILLUSTRATIVE until real corridor-occupancy data (COA) is connected
                "utilization": 96 if o == 2 else 84 if o == 3 else 90,
                # HEURISTIC confidence in executing this option as planned
                "probability": max(28, min(96, 95 - conflicts * 9 - (10 if o == 2 else 16 if o == 3 else 0))),
            },
        }
    return {"options": options, "baseline_delay_min": round(baseline)}
