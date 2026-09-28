"""
Rail-Block AI backend.

Run locally:
    cd backend
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000/docs to see and test every endpoint
in the browser (FastAPI auto-generates this) before wiring up the frontend.
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import os

from scheduler import resolve_conflicts, weekly_block_plan, plan_options

app = FastAPI(title="Rail-Block AI API")

# CORS: allows the frontend (running from a different origin, e.g. a local
# file:// page or a different deployed domain) to call this API from the
# browser. Tighten allow_origins to your real frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "xgb_risk_model.json")
SCORED_TASKS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "scored_tasks.csv")

_model = xgb.XGBClassifier()
_model.load_model(MODEL_PATH)

FEATURES = [
    "severity", "days_overdue", "asset_age_years", "past_failures_12m",
    "traffic_density_trains_per_day", "single_line_section", "safety_critical_asset",
    "weather_score", "days_since_last_maintenance", "block_duration_min",
    "department_enc", "asset_type_enc",
]


# ---------------- request/response schemas ----------------

class TaskFeatures(BaseModel):
    severity: int
    days_overdue: int
    asset_age_years: float
    past_failures_12m: int
    traffic_density_trains_per_day: int
    single_line_section: int
    safety_critical_asset: int
    weather_score: int          # 0=Low, 1=Medium, 2=High
    days_since_last_maintenance: int
    block_duration_min: int
    department_enc: int         # 0=ENG, 1=SNT, 2=TRD (see /meta for exact mapping)
    asset_type_enc: int


class Train(BaseModel):
    id: str
    name: str
    start: float   # hour, e.g. 11.5 = 11:30
    dur: float     # hours


class Block(BaseModel):
    start: float
    end: float


class ConflictRequest(BaseModel):
    trains: List[Train]
    block: Block
    priority_trains: Optional[List[Train]] = []


class ScheduleTask(BaseModel):
    task_id: str
    section: str
    department: str
    duration_min: int
    priority_score: float
    deadline_day: int = 6


class ScheduleRequest(BaseModel):
    tasks: List[ScheduleTask]


# ---------------- endpoints ----------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    return {
        "department_encoding": {"ENG": 0, "SNT": 1, "TRD": 2},
        "weather_encoding": {"Low": 0, "Medium": 1, "High": 2},
        "features_expected": FEATURES,
    }


@app.post("/priority-score")
def priority_score(task: TaskFeatures):
    """Score a single task's risk/priority using the trained XGBoost model."""
    row = pd.DataFrame([task.dict()])[FEATURES]
    score = float(_model.predict_proba(row)[:, 1][0])
    return {"priority_score": round(score, 4)}


@app.get("/tasks")
def get_scored_tasks(top: int = 20):
    """Top N tasks by the model's already-computed priority score."""
    if not os.path.exists(SCORED_TASKS_PATH):
        raise HTTPException(404, "scored_tasks.csv not found — run model/train_model.py first")
    df = pd.read_csv(SCORED_TASKS_PATH)
    df = df.sort_values("ai_priority_score", ascending=False).head(top)
    return df.to_dict(orient="records")


@app.post("/conflict-check")
def conflict_check(req: ConflictRequest):
    """
    Given a proposed maintenance block (and optionally a max-priority train),
    find which scheduled trains conflict and return two resolution options.
    """
    trains = [t.dict() for t in req.trains]
    priority_trains = [t.dict() for t in (req.priority_trains or [])]
    block = req.block.dict()
    return resolve_conflicts(trains, block, priority_trains)


@app.post("/schedule")
def schedule(req: ScheduleRequest):
    """Weekly block plan: fit tasks (already scored by the model) into section/day windows."""
    tasks = [t.dict() for t in req.tasks]
    return weekly_block_plan(tasks)


# ======================================================================
# Endpoints used by the web dashboard (Planner / Dashboard / Alerts / Schedules)
# ======================================================================

class PlanRequest(BaseModel):
    shift_start: float          # hour, e.g. 6.0  (may exceed 24 for night shifts)
    shift_end: float
    block_hours: float          # requested block duration in hours
    trains: List[str]           # names of trains needing blocks
    emergency: List[str] = []   # max-priority trains that must never be disturbed


@app.post("/plan")
def plan(req: PlanRequest):
    """Planner page: timetable + 3 block options + conflicts + metrics."""
    return plan_options(req.shift_start, req.shift_end, req.block_hours, req.trains, req.emergency)


DEPT_LABEL = {"ENG": "ENG", "SNT": "S&T", "TRD": "TRD"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _backlog(n: int = 60) -> pd.DataFrame:
    """Current backlog = the n tasks the XGBoost model ranks as highest risk.
    Section is assigned deterministically (SEC-01..04) because the synthetic
    data has no section column; replace with the real section when TMS/SMMS/TDMS are wired in."""
    if not os.path.exists(SCORED_TASKS_PATH):
        raise HTTPException(404, "scored_tasks.csv not found - run model/train_model.py first")
    df = pd.read_csv(SCORED_TASKS_PATH).sort_values("ai_priority_score", ascending=False).head(n).copy()
    df["section"] = df["task_id"].map(
        lambda t: f"SEC-{int(''.join(c for c in str(t) if c.isdigit())) % 4 + 1:02d}")
    return df


def _sched_tasks(df: pd.DataFrame):
    return [{
        "task_id": r.task_id, "section": r.section, "department": r.department,
        "duration_min": int(r.block_duration_min), "priority_score": float(r.ai_priority_score),
        # the more overdue, the earlier it should be done
        "deadline_day": max(0, 6 - min(6, int(r.days_overdue) // 3)),
    } for r in df.itertuples()]


def _severity(score: float) -> str:
    return "critical" if score >= 0.70 else "high" if score >= 0.55 else "routine"


@app.get("/dashboard")
def dashboard():
    df = _backlog(60)
    plan = weekly_block_plan(_sched_tasks(df.head(20)))
    used = sum(t["duration_min"] for t in _sched_tasks(df.head(20))
               if plan["assignment"][t["task_id"]]["day"] is not None)
    capacity = 4 * 7 * 120
    k = plan["kpis"]
    critical = int((df["ai_priority_score"] >= 0.70).sum())
    queue = [{
        "dept": DEPT_LABEL.get(r.department, r.department), "task": r.defect_type,
        "section": r.section, "overdue_days": int(r.days_overdue),
        "score": round(float(r.ai_priority_score) * 100),
        "severity": _severity(float(r.ai_priority_score)),
    } for r in df.head(8).itertuples()]
    return {
        "kpis": {
            "open_tasks": len(df),
            "blocks_this_week": k["distinct_blocks"],
            "shared_pct": round(100 * k["bundled_blocks"] / max(1, k["distinct_blocks"])),
            "critical_overdue": critical,
            "corridor_use": round(100 * used / capacity),
        },
        "queue": queue,
    }


@app.get("/week-plan")
def week_plan():
    df = _backlog(60).head(20)
    plan = weekly_block_plan(_sched_tasks(df))
    by_id = {r.task_id: r for r in df.itertuples()}
    cells = {}
    for b in plan["blocks"]:
        key = f"{b['section']}_{DAYS[b['day']]}"
        for tid in b["tasks"]:
            r = by_id[tid]
            cells.setdefault(key, []).append(
                {"dept": DEPT_LABEL.get(r.department, r.department), "title": r.defect_type})
    return {"cells": cells, "kpis": plan["kpis"]}
