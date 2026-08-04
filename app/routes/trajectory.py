import os
import json

from fastapi import APIRouter, HTTPException

from app.config import TRAJECTORY_PATH

router = APIRouter()


@router.get("/trajectory/{query_id}")
def get_trajectory(query_id: str):
    path = os.path.join(TRAJECTORY_PATH, f"{query_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Trajectory not found")
    with open(path) as f:
        return json.load(f)
