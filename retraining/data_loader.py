"""Utilities for loading feedback into training-ready tensors."""

from __future__ import annotations

import json
import logging
from typing import Iterable

import numpy as np
import torch

from feedback_store.feedback_db import fetch_feedback


LOGGER = logging.getLogger(__name__)

ACTIONS = ["No Action", "Monitor", "Alert", "Block"]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}


def _feature_vector_from_state(state_json: str, action_taken: str) -> np.ndarray:
    state = json.loads(state_json)
    suggested = ACTION_TO_INDEX.get(str(action_taken), 1)
    return np.asarray(
        [
            float(state.get("risk_score", 0.0)),
            float(state.get("uncertainty", 0.0)),
            float(suggested) / max(len(ACTIONS) - 1, 1),
        ],
        dtype=np.float32,
    )


def load_feedback_dataset(db_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load feedback records from SQLite and convert them to tensors."""

    records = fetch_feedback(db_path)
    if not records:
        LOGGER.warning("No feedback records found in %s", db_path)
        return torch.empty((0, 3), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    features = []
    labels = []
    for record in records:
        features.append(_feature_vector_from_state(record["state"], record["action_taken"]))
        labels.append(ACTION_TO_INDEX.get(str(record["human_action"]), ACTION_TO_INDEX["Monitor"]))

    x_tensor = torch.tensor(np.stack(features), dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    LOGGER.info("Loaded %d feedback samples for supervised retraining", len(labels))
    return x_tensor, y_tensor
