from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import json
import numpy as np


def _to_str(x: Any) -> str:
    return str(x)


@dataclass
class TimeDataSimulatorConfig:
    embedding_path: str
    pairwise_path: str
    indices_path: str
    regression_json_path: str
    total_data_points: Optional[int] = None  # default: len(indices)
    batch_size: int = 256

class TimeDataSimulator:
    def __init__(self, config: TimeDataSimulatorConfig):
        self.config = config

        # Load feature artifacts
        self.embeddings = np.load(config.embedding_path)
        self.pairwise_matrix = np.load(config.pairwise_path)
        with open(config.indices_path, "r") as f:
            self.indices: List[Any] = json.load(f)

        # Build mappings; normalize keys to str for robust matching
        self.idx_to_qid: Dict[int, Any] = {i: qid for i, qid in enumerate(self.indices)}
        self.qid_to_idx: Dict[str, int] = {str(qid): i for i, qid in enumerate(self.indices)}

        # Load regression data and normalize keys to str
        with open(config.regression_json_path, "r") as f:
            raw_reg = json.load(f)

        num_keep = (len(raw_reg) // config.batch_size) * config.batch_size
        keep_keys = list(raw_reg.keys())[:num_keep]
        self.regression_data: Dict[str, Any] = {str(k): v for k, v in raw_reg.items() if k in keep_keys}

        # Optionally set total data points from indices if not provided
        self.total_data_points = (
            config.total_data_points if config.total_data_points is not None else len(self.indices)
        )
        self.batch_size = config.batch_size
        # Basic shape checks
        n = len(self.indices)
        if self.embeddings.shape[0] != n:
            raise ValueError(
                f"Embeddings rows {self.embeddings.shape[0]} != indices length {n}"
            )
        if self.pairwise_matrix.shape[0] != n or self.pairwise_matrix.shape[1] != n:
            raise ValueError(
                f"Pairwise shape {self.pairwise_matrix.shape} incompatible with indices length {n}"
            )

    # ----- Step utilities -----
    def get_total_gradient_steps(self, target_key: str = "mean_acc_per_epoch") -> int:
        # Always use configured batch size
        total_observations = 0
        for _, value in self.regression_data.items():
            # value[target_key] is expected to be a list/sequence over time
            series = value.get(target_key, [])
            total_observations += len(series)
        return total_observations // self.batch_size

    def step_to_epoch_step(self, step: int) -> Tuple[int, int]:
        # Always use configured batch size
        epoch = (step * self.batch_size) // self.total_data_points
        epoch_step = (step * self.batch_size) % self.total_data_points
        return epoch, epoch_step

    def get_data_at_step(
        self,
        step: int,
        target_key: str = "mean_acc_per_epoch",
    ) -> Dict[str, Any]:
        # Always use configured batch size
        epoch, epoch_step = self.step_to_epoch_step(step)
        # only consider keys present in indices/qid mapping
        keys_filtered = [k for k in self.regression_data.keys() if k in self.qid_to_idx]
        # keys_filtered = sorted([k for k in self.regression_data.keys() if k in self.qid_to_idx])
        if not keys_filtered:
            return {}

        n = len(keys_filtered)
        bs = min(self.batch_size, n)
        # cycle over keys to always return a batch
        selected_keys = [keys_filtered[(epoch_step + i) % n] for i in range(bs)]
        # Return the full series for each selected key (mirrors notebook behavior)
        data_at_step = {k: self.regression_data[k][target_key][epoch] for k in selected_keys}
        return data_at_step

    def prepare_time_window_data(
        self,
        step: int,
        window_size: int = 2,
        target_key: str = "mean_acc_per_epoch",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Aggregate train over [step - window_size, step) and test at current step.

        Returns (train_dict, test_dict) where each is {qid: value}.
        """
        test_data = self.get_data_at_step(step, target_key)
        train_data: Dict[str, Any] = {}
        start = max(0, step - window_size)
        for s in range(start, step):
            d = self.get_data_at_step(s, target_key)
            train_data.update(d)
        return train_data, test_data

    # ----- Feature builders -----
    def _extract_embeddings_for_qids(self, qids: List[str]) -> np.ndarray:
        idxs = [self.qid_to_idx[q] for q in qids]
        return self.embeddings[idxs]

    def _extract_pairwise_for_qids(self, qids: List[str]) -> np.ndarray:
        idxs = [self.qid_to_idx[q] for q in qids]
        return self.pairwise_matrix[np.ix_(idxs, idxs)]

    def _filter_known_qids(self, qids: List[Any]) -> Tuple[List[str], List[Any]]:
        """Return (kept_qids_str, kept_qids_original) after filtering to those present in mapping."""
        kept_str: List[str] = []
        kept_orig: List[Any] = []
        for q in qids:
            q_str = _to_str(q)
            if q_str in self.qid_to_idx:
                kept_str.append(q_str)
                kept_orig.append(q)
        return kept_str, kept_orig

    def build_features(self, data: Dict[Any, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
        """Convert {qid: value} into X, P, y and return also the ordered qids used.

        - If value is a list/sequence, we keep it as-is in y (np.array of objects) or
          consider using a summary (e.g., last, mean). Here we will take the last
          value if it's a non-empty sequence; else np.nan.
        """
        qids_orig: List[Any] = list(data.keys())
        qids_str, qids_kept = self._filter_known_qids(qids_orig)
        if len(qids_str) == 0:
            # Empty set; return consistent empty arrays
            return (
                np.empty((0, self.embeddings.shape[1])),
                np.empty((0, 0)),
                np.empty((0,), dtype=float),
                [],
            )

        X = self._extract_embeddings_for_qids(qids_str)
        P = self._extract_pairwise_for_qids(qids_str)

        y_vals: List[float] = []
        for q in qids_kept:
            v = data[q]
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    y_vals.append(np.nan)
                else:
                    y_vals.append(float(v[-1]))  # last value in the series
            else:
                try:
                    y_vals.append(float(v))
                except Exception:
                    y_vals.append(np.nan)
        y = np.asarray(y_vals, dtype=float)
        return X, P, y, qids_kept

    # ----- Public one-shot API -----
    def get_train_test_features(
        self,
        step: int,
        window_size: int = 2,
        target_key: str = "mean_acc_per_epoch",
    ) -> Dict[str, Dict[str, Any]]:
        # Always use configured batch size
        train_dict, test_dict = self.prepare_time_window_data(
            step=step, window_size=window_size, target_key=target_key
        )

        X_tr, P_tr, y_tr, qids_tr = self.build_features(train_dict)
        X_te, P_te, y_te, qids_te = self.build_features(test_dict)
        return {
            "train": {"X": X_tr, "P": P_tr, "y": y_tr, "qids": qids_tr, "indices": [self.qid_to_idx[q] for q in qids_tr]},
            "test": {"X": X_te, "P": P_te, "y": y_te, "qids": qids_te, "indices": [self.qid_to_idx[q] for q in qids_te]},
        }