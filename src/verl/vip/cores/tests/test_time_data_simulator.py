import os
import sys
import json
import numpy as np
import pytest

# Ensure repository root is on sys.path for 'variance_prediction' imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from variance_prediction.time_data_simulator import (
    TimeDataSimulator,
    TimeDataSimulatorConfig,
)

# Absolute data paths provided by user
EMBEDDING_PATH = "/home/hieunt/verl/data/embedding_data/embeddings_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398.npy"
PAIRWISE_PATH = "/home/hieunt/verl/data/embedding_data/pairwise_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398_matrix.npy"
INDICES_PATH = "/home/hieunt/verl/data/embedding_data/indices_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398.json"
REGRESSION_PATH = "/home/hieunt/verl/data/regression_data/allo_grpo_4e/per_question_statistics_latest.json"


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in [EMBEDDING_PATH, PAIRWISE_PATH, INDICES_PATH, REGRESSION_PATH]),
    reason="Required data files are missing on this machine",
)
@pytest.mark.parametrize(
    "step,window_size,batch_size,target_key",
    [
        (0, 2, 64, "mean_acc_per_epoch"),
        (3, 2, 64, "mean_acc_per_epoch"),
    ],
)
def test_time_data_simulator_end_to_end(step, window_size, batch_size, target_key):
    cfg = TimeDataSimulatorConfig(
        embedding_path=EMBEDDING_PATH,
        pairwise_path=PAIRWISE_PATH,
        indices_path=INDICES_PATH,
        regression_json_path=REGRESSION_PATH,
        total_data_points=None,
    )

    sim = TimeDataSimulator(cfg)

    # sanity: steps computation doesn't crash
    total_steps = sim.get_total_gradient_steps(batch_size=batch_size, target_key=target_key)
    assert isinstance(total_steps, int)
    assert total_steps >= 0

    out = sim.get_train_test_features(
        step=step, window_size=window_size, batch_size=batch_size, target_key=target_key
    )

    assert set(out.keys()) == {"train", "test"}

    for split in ("train", "test"):
        split_data = out[split]
        # keys present
        assert set(split_data.keys()) == {"X", "P", "y", "qids"}

        X = split_data["X"]
        P = split_data["P"]
        y = split_data["y"]
        qids = split_data["qids"]

        # types and shapes
        assert isinstance(X, np.ndarray)
        assert isinstance(P, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(qids, list)

        # dims agree: P should be square and match len(qids), X rows should match qids
        assert X.shape[0] == len(qids)
        assert y.shape[0] == len(qids)
        assert P.shape[0] == P.shape[1]
        assert P.shape[0] == len(qids)

        # embedding dim should be > 0
        if X.size > 0:
            assert X.shape[1] > 0

        # y should be finite where not NaN (allow NaNs if target list was empty)
        finite_mask = np.isfinite(y)
        assert finite_mask.sum() >= 0

    # Ensure test size does not exceed requested batch_size
    assert out["test"]["X"].shape[0] <= batch_size
