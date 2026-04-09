"""Regression tests for BlackJAX reproducibility and posterior metadata."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from src.analysis.diagnostics import POSTERIOR_METADATA_KEYS, load_posterior
from src.models.irt_blackjax import run_blackjax
from src.models.irt_numpyro import save_posterior
from src.schemas import ExperimentConfig


class BlackJaxReproducibilityTests(unittest.TestCase):
    """Verify BlackJAX replay is deterministic for a fixed config and seed."""

    def test_run_blackjax_is_deterministic_for_fixed_seed(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml").model_copy(deep=True)
        config.inference.num_warmup = 2
        config.inference.num_samples = 4
        config.inference.num_chains = 2
        observations = {
            "correct": np.asarray([1, 0, 1, 0], dtype=np.int32),
            "judge_idx": np.asarray([0, 0, 1, 1], dtype=np.int32),
            "item_idx": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "n_judges": 2,
            "n_items": 2,
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2"]),
        }

        first = run_blackjax(config, observations)
        second = run_blackjax(config, observations)

        for key in ("theta", "b", "a", "diverging"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_save_posterior_persists_backend_seed_and_chain_metadata(self) -> None:
        observations = {
            "correct": np.asarray([1, 0], dtype=np.int32),
            "judge_ids": np.asarray(["judge-a"]),
            "item_ids": np.asarray(["item-1", "item-2"]),
        }
        samples = {
            "theta": np.asarray([[[0.1]]]),
            "b": np.asarray([[[0.2, -0.1]]]),
            "diverging": np.asarray([[False]]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            posterior_path = Path(temp_dir) / "posterior.npz"
            save_posterior(
                posterior_path,
                samples,
                observations,
                "1PL",
                metadata={
                    "backend": np.asarray("blackjax"),
                    "experiment_seed": np.asarray(7),
                    "num_chains": np.asarray(2),
                },
            )
            posterior = load_posterior(posterior_path)

        self.assertEqual(str(posterior["backend"]), "blackjax")
        self.assertEqual(int(posterior["experiment_seed"]), 7)
        self.assertEqual(int(posterior["num_chains"]), 2)
        self.assertTrue(
            {"backend", "experiment_seed", "num_chains"}.issubset(POSTERIOR_METADATA_KEYS)
        )


if __name__ == "__main__":
    unittest.main()
