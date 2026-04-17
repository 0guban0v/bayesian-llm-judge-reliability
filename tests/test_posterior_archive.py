"""Regression tests for posterior archive schema validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from src.analysis.posterior_archive import load_posterior, load_posterior_archive


class PosteriorArchiveTests(unittest.TestCase):
    """Verify saved posterior archives are validated consistently."""

    def test_load_posterior_archive_rejects_legacy_archive_without_schema_version(self) -> None:
        payload = {
            "theta": np.ones((1, 2, 2)),
            "b": np.ones((1, 2, 3)),
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2", "item-3"]),
            "source_ids": np.asarray(["source-a"]),
            "model_type": np.asarray("1PL"),
            "n_obs": np.asarray(6),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy_posterior.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "Legacy posterior archives are unsupported"):
                load_posterior_archive(path)

    def test_load_posterior_rejects_missing_required_keys(self) -> None:
        payload = {
            "theta": np.ones((1, 2, 2)),
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2"]),
            "source_ids": np.asarray(["source-a"]),
            "model_type": np.asarray("1PL"),
            "n_obs": np.asarray(4),
            "posterior_schema_version": np.asarray(1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid_missing_key.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "missing required keys: b"):
                load_posterior(path)

    def test_load_posterior_rejects_misaligned_theta_source_shape(self) -> None:
        payload = {
            "theta": np.ones((1, 2, 2)),
            "b": np.ones((1, 2, 3)),
            "a": np.ones((1, 2, 3)),
            "theta_source": np.ones((1, 2, 3, 2)),
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2", "item-3"]),
            "source_ids": np.asarray(["source-a"]),
            "model_type": np.asarray("2PL"),
            "n_obs": np.asarray(6),
            "posterior_schema_version": np.asarray(1),
            "judge_accuracy_ppc_mean": np.asarray([0.5, 0.5]),
            "judge_accuracy_ppc_p05": np.asarray([0.4, 0.4]),
            "judge_accuracy_ppc_p95": np.asarray([0.6, 0.6]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid_theta_source.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "theta_source shape does not match"):
                load_posterior(path)

    def test_load_posterior_rejects_missing_judge_accuracy_ppc_fields(self) -> None:
        payload = {
            "theta": np.ones((1, 2, 2)),
            "b": np.ones((1, 2, 3)),
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2", "item-3"]),
            "source_ids": np.asarray(["source-a"]),
            "model_type": np.asarray("1PL"),
            "n_obs": np.asarray(6),
            "posterior_schema_version": np.asarray(1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid_ppc_missing.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "unsupported for current analysis outputs"):
                load_posterior(path)

    def test_load_posterior_rejects_judge_accuracy_ppc_length_mismatch(self) -> None:
        payload = {
            "theta": np.ones((1, 2, 2)),
            "b": np.ones((1, 2, 3)),
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2", "item-3"]),
            "source_ids": np.asarray(["source-a"]),
            "model_type": np.asarray("1PL"),
            "n_obs": np.asarray(6),
            "posterior_schema_version": np.asarray(1),
            "judge_accuracy_ppc_mean": np.asarray([0.5]),
            "judge_accuracy_ppc_p05": np.asarray([0.4]),
            "judge_accuracy_ppc_p95": np.asarray([0.6]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid_ppc_length.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "judge_accuracy_ppc_mean length does not match judge_ids length"):
                load_posterior(path)


if __name__ == "__main__":
    unittest.main()
