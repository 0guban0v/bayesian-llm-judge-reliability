"""Regression tests for PyMC IRT inference and archive shape."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
from src.analysis.posterior_archive import load_posterior
from src.models.infer import run_and_save_posterior
from src.models.irt_common import save_posterior
from src.models.irt_pymc import run_mcmc
from src.schemas import ExperimentConfig, PriorConfig


class PyMCModelTests(unittest.TestCase):
    """Verify the PyMC backend returns arrays compatible with downstream consumers."""

    def _make_config(self) -> ExperimentConfig:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml").model_copy(deep=True)
        config.inference.num_warmup = 2
        config.inference.num_samples = 4
        config.inference.num_chains = 1
        return config

    def _make_observations(self) -> dict[str, np.ndarray | int]:
        return {
            "correct": np.asarray([1, 0, 1, 0], dtype=np.int32),
            "judge_idx": np.asarray([0, 0, 1, 1], dtype=np.int32),
            "item_idx": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "source_idx": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "n_judges": 2,
            "n_items": 2,
            "n_sources": 2,
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2"]),
            "source_ids": np.asarray(["source-a", "source-b"]),
        }

    def test_run_mcmc_supports_source_hier_variant(self) -> None:
        config = self._make_config()
        config.model.variant = "source_hier"
        config.model.priors.tau_theta = PriorConfig(dist="lognormal", loc=0.0, scale=0.5)

        _, samples, ppc_summary = run_mcmc(config, self._make_observations())

        self.assertIn("theta", samples)
        self.assertIn("tau_theta", samples)
        self.assertIn("theta_source", samples)
        self.assertIn("judge_accuracy_ppc_mean", ppc_summary)
        self.assertEqual(samples["theta"].ndim, 3)
        self.assertEqual(samples["theta_source"].ndim, 4)
        self.assertEqual(samples["diverging"].ndim, 2)

    def test_run_mcmc_omits_2pl_and_hierarchical_terms_for_1pl_global(self) -> None:
        config = self._make_config()
        config.model.type = "1PL"
        config.model.variant = "global"
        config.model.priors.tau_theta = None

        _, samples, ppc_summary = run_mcmc(config, self._make_observations())

        self.assertIn("theta", samples)
        self.assertIn("b", samples)
        self.assertNotIn("a", samples)
        self.assertNotIn("tau_theta", samples)
        self.assertNotIn("theta_source", samples)
        self.assertEqual(ppc_summary["judge_accuracy_ppc_mean"].shape, (2,))
        self.assertEqual(samples["theta"].shape[:2], (1, 4))

    def test_run_mcmc_skips_log_likelihood_when_not_requested(self) -> None:
        config = self._make_config()
        config.inference.save_log_likelihood = False
        observations = self._make_observations()
        fake_idata = MagicMock()
        fake_idata.posterior = {
            "theta": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2))))),
            "b": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2))))),
            "a": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2))))),
            "tau_theta": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2))))),
            "theta_source": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2, 2))))),
        }
        fake_idata.sample_stats = {
            "diverging": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.zeros((1, 4)))))
        }
        fake_idata.posterior_predictive = {
            "correct": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 4)))))
        }
        fake_ppc_idata = MagicMock()

        with (
            patch("src.models.irt_pymc._build_model") as build_model,
            patch("src.models.irt_pymc.pm.sample", return_value=fake_idata),
            patch("src.models.irt_pymc.pm.compute_log_likelihood") as compute_log_likelihood,
            patch("src.models.irt_pymc.pm.sample_posterior_predictive", return_value=fake_ppc_idata),
            patch(
                "src.models.irt_pymc.aggregate_judge_accuracy_ppc",
                return_value={
                    "judge_accuracy_ppc_mean": np.ones(2),
                    "judge_accuracy_ppc_p05": np.ones(2),
                    "judge_accuracy_ppc_p95": np.ones(2),
                },
            ),
        ):
            build_model.return_value.__enter__.return_value = MagicMock()
            build_model.return_value.__exit__.return_value = None

            run_mcmc(config, observations)

        compute_log_likelihood.assert_not_called()

    def test_run_mcmc_computes_log_likelihood_when_requested(self) -> None:
        config = self._make_config()
        config.inference.save_log_likelihood = True
        observations = self._make_observations()
        fake_idata = MagicMock()
        fake_idata.posterior = {
            "theta": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2))))),
            "b": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2))))),
            "a": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2))))),
            "tau_theta": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2))))),
            "theta_source": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 2, 2))))),
        }
        fake_idata.sample_stats = {
            "diverging": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.zeros((1, 4)))))
        }
        fake_idata.posterior_predictive = {
            "correct": MagicMock(transpose=MagicMock(return_value=MagicMock(values=np.ones((1, 4, 4)))))
        }
        fake_ppc_idata = MagicMock()

        with (
            patch("src.models.irt_pymc._build_model") as build_model,
            patch("src.models.irt_pymc.pm.sample", return_value=fake_idata),
            patch("src.models.irt_pymc.pm.compute_log_likelihood") as compute_log_likelihood,
            patch("src.models.irt_pymc.pm.sample_posterior_predictive", return_value=fake_ppc_idata),
            patch(
                "src.models.irt_pymc.aggregate_judge_accuracy_ppc",
                return_value={
                    "judge_accuracy_ppc_mean": np.ones(2),
                    "judge_accuracy_ppc_p05": np.ones(2),
                    "judge_accuracy_ppc_p95": np.ones(2),
                },
            ),
        ):
            build_model.return_value.__enter__.return_value = MagicMock()
            build_model.return_value.__exit__.return_value = None

            run_mcmc(config, observations)

        compute_log_likelihood.assert_called_once_with(
            fake_idata,
            var_names=["correct"],
            extend_inferencedata=True,
            progressbar=False,
        )

    def test_run_and_save_posterior_rejects_incomplete_judge_coverage(self) -> None:
        config = self._make_config()
        config.judges = [
            config.judges[0].model_copy(update={"id": "judge-a"}),
            config.judges[1].model_copy(update={"id": "judge-b"}),
        ]
        matrix = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-2"],
                "item_id": ["item-1", "item-2"],
                "label": ["A>B", "B>A"],
                "original_id": [1, 2],
                "question": ["q1", "q2"],
                "source": ["source-a", "source-b"],
                "split": ["gpt", "gpt"],
                "judge-a": [1, 0],
                "judge-b": [1, None],
            }
        )

        with patch("src.models.infer.run_mcmc") as run_mcmc_mock:
            with self.assertRaisesRegex(ValueError, "complete judge coverage|Incomplete judges"):
                run_and_save_posterior(config, matrix)

        run_mcmc_mock.assert_not_called()

    def test_saved_archive_round_trips_with_expected_metadata(self) -> None:
        config = self._make_config()
        observations = self._make_observations()

        _, samples, ppc_summary = run_mcmc(config, observations)

        with tempfile.TemporaryDirectory() as temp_dir:
            posterior_path = Path(temp_dir) / "posterior.npz"
            save_posterior(
                posterior_path,
                samples,
                observations,
                config.model.type,
                metadata={
                    "backend": np.asarray("pymc"),
                    "experiment_seed": np.asarray(config.experiment.seed),
                    "num_chains": np.asarray(samples["theta"].shape[0]),
                    **ppc_summary,
                },
            )
            posterior = load_posterior(posterior_path)

        for key in (
            "backend",
            "judge_ids",
            "item_ids",
            "source_ids",
            "model_type",
            "n_obs",
            "experiment_seed",
            "num_chains",
            "posterior_schema_version",
            "judge_accuracy_ppc_mean",
            "judge_accuracy_ppc_p05",
            "judge_accuracy_ppc_p95",
            "diverging",
            "theta",
            "b",
            "a",
            "tau_theta",
            "theta_source",
        ):
            self.assertIn(key, posterior)
        self.assertEqual(str(posterior["backend"]), "pymc")
        self.assertEqual(int(posterior["num_chains"]), samples["theta"].shape[0])
        self.assertEqual(int(posterior["posterior_schema_version"]), 1)
        self.assertEqual(int(posterior["n_obs"]), len(observations["correct"]))
        self.assertEqual(posterior["theta"].shape[:2], samples["theta"].shape[:2])
        self.assertEqual(posterior["theta_source"].shape[:2], samples["theta_source"].shape[:2])


if __name__ == "__main__":
    unittest.main()
