"""Regression tests for MLX model cache management."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from src.judges import mlx_backend


class MlxBackendCacheTests(unittest.TestCase):
    """Verify the MLX backend cache can be reused and explicitly cleared."""

    def tearDown(self) -> None:
        mlx_backend.clear_model_cache()

    def test_get_model_reuses_cache_until_cleared(self) -> None:
        load_calls: list[tuple[str, dict[str, bool] | None]] = []

        def fake_load(
            model_name: str,
            tokenizer_config: dict[str, bool] | None = None,
        ) -> tuple[object, object]:
            load_calls.append((model_name, tokenizer_config))
            return object(), object()

        fake_mlx_lm = types.SimpleNamespace(load=fake_load)

        with patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}):
            first = mlx_backend.get_model("judge-a", trust_remote_code=False)
            second = mlx_backend.get_model("judge-a", trust_remote_code=False)

            self.assertIs(first, second)
            self.assertEqual(load_calls, [("judge-a", None)])

            mlx_backend.clear_model_cache()

            third = mlx_backend.get_model("judge-a", trust_remote_code=False)

        self.assertIsNot(first, third)
        self.assertEqual(load_calls, [("judge-a", None), ("judge-a", None)])


if __name__ == "__main__":
    unittest.main()
