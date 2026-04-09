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

    def test_generate_text_constrains_output_and_reconstructs_full_verdict(self) -> None:
        generate_calls: list[dict[str, object]] = []

        class FakeTokenizer:
            eos_token_ids = [99]

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                *,
                tokenize: bool,
                continue_final_message: bool,
                add_generation_prompt: bool,
            ) -> str:
                self.messages = messages
                self.template_flags = {
                    "tokenize": tokenize,
                    "continue_final_message": continue_final_message,
                    "add_generation_prompt": add_generation_prompt,
                }
                return "rendered prompt"

            def encode(self, text: str, _add_special_tokens: bool = False) -> list[int]:
                mapping = {
                    "A": [11],
                    "B": [12],
                    " A": [21],
                    " B": [22],
                }
                return mapping[text]

        fake_tokenizer = FakeTokenizer()
        fake_model = object()

        def fake_load(
            model_name: str,
            tokenizer_config: dict[str, bool] | None = None,
        ) -> tuple[object, FakeTokenizer]:
            self.assertEqual(model_name, "judge-a")
            self.assertIsNone(tokenizer_config)
            return fake_model, fake_tokenizer

        def fake_generate(model: object, tokenizer: object, **kwargs: object) -> str:
            self.assertIs(model, fake_model)
            self.assertIs(tokenizer, fake_tokenizer)
            generate_calls.append(kwargs)
            return " A\n"

        fake_mx_core = types.SimpleNamespace(
            arange=lambda size: list(range(size)),
            logical_or=lambda left, right: left or right,
            where=lambda condition, left, right: left if condition else right,
            full=lambda shape, value: value,
        )
        fake_mlx_lm = types.SimpleNamespace(load=fake_load, generate=fake_generate)

        with patch.dict(
            sys.modules,
            {
                "mlx_lm": fake_mlx_lm,
                "mlx": types.SimpleNamespace(core=fake_mx_core),
                "mlx.core": fake_mx_core,
            },
        ):
            response = mlx_backend.generate_text(
                model_name="judge-a",
                prompt="judge prompt",
                max_tokens=5,
                trust_remote_code=False,
            )

        self.assertEqual(response, "FINAL VERDICT: A")
        self.assertEqual(len(generate_calls), 1)
        generate_kwargs = generate_calls[0]
        self.assertEqual(generate_kwargs["prompt"], "rendered prompt")
        self.assertEqual(generate_kwargs["max_tokens"], 5)
        self.assertFalse(generate_kwargs["verbose"])
        self.assertEqual(len(generate_kwargs["logits_processors"]), 1)
        self.assertEqual(
            fake_tokenizer.messages,
            [
                {"role": "user", "content": "judge prompt"},
                {"role": "assistant", "content": "FINAL VERDICT: "},
            ],
        )
        self.assertEqual(
            fake_tokenizer.template_flags,
            {
                "tokenize": False,
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
