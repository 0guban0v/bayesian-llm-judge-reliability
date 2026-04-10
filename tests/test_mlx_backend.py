"""Regression tests for MLX model cache management."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
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
            arange=np.arange,
            logical_or=np.logical_or,
            where=np.where,
            full=np.full,
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
        processor = generate_kwargs["logits_processors"][0]
        base_logits = np.arange(100, dtype=float)[None, :]
        verdict_step = processor([], base_logits.copy())
        eos_step = processor([], base_logits.copy())
        allowed_verdict_ids = {11, 12, 21, 22}
        for token_id in range(base_logits.shape[-1]):
            if token_id in allowed_verdict_ids:
                self.assertEqual(verdict_step[0, token_id], base_logits[0, token_id])
            else:
                self.assertEqual(verdict_step[0, token_id], float("-inf"))
        for token_id in range(base_logits.shape[-1]):
            if token_id == 99:
                self.assertEqual(eos_step[0, token_id], base_logits[0, token_id])
            else:
                self.assertEqual(eos_step[0, token_id], float("-inf"))
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


class MlxBackendTokenizerHelperTests(unittest.TestCase):
    """Verify tokenizer helper behavior for constrained decoding."""

    class FakeTokenizerWithKeywords:
        eos_token_ids = [91, 92]

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            mapping = {
                "A": [11],
                "B": [12],
                " A": [21],
                " B": [22],
                "multi": [1, 2],
            }
            return mapping[text]

    class FakeTokenizerWithoutKeywordArg:
        eos_token_id = 99

        def encode(self, text: str) -> list[int]:
            mapping = {
                "A": [11],
                "B": [12],
                " A": [21],
                " B": [22],
            }
            return mapping[text]

    def test_encode_token_ids_uses_add_special_tokens_when_supported(self) -> None:
        tokenizer = self.FakeTokenizerWithKeywords()

        token_ids = mlx_backend._encode_token_ids(tokenizer, "multi")

        self.assertEqual(token_ids, [1, 2])

    def test_encode_token_ids_falls_back_when_tokenizer_rejects_keyword(self) -> None:
        tokenizer = self.FakeTokenizerWithoutKeywordArg()

        token_ids = mlx_backend._encode_token_ids(tokenizer, "A")

        self.assertEqual(token_ids, [11])

    def test_resolve_verdict_token_ids_returns_all_single_token_a_b_forms(self) -> None:
        tokenizer = self.FakeTokenizerWithKeywords()

        token_ids = mlx_backend._resolve_verdict_token_ids(tokenizer)

        self.assertEqual(token_ids, [11, 12, 21, 22])

    def test_resolve_verdict_token_ids_rejects_missing_single_token_forms(self) -> None:
        class MultiTokenOnlyTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                assert add_special_tokens is False
                return [1, 2]

        with self.assertRaisesRegex(
            ValueError,
            "Tokenizer does not provide single-token verdict labels for both A and B",
        ):
            mlx_backend._resolve_verdict_token_ids(MultiTokenOnlyTokenizer())

    def test_resolve_verdict_token_ids_rejects_when_only_one_label_is_single_token(self) -> None:
        class OnlyATokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                assert add_special_tokens is False
                mapping = {
                    "A": [11],
                    " A": [21],
                    "B": [1, 2],
                    " B": [3, 4],
                }
                return mapping[text]

        with self.assertRaisesRegex(
            ValueError,
            "Tokenizer does not provide single-token verdict labels for both A and B",
        ):
            mlx_backend._resolve_verdict_token_ids(OnlyATokenizer())

    def test_resolve_eos_token_ids_prefers_plural_eos_ids(self) -> None:
        tokenizer = self.FakeTokenizerWithKeywords()

        eos_ids = mlx_backend._resolve_eos_token_ids(tokenizer)

        self.assertEqual(eos_ids, [91, 92])

    def test_resolve_eos_token_ids_falls_back_to_singular_eos_id(self) -> None:
        tokenizer = self.FakeTokenizerWithoutKeywordArg()

        eos_ids = mlx_backend._resolve_eos_token_ids(tokenizer)

        self.assertEqual(eos_ids, [99])

    def test_resolve_eos_token_ids_rejects_missing_eos(self) -> None:
        class NoEosTokenizer:
            pass

        with self.assertRaisesRegex(
            ValueError,
            "Tokenizer does not expose EOS token IDs for constrained generation",
        ):
            mlx_backend._resolve_eos_token_ids(NoEosTokenizer())


if __name__ == "__main__":
    unittest.main()
