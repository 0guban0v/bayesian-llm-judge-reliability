"""Focused tests for pairwise verdict parsing."""

from __future__ import annotations

import unittest

from src.judges.parsers import parse_correctness, parse_verdict, swap_verdict


class ParseVerdictTests(unittest.TestCase):
    """Verify terminal verdict parsing prefers precision over broad recovery."""

    def test_parses_exact_final_verdict_a(self) -> None:
        self.assertEqual(parse_verdict("FINAL VERDICT: A"), "A")

    def test_parses_exact_final_verdict_b(self) -> None:
        self.assertEqual(parse_verdict("FINAL VERDICT: B"), "B")

    def test_prefers_terminal_verdict_over_reasoning_mentions(self) -> None:
        raw_text = (
            "Assistant A is more detailed, but Assistant B is more concise.\n"
            "After weighing correctness and helpfulness:\n"
            "FINAL VERDICT: B"
        )
        self.assertEqual(parse_verdict(raw_text), "B")

    def test_returns_unknown_for_reasoning_without_terminal_verdict(self) -> None:
        raw_text = (
            "Assistant A covers more of the request, while Assistant B is shorter.\n"
            "Both have tradeoffs, but the comparison remains close."
        )
        self.assertEqual(parse_verdict(raw_text), "UNKNOWN")

    def test_parses_terminal_bare_verdict(self) -> None:
        self.assertEqual(parse_verdict("Some reasoning here.\nB"), "B")

    def test_parses_terminal_tie_marker(self) -> None:
        self.assertEqual(parse_verdict("Comparison complete.\n[[A=B]]"), "TIE")

    def test_does_not_parse_suffix_word_as_tie(self) -> None:
        self.assertEqual(parse_verdict("Comparison complete.\nNECKTIE"), "UNKNOWN")

    def test_swap_verdict_still_maps_reversed_order(self) -> None:
        self.assertEqual(swap_verdict(parse_verdict("FINAL VERDICT: A")), "B")


class ParseCorrectnessTests(unittest.TestCase):
    """Verify verdict-to-label correctness conversion for matrix construction."""

    def test_returns_true_for_matching_a_verdict(self) -> None:
        self.assertTrue(parse_correctness("A", "A>B"))

    def test_returns_true_for_matching_b_verdict(self) -> None:
        self.assertTrue(parse_correctness("B", "B>A"))

    def test_returns_false_for_non_matching_verdict(self) -> None:
        self.assertFalse(parse_correctness("A", "B>A"))

    def test_returns_none_for_tie_verdict(self) -> None:
        self.assertIsNone(parse_correctness("TIE", "A>B"))

    def test_returns_none_for_unknown_verdict(self) -> None:
        self.assertIsNone(parse_correctness("UNKNOWN", "B>A"))

    def test_normalizes_lowercase_label_before_comparison(self) -> None:
        self.assertTrue(parse_correctness("A", "a>b"))

    def test_raises_for_unsupported_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported label: invalid"):
            parse_correctness("A", "invalid")


if __name__ == "__main__":
    unittest.main()
