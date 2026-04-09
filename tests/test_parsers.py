"""Focused tests for pairwise verdict parsing."""

from __future__ import annotations

import unittest

from src.judges.parsers import parse_verdict, swap_verdict


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

    def test_swap_verdict_still_maps_reversed_order(self) -> None:
        self.assertEqual(swap_verdict(parse_verdict("FINAL VERDICT: A")), "B")


if __name__ == "__main__":
    unittest.main()
