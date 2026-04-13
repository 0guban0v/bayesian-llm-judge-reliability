"""Regression tests for profile command helpers."""

from __future__ import annotations

import unittest

from scripts.profile_command import normalize_child_ru_maxrss_bytes


class NormalizeChildRuMaxRssBytesTests(unittest.TestCase):
    """Verify platform-specific ru_maxrss normalization."""

    def test_linux_values_are_interpreted_as_kibibytes(self) -> None:
        self.assertEqual(normalize_child_ru_maxrss_bytes(512, "linux"), 512 * 1024)

    def test_darwin_values_are_interpreted_as_bytes(self) -> None:
        self.assertEqual(normalize_child_ru_maxrss_bytes(512, "darwin"), 512)

    def test_negative_values_are_clamped_to_zero(self) -> None:
        self.assertEqual(normalize_child_ru_maxrss_bytes(-1, "linux"), 0)


if __name__ == "__main__":
    unittest.main()
