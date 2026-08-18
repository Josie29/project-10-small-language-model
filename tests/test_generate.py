from __future__ import annotations

import unittest

from generate import dry_candidates, extract_json_array
from slm.generation import enumerate_cells


class JsonExtractionTests(unittest.TestCase):
    """Teachers wrap JSON in prose or a fence often enough that a strict `json.loads` on
    the whole response would silently discard usable batches — each empty parse is a
    whole cell of the grid left unfilled."""

    def test_parses_a_plain_array(self) -> None:
        self.assertEqual(extract_json_array('[{"a": 1}, {"b": 2}]'), [{"a": 1}, {"b": 2}])

    def test_parses_an_array_inside_a_markdown_fence(self) -> None:
        text = 'Here you go:\n```json\n[{"a": 1}]\n```'

        self.assertEqual(extract_json_array(text), [{"a": 1}])

    def test_parses_a_bare_object_whose_first_bracket_is_a_nested_list(self) -> None:
        """The array pattern greedily matches the nested `[1, 2]`, parses it into a list
        of non-objects, and would report the whole response as unparseable — so a single
        example carrying any list field would vanish."""
        text = '{"forbidden_fix_tokens": [1, 2], "code": "x"}'

        self.assertEqual(
            extract_json_array(text), [{"forbidden_fix_tokens": [1, 2], "code": "x"}]
        )

    def test_returns_empty_for_prose_and_broken_json(self) -> None:
        self.assertEqual(extract_json_array("I cannot do that."), [])
        self.assertEqual(extract_json_array('[{"a": }]'), [])


class DryRunTests(unittest.TestCase):
    def test_dry_run_covers_every_cell(self) -> None:
        """The dry run is the only check that the whole pipeline is wired before any money
        is spent, so it has to exercise the full grid rather than a sample of it."""
        candidates = dry_candidates()

        self.assertEqual(len(candidates), len(enumerate_cells()))
        self.assertEqual({c.cell for c in candidates}, set(enumerate_cells()))
