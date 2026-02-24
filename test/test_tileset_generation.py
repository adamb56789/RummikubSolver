from unittest import TestCase

from rummi_cube.structs import Tileset
from rummi_cube.tileset_generation import generate_all_runs, generate_all_groups, generate_all_sets


class TestTilesetGeneration(TestCase):

    def test_generate_all_runs_count(self):
        self.assertEqual(1324, len(generate_all_runs()))

    def test_generate_all_runs_examples(self):
        runs = generate_all_runs()

        self.assertIn(Tileset.from_str("J J a3 a4"), runs)
        self.assertIn(Tileset.from_str("a3 a4 J J"), runs)
        self.assertIn(Tileset.from_str("a1 a2 a3 a4 a5"), runs)

        self.assertIn(Tileset.from_str("y11 J J"), runs)
        self.assertIn(Tileset.from_str("J y11 J"), runs)
        self.assertIn(Tileset.from_str("J J y11"), runs)

        self.assertNotIn(Tileset.from_str("J J a2 a3"), runs)

    def test_generate_all_groups_count(self):
        self.assertEqual(325, len(generate_all_groups()))

    def test_generate_all_tilesets_count(self):
        self.assertEqual(1605, len(generate_all_sets()))
