from unittest import TestCase

from rummi_cube.structs import Tileset


class TestTilesetValueToEnterGame(TestCase):

    def test_run(self):
        value = Tileset.from_str("a2 a3 a4").numerical_value_to_enter_game()
        self.assertEqual(9, value)

    def test_group(self):
        value = Tileset.from_str("y10 r10 a10").numerical_value_to_enter_game()
        self.assertEqual(30, value)

    def test_run_with_joker(self):
        value = Tileset.from_str("a2 J a4").numerical_value_to_enter_game()
        self.assertEqual(9, value)

    def test_ambiguous_chooses_highest(self):
        value = Tileset.from_str("a2 J J").numerical_value_to_enter_game()
        self.assertEqual(9, value)
