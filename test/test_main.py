import random
import unittest
from collections import Counter
from itertools import product

from rummi import Tile, MaximizeMode, JokerMode, find_best_move, Config, find_best_move_strings

ALL_TILES_STRINGS = [c + str(val) for c, val in product("brya", range(1, 14))] * 2
ALL_TILES_STRINGS_WITH_JOKERS = ALL_TILES_STRINGS + ["J"] * 2


class TestRummi(unittest.TestCase):

    def validate_sets(
            self,
            rack_str,
            expected_remaining,
            table_sets=None,
            table_set_strings=None,
            expected_unmodified=None
    ):
        if table_sets is None:
            table_sets = []
            if table_set_strings is not None:
                for s in table_set_strings:
                    table_sets.append((tuple(Tile.from_str(s))))
        rack_tiles = Tile.from_str(rack_str)

        config = Config(joker_mode=JokerMode.FREE, maximize_mode=MaximizeMode.TILES_PLACED)
        result = find_best_move(table_sets, rack_tiles, config)

        print(result)

        self.assertEqual(
            sum(len(s) for s in table_sets) + len(rack_tiles) - expected_remaining,
            sum(len(s) for s in result.table), result
        )
        self.assertEqual(len(rack_tiles) - expected_remaining, len(result.placed), result)
        self.assertEqual(expected_remaining, len(result.remaining), result)

        if expected_unmodified is not None:
            before_sets = Counter(table_sets)
            after_sets = Counter(result.table)
            unmodified_sets = before_sets & after_sets
            print("Unmodified sets:", len(unmodified_sets))

            self.assertEqual(expected_unmodified, len(unmodified_sets))

    def test_legacy(self):
        cases = [
            ("r1 r2 r3 r4 r5 r6 r7 r8 r9 a2 a3 a4", 0),
            ("r1 r2 r3 r4 r5 r6 r7 r8 r9 a2 a3 a4 a5", 0),
            ("r1 r2 r3 r4 r5 r6 r7 r8 r9 a2 a3 a4 a5 b8 b9 b10 b11 b12 b13", 0),
            ("r1 r2 r3 r4 r5 r6 y13", 1),
            ("r1 r2 r3 r4 r5 r6 r7 r8 y13", 1),
            ("a1 b1 r2 y2 a1 a2 a3", 4),
        ]
        for case in cases:
            with self.subTest(msg=case):
                self.validate_sets(case[0], case[1])

    def test_handles_duplicates(self):
        self.validate_sets("a1 b1 r2 y2 a1 a2 a3", 4)

    def test_joker(self):
        self.validate_sets("a1 b1 r2 y2 a1 a2 a3 J0 J0", 0)

    def test_one_joker(self):
        self.validate_sets("a1 a2 a3 J", 0)

    def test_no_tiles(self):
        self.validate_sets("", 0)

    def test_random_tiles(self):
        random.seed(0)
        expected_remaining = [9, 12, 9, 8, 9, 11, 13, 16, 8, 4]
        for expected in expected_remaining:
            tiles = " ".join(random.sample(ALL_TILES_STRINGS, k=50))
            with self.subTest(msg=tiles):
                self.validate_sets(tiles, expected)

    def test_full_cover_random_50(self):
        random.seed(273)
        tiles = " ".join(random.sample(ALL_TILES_STRINGS, k=50))
        self.validate_sets(tiles, 0)

    def test_full_cover_random_45(self):
        random.seed(2836)
        tiles = " ".join(random.sample(ALL_TILES_STRINGS, k=45))
        self.validate_sets(tiles, 0)

    def test_full_cover_random_40(self):
        random.seed(49473)
        tiles = " ".join(random.sample(ALL_TILES_STRINGS, k=40))
        self.validate_sets(tiles, 0)

    def test_all_tiles(self):
        self.validate_sets(" ".join(ALL_TILES_STRINGS_WITH_JOKERS), 0)

    # def test_all_tiles_some_on_table(self):
    #     # TODO sometimes only finds 20?
    #     random.seed(1)
    #     tile_strings = ALL_TILES_STRINGS_WITH_JOKERS.copy()
    #     random.shuffle(tile_strings)
    #     rack = " ".join(tile_strings[80:])
    #     table_tiles = " ".join(tile_strings[:80])
    #
    #     # Place the first 80 tiles
    #     sets_on_table, tiles_placed, remaining_tiles = find_best_move([], Tile.from_str(table_tiles))
    #     print_results(sets_on_table, tiles_placed, remaining_tiles)
    #     self.assertEqual(80, len(tiles_placed))
    #
    #     # Try to place the rest of them
    #     # Without the unnecessary change optimization there are a variable number usually around 0-5 unmodified,
    #     # with it should be optimal every time.
    #     self.validate_sets(rack, 0, sets_on_table, expected_unmodified=19)

    def test_set_on_table_not_in_minimized_sets(self):
        self.validate_sets("a4", 0, table_set_strings=["J J a2 a3"])

    def test_maximize_tiles(self):
        # Maximizing the number of tiles it should put the joker in the 5-run
        config = Config(JokerMode.FREE, MaximizeMode.TILES_PLACED)
        result = find_best_move_strings([], "a1 a2 a4 a5 a13 b13 J", config)
        self.assertCountEqual(Tile.from_str("a1 a2 J a4 a5"), result.placed)

    def test_maximize_value(self):
        # Maximizing the value it should use the joker for the 13 group even though it is fewer tiles
        config = Config(JokerMode.FREE, MaximizeMode.VALUE_PLACED, joker_value=0)
        result = find_best_move_strings([], "a1 a2 a4 a5 a13 b13 J", config)
        self.assertCountEqual(Tile.from_str("a13 b13 J"), result.placed)

    def test_maximize_value_plays_joker_when_30(self):
        config = Config(JokerMode.FREE, MaximizeMode.VALUE_PLACED, joker_value=30)
        result = find_best_move_strings([], "a1 a2 a3 J", config)
        self.assertCountEqual(Tile.from_str("a1 a2 a3 J"), result.placed)

    def test_maximize_value_does_not_play_joker_when_negative(self):
        config = Config(JokerMode.FREE, MaximizeMode.VALUE_PLACED, joker_value=-1)
        result = find_best_move_strings([], "a1 a2 a3 J", config)
        self.assertCountEqual(Tile.from_str("a1 a2 a3"), result.placed)

    def test_minimize_rearrange(self):
        # To ensure the model doesn't just happen to pick the right one, we test it can both minimize and maximize rearrangement

        config = Config(JokerMode.FREE, MaximizeMode.TILES_PLACED, rearrange_value=1 / 40)
        result = find_best_move_strings(["a1 a2 a3", "r1 r2 r3", "b1 b2 b3"], "a4 r4 b4", config)

        expected_table = ["a1 a2 a3", "r1 r2 r3", "b1 b2 b3", "b4 r4 a4"]
        self.assertCountEqual([tuple(Tile.from_str(s)) for s in expected_table], result.table)

        config = Config(JokerMode.FREE, MaximizeMode.TILES_PLACED, rearrange_value=-1 / 40)
        result = find_best_move_strings(["a1 a2 a3", "r1 r2 r3", "b1 b2 b3"], "a4 r4 b4", config)

        expected_table = ["b1 r1 a1", "b2 r2 a2", "b3 r3 a3", "b4 r4 a4"]
        self.assertCountEqual([tuple(Tile.from_str(s)) for s in expected_table], result.table)

    def test_joker_locked_cannot_move(self):
        config = Config(JokerMode.FREE, MaximizeMode.VALUE_PLACED, joker_value=0)
        result = find_best_move_strings(["J a2 a3 a4"], "r1 r2", config)
        print(result)
