import random
import unittest
from collections import Counter
from itertools import product

from rummi import Tile, find_best_move

ALL_TILES_STRINGS = [c + str(val) for c, val in product("brya", range(1, 14))] * 2
ALL_TILES_STRINGS_WITH_JOKERS = ALL_TILES_STRINGS + ["J"] * 2


def find_best_move_strings(table_set_strings, rack_string, maximize_value=False):
    return find_best_move(
        [tuple(Tile.from_str(s)) for s in table_set_strings],
        Tile.from_str(rack_string),
        maximize_value
    )


def print_results(sets_on_table, tiles_placed, remaining_tiles):
    print("Table sets:")
    for s in sets_on_table:
        print(s)

    print("Tiles placed:")
    print(tiles_placed)

    print("Remaining tiles:")
    print(remaining_tiles)


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
        result_sets_on_table, tiles_placed, remaining_tiles = find_best_move(table_sets, rack_tiles)

        print_results(result_sets_on_table, tiles_placed, remaining_tiles)

        msg = f"sets: {result_sets_on_table} placed: {tiles_placed}; remaining: {remaining_tiles};"
        self.assertEqual(
            sum(len(s) for s in table_sets) + len(rack_tiles) - expected_remaining,
            sum(len(s) for s in result_sets_on_table), msg
        )
        self.assertEqual(len(rack_tiles) - expected_remaining, len(tiles_placed), msg)
        self.assertEqual(expected_remaining, len(remaining_tiles), msg)

        if expected_unmodified is not None:
            before_sets = Counter(table_sets)
            after_sets = Counter(result_sets_on_table)
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

    def test_all_tiles_some_on_table(self):
        random.seed(1)
        tile_strings = ALL_TILES_STRINGS_WITH_JOKERS.copy()
        random.shuffle(tile_strings)
        rack = " ".join(tile_strings[80:])
        table_tiles = " ".join(tile_strings[:80])

        # Place the first 80 tiles
        sets_on_table, tiles_placed, remaining_tiles = find_best_move([], Tile.from_str(table_tiles))
        print_results(sets_on_table, tiles_placed, remaining_tiles)
        self.assertEqual(80, len(tiles_placed))

        # Try to place the rest of them
        # Without the unnecessary change optimization there are a variable number usually around 0-5 unmodified,
        # with it should be optimal every time.
        self.validate_sets(rack, 0, sets_on_table, expected_unmodified=19)

    def test_set_on_table_not_in_minimized_sets(self):
        self.validate_sets("a4", 0, table_set_strings=["J J a2 a3"])

    def test_maximize_tiles(self):
        # Maximizing the number of tiles it should put the joker in the 5-run
        table, placed, remaining = find_best_move_strings([], "a1 a2 a4 a5 a13 b13 J", False)
        self.assertCountEqual(Tile.from_str("a1 a2 J a4 a5"), placed)

    def test_maximize_value(self):
        # Maximizing the value it should use the joker for the 13 group even though it is fewer tiles
        table, placed, remaining = find_best_move_strings([], "a1 a2 a4 a5 a13 b13 J", True)
        self.assertCountEqual(Tile.from_str("a13 b13 J"), placed)
