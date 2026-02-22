import random
import unittest
from collections import Counter
from itertools import product

from rummi import find_best_move, find_best_move_strings, SETS
from structs import Tile, MaximizeMode, JokerMode, Config, RummiResult

JOKER_LOCK_CONFIG = Config(JokerMode.LOCKING, MaximizeMode.VALUE_PLACED, joker_value=0)

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

    def test_group_order_does_not_matter(self):
        self.validate_sets("y1", 0, table_set_strings=["a1 b1 r1"])
        self.validate_sets("y1", 0, table_set_strings=["b1 a1 r1"])

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

    def test_random_tiles_with_jokers(self):
        random.seed(0)
        expected_remaining = [3, 5, 7, 7, 7, 11, 16, 7, 8, 9]
        for expected in expected_remaining:
            tiles = " ".join(random.sample(ALL_TILES_STRINGS_WITH_JOKERS, k=50))
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

        self.assert_sets_equal(["a1 a2 a3", "r1 r2 r3", "b1 b2 b3", "a4 b4 r4"], result.table)

        config = Config(JokerMode.FREE, MaximizeMode.TILES_PLACED, rearrange_value=-1 / 40)
        result = find_best_move_strings(["a1 a2 a3", "r1 r2 r3", "b1 b2 b3"], "a4 r4 b4", config)

        self.assert_sets_equal(["a1 b1 r1", "a2 b2 r2", "a3 b3 r3", "a4 b4 r4"], result.table)

    def test_joker_locked_cannot_move(self):
        result = find_best_move_strings(["J a4 a5 a6"], "r1 r2", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["J a4 a5 a6"], "", "r1 r2")
        self.assert_result_equal(expected, result)

    def test_joker_locked_cannot_move_group(self):
        result = find_best_move_strings(["J a4 b4 y4"], "r2 r3", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["J a4 b4 y4"], "", "r2 r3")
        self.assert_result_equal(expected, result)

    def test_joker_locked_with_sub_can_move(self):
        result = find_best_move_strings(["J a4 a5"], "r1 r2 a3", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["a3 a4 a5", "r1 r2 J"], "r1 r2 a3", "")
        self.assert_result_equal(expected, result)

    def test_joker_locked_with_sub_can_move_group(self):
        result = find_best_move_strings(["a4 b4 J"], "y4 r2 r3", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["a4 b4 y4", "r2 r3 J"], "y4 r2 r3", "")
        self.assert_result_equal(expected, result)

    def test_locked_joker_cannot_take_other_tiles(self):
        result = find_best_move_strings(["J y2 y3 y4 y5 y6"], "b6 a6", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["J y2 y3 y4 y5 y6"], "", "b6 a6")
        self.assert_result_equal(expected, result)

    def test_locked_joker_can_add(self):
        result = find_best_move_strings(["y3 y4 J y6 y7 y8 y9 y10 y11 y12"], "y1 y2 y13", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["y1 y2 y3 y4 J y6 y7 y8 y9 y10 y11 y12 y13"], "y1 y2 y13", "")
        self.assert_result_equal(expected, result)

    def test_locked_joker_can_add_group(self):
        result = find_best_move_strings(["a4 b4 J"], "r4", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["J a4 b4 r4"], "r4", "")
        self.assert_result_equal(expected, result)

        result = find_best_move_strings(["a4 b4 J"], "y4", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["J a4 b4 y4"], "y4", "")
        self.assert_result_equal(expected, result)

    def test_two_jokers_can_only_replace_one(self):
        result = find_best_move_strings(["J a4 a5", "J y2 y3 y4 y5 y6"], "r1 r2 a3 b6 a6", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["J y2 y3 y4 y5 y6", "a3 a4 a5", "a6 b6 J"], "a3 b6 a6", "r1 r2")
        self.assert_result_equal(expected, result)

        # Check that we can manipulate both if not joker locking
        config = Config(JokerMode.FREE, MaximizeMode.VALUE_PLACED, joker_value=0)
        result = find_best_move_strings(["J a4 a5", "J y2 y3 y4 y5 y6"], "r1 r2 a3 b6 a6", config)

        expected = RummiResult.from_strings(["y2 y3 y4 y5 y6", "a3 a4 a5", "a6 b6 J", "r1 r2 J"], "r1 r2 a3 b6 a6", "")
        self.assert_result_equal(expected, result)

    def test_jokers_compete_for_same_tile(self):
        result = find_best_move_strings(["J a4 a5 a6", "J a4 a5 a6"], "a3 r6 r6 b6 b6", JOKER_LOCK_CONFIG)
        # Table                               Tiles in play     Rack
        # ["J a4 a5 a6", "J a4 a5 a6"]        ""                "a3 r6 r6 b6 b6"
        # ["J a4 a5 a6", "a3 a4 a5 a6"]       "J"               "r6 r6 b6 b6"
        # ["J a4 a5 a6", "a3 a4 a5"]          "J a6"            "r6 r6 b6 b6"
        # ["J a4 a5 a6", "a3 a4 a5", "a6 b6 r6", "a6 b6 J"]

        expected = RummiResult.from_strings(["J a4 a5 a6", "a3 a4 a5", "a6 b6 r6", "b6 r6 J"], "a3 r6 r6 b6 b6", "")
        self.assert_result_equal(expected, result)

    def test_jokers_compete_for_same_tile_strict(self):
        # a7 can illegally sub for both jokers to place all tiles
        # check that this is prevented and it can only sub for one of them
        result = find_best_move_strings(["a4 a5 a6 J", "J a8 a9 a10"], "a7 b4 r4 b10", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["a4 a5 a6 J", "a7 a8 a9", "a10 b10 J", ], "a7 b10", "b4 r4")
        self.assert_result_equal(expected, result)

    def test_either_colour_can_replace_joker_in_group(self):
        result = find_best_move_strings(["J a2 b2", "y3 y4 y5"], "r2 y2 b7 b8", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["a2 b2 r2 y2", "y3 y4 y5", "b7 b8 J"], "r2 y2 b7 b8", "")
        self.assert_result_equal(expected, result)

    def test_two_jokers_subbed_by_two_of_same_tile(self):
        result = find_best_move_strings(["J a2 a3 a4", "J a2 a3 a4"], "a1 b4 r4 a1 b4 r4", JOKER_LOCK_CONFIG)

        table = ["a1 a2 a3 a4", "a1 a2 a3 a4", "b4 r4 J", "b4 r4 J"]
        expected = RummiResult.from_strings(table, "a1 b4 r4 a1 b4 r4", "")
        self.assert_result_equal(expected, result)

    def test_one_joker_could_be_subbed_twice_but_is_not(self):
        result = find_best_move_strings(["a4 b4 J"], "a4 r4 y4", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["a4 b4 J", "a4 r4 y4"], "a4 r4 y4", "")
        self.assert_result_equal(expected, result)

    def test_two_jokers_on_table_in_one_set_can_add(self):
        result = find_best_move_strings(["a1 J J a4"], "a5", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["a1 J J a4 a5"], "a5", "")
        self.assert_result_equal(expected, result)

    def test_two_jokers_in_group_on_table_replace_one(self):
        result = find_best_move_strings(["J J a1 b1"], "r1 a2 b2", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["J a1 b1 r1", "a2 b2 J"], "r1 a2 b2", "")
        self.assert_result_equal(expected, result)

    def test_ambiguous_two_joker_set_can_be_run(self):
        result = find_best_move_strings(["a1 J J"], "a3 y4 y5", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["a1 J a3", "y4 y5 J"], "a3 y4 y5", "")
        self.assert_result_equal(expected, result)

    def test_ambiguous_two_joker_set_can_be_group(self):
        result = find_best_move_strings(["a1 J J"], "b1 y4 y5", JOKER_LOCK_CONFIG)

        expected = RummiResult.from_strings(["J a1 b1", "y4 y5 J"], "b1 y4 y5", "")
        self.assert_result_equal(expected, result)

    def test_random_two_sets_on_table_with_one_joker_each_locking(self):
        random.seed(0)
        expected_placed_list = [10, 9, 6, 7, 7, 7, 10, 9, 10, 4]
        for expected_placed in expected_placed_list:
            sets_with_no_joker = [s for s in SETS if Counter(t.colour for t in s).get("J", 0) == 0]
            sets_with_one_joker = [s for s in SETS if Counter(t.colour for t in s).get("J", 0) == 1]
            table_sets = random.choices(sets_with_one_joker, k=2) + random.choices(sets_with_no_joker, k=10)

            rack_string = " ".join(random.sample(ALL_TILES_STRINGS, k=10))

            with self.subTest(msg=rack_string):
                result = find_best_move(table_sets, Tile.from_str(rack_string), JOKER_LOCK_CONFIG)
                self.assertEqual(expected_placed, len(result.placed))

    def test_random_one_set_on_table_with_both_jokers_locking(self):
        random.seed(0)
        expected_placed_list = [6, 8, 8, 9, 8, 7, 3, 2, 8, 5]
        for expected_placed in expected_placed_list:
            sets_with_no_joker = [s for s in SETS if Counter(t.colour for t in s).get("J", 0) == 0]
            sets_with_two_jokers = [s for s in SETS if Counter(t.colour for t in s).get("J", 0) == 2]
            table_sets = random.choices(sets_with_two_jokers, k=1) + random.choices(sets_with_no_joker, k=10)

            rack_string = " ".join(random.sample(ALL_TILES_STRINGS, k=10))

            with self.subTest(msg=rack_string):
                result = find_best_move(table_sets, Tile.from_str(rack_string), JOKER_LOCK_CONFIG)
                self.assertEqual(expected_placed, len(result.placed))

    def test_random_one_set_on_table_with_one_joker_and_joker_on_rack_locking(self):
        random.seed(0)
        expected_placed_list = [9, 9, 9, 11, 8, 9, 7, 7, 8, 6]
        for expected_placed in expected_placed_list:
            sets_with_no_joker = [s for s in SETS if Counter(t.colour for t in s).get("J", 0) == 0]
            sets_with_one_joker = [s for s in SETS if Counter(t.colour for t in s).get("J", 0) == 1]
            table_sets = random.choices(sets_with_one_joker, k=1) + random.choices(sets_with_no_joker, k=10)

            rack_string = " ".join(random.sample(ALL_TILES_STRINGS, k=10)) + " J"

            with self.subTest(msg=rack_string):
                result = find_best_move(table_sets, Tile.from_str(rack_string), JOKER_LOCK_CONFIG)
                self.assertEqual(expected_placed, len(result.placed))

    def test_what_happens_if_you_add_more_than_two_jokers(self):
        result = find_best_move_strings(["a1 J J J"], "b1 y4 y5", JOKER_LOCK_CONFIG)
        expected = RummiResult.from_strings(["J J a1 b1", "y4 y5 J"], "b1 y4 y5", "")
        self.assert_result_equal(expected, result)

        result = find_best_move_strings(["J a2 J J a5 a6 J"], "a4 y4 y5", JOKER_LOCK_CONFIG)
        expected = RummiResult.from_strings(["J a2 J a4 a5 a6 J", "y4 y5 J"], "a4 y4 y5", "")
        self.assert_result_equal(expected, result)

        result = find_best_move_strings(["J J J J J J a7 J J J J J J"], "a6 y4 y5", JOKER_LOCK_CONFIG)
        expected = RummiResult.from_strings(["J J J J J a6 a7 J J J J J J", "y4 y5 J"], "a6 y4 y5", "")
        self.assert_result_equal(expected, result)

    def assert_sets_equal(self, expected_sets: list[str], actual_sets: list[tuple[Tile, ...]]):
        self.assertCountEqual([tuple(Tile.from_str(s)) for s in expected_sets], actual_sets)

    def assert_result_equal(self, expected: RummiResult, actual: RummiResult):
        self.assertCountEqual(expected.table, actual.table)
        self.assertCountEqual(expected.placed, actual.placed)
        self.assertCountEqual(expected.remaining, actual.remaining)
