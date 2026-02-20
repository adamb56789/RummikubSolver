import itertools
from collections import Counter

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from structs import Tile, MaximizeMode, JokerMode, Config, RummiResult, COLOURS, JokerParams


def create_runs():
    runs: list[tuple[Tile, ...]] = []
    for c in COLOURS:
        for run_length in [3, 4, 5]:

            # No Jokers

            for i in range(1, 15 - run_length):
                runs.append(tuple(Tile(c, j) for j in range(i, i + run_length)))

            # One joker

            # Place jokers starting from the 2nd position, since using the first overlaps with another
            # For example b2 b3 J is equivalent to J b2 b3
            for i in range(1, 15 - run_length):
                for joker_pos in range(i + 1, i + run_length):
                    runs.append(tuple(
                        JOKER if j == joker_pos else Tile(c, j)
                        for j in range(i, i + run_length)
                    ))

            # Special case to put the joker in the 1st position for the 11, 12, 13 run
            runs.append(tuple(
                JOKER if j == 14 - run_length else Tile(c, j)
                for j in range(14 - run_length, 14)
            ))

            # Two jokers

            for i in range(1, 15 - run_length):
                for joker_pos_1 in range(i + 1, i + run_length - 1):
                    for joker_pos_2 in range(joker_pos_1 + 1, i + run_length):
                        runs.append(tuple(
                            JOKER if j == joker_pos_1 or j == joker_pos_2 else Tile(c, j)
                            for j in range(i, i + run_length)
                        ))

            joker_pos_1 = 14 - run_length
            for joker_pos_2 in range(joker_pos_1 + 1, 14):
                runs.append(tuple(
                    JOKER if j == joker_pos_1 or j == joker_pos_2 else Tile(c, j) for
                    j in range(14 - run_length, 14)
                ))
    return runs


def create_groups():
    groups: list[tuple[Tile, ...]] = []

    for v in range(1, 14):
        for cs in itertools.combinations(COLOURS, 3):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)))

        for cs in itertools.combinations(COLOURS, 2):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)) + tuple([JOKER]))

        # Groups of 3 tiles with 2 jokers already covered by runs

        for cs in itertools.combinations(COLOURS, 4):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)))

        for cs in itertools.combinations(COLOURS, 3):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)) + tuple([JOKER]))

        for cs in itertools.combinations(COLOURS, 2):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)) + tuple([JOKER] * 2))

    return groups


JOKER = Tile("J", 0)
TILES = [Tile(c, v) for c in COLOURS for v in range(1, 14)] + [JOKER]
TILE_VALUES = np.array([t.value for t in TILES])

RUNS = create_runs()
GROUPS = create_groups()
SETS = RUNS + GROUPS

SET_TO_INDEX_MAP = {s: i for i, s in enumerate(SETS)}
SET_TILE_MATRIX = np.zeros((len(TILES), len(SETS)), int)

for tiles, j in SET_TO_INDEX_MAP.items():
    for t in tiles:
        SET_TILE_MATRIX[t.index(), j] += 1


def find_best_move(
        table_sets: list[tuple[Tile, ...]],
        rack_tiles: list[Tile],
        config: Config
) -> RummiResult:
    table_tiles_array = np.zeros((len(TILES),), int)

    rack_tiles_array = np.zeros((len(TILES),), int)
    for tile in rack_tiles:
        rack_tiles_array[tile.index()] += 1

    table_sets_array = np.zeros((len(SETS),), int)

    # Incoming sets can be too long, in which case we just split them up,
    # unless it has jokers which need to be locked but that is handled separately.
    # This kinda breaks the manipulation minimization term but that's just a bonus.
    sets_to_process = []
    for ts in table_sets:
        if len(ts) > 5 and (config.joker_mode == JokerMode.FREE or JOKER not in ts):
            sets_to_process.extend(split_tuple(ts))
        elif len(Counter([t.colour for t in ts])) > 2 or Counter([t.colour for t in ts]).get("J", 0) == 2:
            # A group, should be in sorted order
            sets_to_process.append(tuple(sorted(ts)))
        else:
            sets_to_process.append(ts)

    joker_params: list[JokerParams] = []
    for ts in sets_to_process:
        if config.joker_mode == JokerMode.LOCKING and JOKER in ts:
            joker_params.append(prepare_joker_params(ts))
        else:
            # Incoming sets might have jokers at the start like "J a2 a3" which is not in our/the paper's sets, since they
            # are made redundant by "a2 a3 J". In such cases we automatically rearrange them to our preferred format, since that is allowed.
            if ts not in SET_TO_INDEX_MAP:
                if ts[0].colour != "J":
                    raise RuntimeError("Invalid set")
                ts = ts[1:] + tuple([ts[0]])

                # Could be 2 jokers at the front so do it again
                if ts[0].colour == "J":
                    ts = ts[1:] + tuple([ts[0]])

            table_sets_array[SET_TO_INDEX_MAP[ts]] += 1

        for tile in ts:
            table_tiles_array[tile.index()] += 1

    result_sets_matrix, result_tiles_matrix, joker_sets_on_table_list = solve_cp_model(
        table_tiles_array,
        rack_tiles_array,
        table_sets_array,
        joker_params,
        config,
    )

    result_table_sets = []
    for s in result_sets_matrix:
        j, n = s[0], s[1]
        for _ in range(n):
            result_table_sets.append(SETS[j])

    for i, params in enumerate(joker_params):
        for s in joker_sets_on_table_list[i]:
            j, n = s[0], s[1]
            for _ in range(n):
                result_table_sets.append(params.sets[j])

    placed_tiles = []
    for t in result_tiles_matrix:
        i, n = t[0], t[1]
        for _ in range(n):
            placed_tiles.append(TILES[i])

    remaining_tiles = rack_tiles.copy()
    for tile in placed_tiles:
        remaining_tiles.remove(tile)

    result = RummiResult(sorted(result_table_sets), sorted(placed_tiles), sorted(remaining_tiles))

    return result


def prepare_joker_params(tile_set: tuple[Tile, ...]) -> JokerParams:
    # If there is only one colour it is a run
    colour_count = Counter([t.colour for t in tile_set])

    if colour_count["J"] == 2:
        raise NotImplementedError("IDEK")

    # 2 is one colour + joker
    if len(colour_count) == 2:
        group_colours = list(colour_count.keys())
        group_colours.remove("J")
        run_colour = group_colours.pop()

        joker_index = tile_set.index(JOKER)
        offset = -1 if joker_index > 0 else 1
        joker_value = tile_set[joker_index + offset].value - offset

        replacement_tile = Tile(run_colour, joker_value)

        substitution_tile_array = np.zeros((len(TILES)), int)
        substitution_tile_array[replacement_tile.index()] = 1

        sets: list[tuple[Tile, ...]] = []

        for start in range(1, 1 + joker_value - joker_index):
            for end in range(joker_value - joker_index + len(tile_set), 15):
                run_tiles = []
                for val in range(start, end):
                    if val == joker_value:
                        run_tiles.append(JOKER)
                    else:
                        run_tiles.append(Tile(run_colour, val))
                sets.append(tuple(run_tiles))
    else:
        group_value = next(t for t in tile_set if t != JOKER).value

        group_colours = list(colour_count.keys())
        group_colours.remove("J")

        replacement_colours = [c for c in COLOURS if c not in group_colours]
        replacement_tile_indexes = [Tile(c, group_value).index() for c in replacement_colours]

        substitution_tile_array = np.zeros((len(TILES)), int)
        substitution_tile_array[replacement_tile_indexes] = 1

        sets: list[tuple[Tile, ...]] = [tile_set]
        if len(tile_set) == 3:
            for colour in replacement_colours:
                sets.append(tuple(sorted(list(tile_set) + [Tile(colour, group_value)])))

    set_to_index_map = {s: i for i, s in enumerate(sets)}

    table_sets_array = np.zeros((len(sets)), int)
    table_sets_array[set_to_index_map[tile_set]] = 1

    set_tile_matrix = np.zeros((len(TILES), len(sets)), int)
    for tiles, j in set_to_index_map.items():
        for t in tiles:
            set_tile_matrix[t.index(), j] += 1

    return JokerParams(set_to_index_map, table_sets_array, set_tile_matrix, substitution_tile_array, sets)


def solve_cp_model(table_tiles_array, rack_tiles_array, table_sets_array, joker_params: list[JokerParams],
                   config: Config):
    model = cp_model.CpModel()

    placed_sets_var = model.new_int_var_series("placed_sets", pd.Index(range(len(SETS))), 0, 2)
    placed_tiles_var = model.new_int_var_series("placed_tiles", pd.Index(range(len(TILES))), 0, 2)
    unmodified_sets_var = model.new_int_var_series("unmodified_sets", pd.Index(range(len(SETS))), 0, 2)

    joker_placed_sets_vars = []
    for i, params in enumerate(joker_params):
        joker_placed_sets_vars.append(
            model.new_int_var_series(f"joker_placed_sets_{i}", pd.Index(range(len(params.sets))), 0, 1))

    for i in range(len(TILES)):
        # The first constraint ensures that you can only make sets of the tiles that are on your rack or on the
        # table. The right-hand side of this constraint denotes the number of tile i that are already on the table
        # plus that are placed from the rack onto the table. The left-hand side adds up the number of tile i present
        # in the sets that are finally on the table.

        standard_term = sum(SET_TILE_MATRIX[i, j] * placed_sets_var[j] for j in np.nonzero(SET_TILE_MATRIX[i])[0])
        joker_term = sum(
            sum(
                params.set_tile_matrix[i, j] * joker_placed_sets_vars[joker_i][j]
                for j in np.nonzero(params.set_tile_matrix[i])[0]
            )
            for joker_i, params in enumerate(joker_params)
        )
        model.add(standard_term + joker_term == table_tiles_array[i] + placed_tiles_var[i])

        # The second constraint states that the tiles you can place from your rack onto the table cannot be more than
        # the tiles that are on your rack.
        model.add(placed_tiles_var[i] <= rack_tiles_array[i])

    for joker_i, params in enumerate(joker_params):
        # If the substitution tiles have been placed then joker placed sets may be 0, otherwise it must be 1
        subbed_tile_bools = []
        for i in range(len(TILES)):
            if params.substitution_tile_array[i]:
                b = model.new_bool_var(f"subbed_tile_{joker_i}_{i}")
                subbed_tile_bools.append(b)
                model.add(placed_tiles_var[i] >= 1).only_enforce_if(b)
                model.add(placed_tiles_var[i] == 0).only_enforce_if(b.negated())

        joker_is_subbed = model.new_bool_var(f"joker_is_subbed_{joker_i}")
        model.add_max_equality(joker_is_subbed,
                               subbed_tile_bools)  # Effectively joker_is_subbed = or(subbed_tile_bools)

        joker_sets_used = sum(joker_placed_sets_vars[joker_i])

        model.add(joker_sets_used == 0).only_enforce_if(joker_is_subbed)
        model.add(joker_sets_used == 1).only_enforce_if(joker_is_subbed.negated())
        model.add(joker_sets_used < 2)

    for j in range(len(SETS)):
        # Unmodified sets are maximized in the optimization step, so it chooses the highest value <= both the table
        # set and placed set, in other words unmodified_sets_var[j] = min(table_sets_array[j], placed_sets_var[j])
        model.add_min_equality(unmodified_sets_var[j], [table_sets_array[j], placed_sets_var[j]])

    if config.maximize_mode == MaximizeMode.VALUE_PLACED:
        TILE_VALUES[JOKER.index()] = config.joker_value
        model.maximize((placed_tiles_var * TILE_VALUES).sum() + config.rearrange_value * unmodified_sets_var.sum())
    else:
        # The paper does not include the change-minimization term in this version, but we add it anyway
        model.maximize(placed_tiles_var.sum() + config.rearrange_value * unmodified_sets_var.sum())

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    # TODO tune this, for some reason it is faster on 1 when running the test suite
    solver.parameters.num_workers = 1

    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Invalid state")

    sets_solution = solver.values(placed_sets_var).to_numpy()
    tiles_solution = solver.values(placed_tiles_var).to_numpy()

    sets_on_table = nonzero_pairs(sets_solution)
    tiles_placed = nonzero_pairs(tiles_solution)

    joker_sets_on_table_list = []
    for i, params in enumerate(joker_params):
        joker_sets_solution = solver.values(joker_placed_sets_vars[i]).to_numpy()
        joker_sets_on_table_list.append(nonzero_pairs(joker_sets_solution))

    return sets_on_table, tiles_placed, joker_sets_on_table_list


def find_best_move_strings(table_set_strings: list[str], rack_string: str, config: Config) -> RummiResult:
    return find_best_move(
        [tuple(Tile.from_str(s)) for s in table_set_strings],
        Tile.from_str(rack_string),
        config
    )


def nonzero_pairs(arr):
    idx = np.where(arr > 0)[0]
    return np.column_stack((idx, arr[idx]))


def split_tuple(t):
    n = len(t)
    if n < 6:
        raise ValueError("Tuple must have length at least 6")

    result = []
    i = 0

    while n > 0:
        if n == 6:
            sizes = (3, 3)
        elif n == 7:
            sizes = (3, 4)
        elif n == 8:
            sizes = (4, 4)
        elif n == 9:
            sizes = (4, 5)
        else:
            sizes = (5,)

        for s in sizes:
            result.append(t[i:i + s])
            i += s
            n -= s

    return result
