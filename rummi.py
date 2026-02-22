from collections import Counter
from itertools import combinations, chain

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar
from pandas import Series

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
        for cs in combinations(COLOURS, 3):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)))

        for cs in combinations(COLOURS, 2):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)) + tuple([JOKER]))

        # Groups of 3 tiles with 2 jokers already covered by runs

        for cs in combinations(COLOURS, 4):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)))

        for cs in combinations(COLOURS, 3):
            groups.append(tuple(Tile(c, v) for c in sorted(cs)) + tuple([JOKER]))

        for cs in combinations(COLOURS, 2):
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
        elif len(Counter([t.colour for t in ts])) > 2:
            # A group, should be in sorted order
            sets_to_process.append(tuple(sorted(ts)))
        else:
            sets_to_process.append(ts)

    if config.joker_mode == JokerMode.LOCKING:
        joker_params = prepare_joker_params(sets_to_process)
    else:
        joker_params = JokerParams({}, [], {}, 0)

    for ts in sets_to_process:
        if not (config.joker_mode == JokerMode.LOCKING and JOKER in ts):
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

    result_sets_matrix, result_tiles_matrix, joker_sets_on_tables_by_k_set = solve_cp_model(
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

    for k_set, joker_sets_on_table in joker_sets_on_tables_by_k_set.items():
        for s in joker_sets_on_table:
            j, n = s[0], s[1]
            for _ in range(n):
                result_table_sets.append(joker_params.tilesets_by_k_set[k_set][j])

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


def prepare_joker_params(tile_sets: list[tuple[Tile, ...]]) -> JokerParams:
    substitution_tile_arrays = []
    tilesets_with_jokers_map = {}
    set_tile_matrices = {}
    total_joker_count = 0
    for tile_set in tile_sets:
        # If there is only one colour it is a run
        colour_count = Counter([t.colour for t in tile_set])

        number_of_jokers = colour_count["J"]

        if number_of_jokers == 0:
            continue

        if len(colour_count) == 2 and len(tile_set) - number_of_jokers == 1:
            raise NotImplementedError("Group with jokers could be a run or a group and IDEK")

        if len(colour_count) == 2:
            group_colours = list(colour_count.keys())
            group_colours.remove("J")
            run_colour = group_colours.pop()

            first_normal_tile_index, first_normal_tile_value = next(
                (i, t.value) for i, t in enumerate(tile_set) if t != JOKER
            )
            first_tile_value = first_normal_tile_value - first_normal_tile_index

            joker_indexes = [i for i, t in enumerate(tile_set) if t == JOKER]
            joker_values = [first_tile_value + i for i in joker_indexes]

            for joker_value in joker_values:
                replacement_tile = Tile(run_colour, joker_value)

                substitution_tile_array = np.zeros((len(TILES)), int)
                substitution_tile_array[replacement_tile.index()] = 1
                substitution_tile_arrays.append(substitution_tile_array)

            for K in non_empty_powerset(range(total_joker_count, total_joker_count + number_of_jokers)):
                tilesets_with_jokers: list[tuple[Tile, ...]] = []

                for start in range(1, 1 + first_tile_value):
                    for end in range(first_tile_value + len(tile_set), 15):
                        run_tiles = []
                        for val in range(start, end):
                            if val in {joker_values[k - number_of_jokers] for k in K}:
                                run_tiles.append(JOKER)
                            else:
                                run_tiles.append(Tile(run_colour, val))
                        tilesets_with_jokers.append(tuple(run_tiles))

                tilesets_with_jokers_map[K] = tilesets_with_jokers
        else:
            # Since jokers in groups can be substituted for either colour there is repetition here:
            # - The substitution tiles for each joker are all the same.
            # - Replacing the first joker and replacing the second etc is also arbitrary, so the tilesets are the same for all K of the same size.
            # This is fine and how the model is set up, possibly inefficient but since groups are only length 3-4 and only 2 jokers, not a big deal.
            group_value = next(t for t in tile_set if t != JOKER).value

            group_colours = list(colour_count.keys())
            group_colours.remove("J")

            missing_colours = [c for c in COLOURS if c not in group_colours]
            replacement_tile_indexes = [Tile(c, group_value).index() for c in missing_colours]
            substitution_tile_array = np.zeros((len(TILES)), int)
            substitution_tile_array[replacement_tile_indexes] = 1

            for i in range(number_of_jokers):
                substitution_tile_arrays.append(substitution_tile_array)

            for K in non_empty_powerset(range(total_joker_count, total_joker_count + number_of_jokers)):
                tilesets_with_jokers = []
                for replaced_tile_colours in combinations(missing_colours, number_of_jokers - len(K)):
                    for added_tile_count in range(5 - len(tile_set)):
                        for added_tile_colours in combinations(set(missing_colours) - set(replaced_tile_colours), added_tile_count):
                            existing_tiles = [Tile(c, group_value) for c in group_colours]
                            jokers = [JOKER] * len(K)
                            replaced_tiles = [Tile(c, group_value) for c in replaced_tile_colours]
                            added_tiles = [Tile(c, group_value) for c in added_tile_colours]
                            possible_set = tuple(sorted(existing_tiles + jokers + replaced_tiles + added_tiles))

                            tilesets_with_jokers.append(possible_set)

                tilesets_with_jokers_map[K] = tilesets_with_jokers

        for K in tilesets_with_jokers_map.keys():
            set_to_index_map = {s: i for i, s in enumerate(tilesets_with_jokers_map[K])}

            set_tile_matrix = np.zeros((len(TILES), len(tilesets_with_jokers_map[K])), int)
            for tiles, j in set_to_index_map.items():
                for t in tiles:
                    set_tile_matrix[t.index(), j] += 1

            set_tile_matrices[K] = set_tile_matrix

        total_joker_count += number_of_jokers

    return JokerParams(set_tile_matrices, substitution_tile_arrays, tilesets_with_jokers_map, total_joker_count)


def solve_cp_model(
        table_tiles_array,
        rack_tiles_array,
        table_sets_array,
        joker_params: JokerParams,
        config: Config
):
    model = cp_model.CpModel()

    placed_sets_var = model.new_int_var_series("placed_sets", pd.Index(range(len(SETS))), 0, 2)
    placed_tiles_var = model.new_int_var_series("placed_tiles", pd.Index(range(len(TILES))), 0, 2)
    unmodified_sets_var = model.new_int_var_series("unmodified_sets", pd.Index(range(len(SETS))), 0, 2)

    joker_placed_sets_vars: dict[tuple[int, ...], Series] = {}
    for k_set, sets in joker_params.tilesets_by_k_set.items():
        name = f"joker_{'_'.join(str(k) for k in k_set)}_placed_sets"
        joker_placed_sets_vars[k_set] = model.new_bool_var_series(name, pd.Index(range(len(sets))))

    # true iff joker_k uses tile_i as its substitution
    subbed_tile: dict[tuple[int, int], IntVar] = {}
    for k, substitution_tile_array in enumerate(joker_params.substitution_tile_arrays):
        for i in np.nonzero(substitution_tile_array)[0]:
            subbed_tile[k, int(i)] = model.new_bool_var(f"joker_{k}_takes_tile_{i}")

    for i in range(len(TILES)):
        # The first constraint ensures that you can only make sets of the tiles that are on your rack or on the
        # table. The right-hand side of this constraint denotes the number of tile i that are already on the table
        # plus that are placed from the rack onto the table. The left-hand side adds up the number of tile i present
        # in the sets that are finally on the table.

        standard_term = sum(SET_TILE_MATRIX[i, j] * placed_sets_var[j] for j in np.nonzero(SET_TILE_MATRIX[i])[0])
        joker_term = sum(
            sum(
                set_tile_matrix[i, l] * joker_placed_sets_vars[k_set][l]
                for l in np.nonzero(set_tile_matrix[i])[0]
            )
            for k_set, set_tile_matrix in joker_params.set_tile_matrices_by_k_set.items()
        )
        model.add(standard_term + joker_term == table_tiles_array[i] + placed_tiles_var[i])

        # The second constraint states that the tiles you can place from your rack onto the table cannot be more than
        # the tiles that are on your rack.
        model.add(placed_tiles_var[i] <= rack_tiles_array[i])

    # Enforce that enough tiles are placed to satisfy the claims
    for i in range(len(TILES)):
        claimers = [subbed_tile[k, i] for k in range(joker_params.joker_count) if (k, i) in subbed_tile]
        if claimers:
            model.add(sum(claimers) <= placed_tiles_var[i])

    for k in range(joker_params.joker_count):
        joker_sets_used = sum(
            sum(joker_placed_sets_vars[k_set]) for k_set in joker_params.tilesets_by_k_set.keys() if k in k_set)
        sub_tile_indexes = np.nonzero(joker_params.substitution_tile_arrays[k])[0]
        tiles_subbing_for_joker = sum(subbed_tile[k, int(i)] for i in sub_tile_indexes)

        # Either use exactly one joker-set OR substitute with exactly one tile
        model.add(joker_sets_used + tiles_subbing_for_joker == 1)

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
        if status == cp_model.INFEASIBLE:
            raise RuntimeError("Infeasible solution")
        raise RuntimeError("Invalid state")

    sets_solution = solver.values(placed_sets_var).to_numpy()
    tiles_solution = solver.values(placed_tiles_var).to_numpy()

    sets_on_table = nonzero_pairs(sets_solution)
    tiles_placed = nonzero_pairs(tiles_solution)

    joker_sets_on_tables_by_k_set: dict[tuple[int, ...], np.ndarray] = {}
    for k_set in joker_params.tilesets_by_k_set.keys():
        joker_sets_solution = solver.values(joker_placed_sets_vars[k_set]).to_numpy()
        joker_sets_on_tables_by_k_set[k_set] = nonzero_pairs(joker_sets_solution)

    return sets_on_table, tiles_placed, joker_sets_on_tables_by_k_set


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


def non_empty_powerset(super_set):
    return chain.from_iterable(combinations(super_set, r) for r in range(1, len(super_set) + 1))
