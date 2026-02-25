from collections import defaultdict
from itertools import combinations, chain

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar
from pandas import Series

from rummi_cube.structs import Tile, MaximizeMode, JokerMode, Config, RummiResult, COLOURS, TilesetModelParams, Tileset, \
    JOKER
from rummi_cube.tileset_generation import generate_all_sets

TILES = [Tile(c, v) for c in COLOURS for v in range(1, 14)] + [JOKER]
TILE_VALUES = np.array([t.value for t in TILES])

SETS = generate_all_sets()

SET_TO_INDEX_MAP: dict[Tileset, int] = {s: i for i, s in enumerate(SETS)}
SET_TILE_MATRIX = np.zeros((len(TILES), len(SETS)), int)

for tiles, j in SET_TO_INDEX_MAP.items():
    for t in tiles:
        SET_TILE_MATRIX[t.index(), j] += 1


def find_best_move(
        table_sets: list[Tileset],
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
    sets_to_process: list[Tileset] = []
    for ts in table_sets:
        if len(ts) > 5 and (config.joker_mode == JokerMode.FREE or not ts.contains_joker):
            sets_to_process.extend(ts.split_tileset())
        else:
            sets_to_process.append(ts)

    if config.joker_mode == JokerMode.LOCKING:
        tileset_model_params = prepare_joker_locking_tileset_params(sets_to_process)
    else:
        tileset_model_params = TilesetModelParams({}, [], {}, 0)

    # Normal case of no jokers
    tileset_model_params.tilesets_by_k_set[()] = SETS
    tileset_model_params.set_tile_matrices_by_k_set[()] = SET_TILE_MATRIX

    for ts in sets_to_process:
        if not (config.joker_mode == JokerMode.LOCKING and ts.contains_joker):
            table_sets_array[SET_TO_INDEX_MAP[ts]] += 1

        for tile in ts:
            table_tiles_array[tile.index()] += 1

    sets_on_tables_by_k_set, result_tiles_matrix = solve_cp_model(
        table_tiles_array,
        rack_tiles_array,
        table_sets_array,
        tileset_model_params,
        config,
    )

    result_table_sets = []
    for k_set, joker_sets_on_table in sets_on_tables_by_k_set.items():
        for s in joker_sets_on_table:
            j, n = s[0], s[1]
            for _ in range(n):
                result_table_sets.append(tileset_model_params.tilesets_by_k_set[k_set][j])

    placed_tiles = []
    for t in result_tiles_matrix:
        i, n = t[0], t[1]
        for _ in range(n):
            placed_tiles.append(TILES[i])

    remaining_tiles = rack_tiles.copy()
    for tile in placed_tiles:
        remaining_tiles.remove(tile)

    return RummiResult(sorted(result_table_sets), sorted(placed_tiles), sorted(remaining_tiles))


def prepare_joker_locking_tileset_params(tilesets: list[Tileset]) -> TilesetModelParams:
    tilesets_by_k_set: dict[tuple, list[Tileset]] = defaultdict(list)
    substitution_tiles_by_k: list[list[int]] = []

    seen_joker_count = 0
    for tileset in tilesets:
        if tileset.number_of_jokers == 0:
            continue

        for i in range(tileset.number_of_jokers):
            substitution_tiles_by_k.append([])

        if tileset.is_run:
            joker_indexes = [i for i, t in enumerate(tileset) if t.is_joker()]
            joker_values = [tileset.run_first_tile_value + i for i in joker_indexes]

            for i, joker_value in enumerate(joker_values):
                substitution_tiles_by_k[seen_joker_count + i].append(Tile(tileset.run_colour, joker_value).index())

            for K in non_empty_powerset(range(seen_joker_count, seen_joker_count + tileset.number_of_jokers)):
                tilesets_with_jokers: list[Tileset] = []

                for start in range(1, 1 + tileset.run_first_tile_value):
                    for end in range(tileset.run_first_tile_value + len(tileset), 15):
                        run_tiles = []
                        for val in range(start, end):
                            if val in {joker_values[k - tileset.number_of_jokers] for k in K}:
                                run_tiles.append(JOKER)
                            else:
                                run_tiles.append(Tile(tileset.run_colour, val))
                        tilesets_with_jokers.append(Tileset(run_tiles))

                tilesets_by_k_set[K].extend(tilesets_with_jokers)

        if tileset.is_group:
            # Since jokers in groups can be substituted for either colour there is repetition here:
            # - The substitution tiles for each joker are all the same.
            # - Replacing the first joker and replacing the second etc is also arbitrary, so the tilesets are the same for all K of the same size.
            # This is fine and how the model is set up, possibly inefficient but since groups are only length 3-4 and only 2 jokers, not a big deal.
            missing_colours = [c for c in COLOURS if c not in tileset.colours]
            for i in range(tileset.number_of_jokers):
                for c in missing_colours:
                    substitution_tiles_by_k[seen_joker_count + i].append(Tile(c, tileset.group_value).index())

            for K in non_empty_powerset(range(seen_joker_count, seen_joker_count + tileset.number_of_jokers)):
                tilesets_with_jokers: list[Tileset] = []
                for replaced_tile_colours in combinations(missing_colours, tileset.number_of_jokers - len(K)):
                    for added_tile_count in range(5 - len(tileset)):
                        for added_tile_colours in combinations(set(missing_colours) - set(replaced_tile_colours),
                                                               added_tile_count):
                            existing_tiles = [Tile(c, tileset.group_value) for c in tileset.colours]
                            jokers = [JOKER] * len(K)
                            replaced_tiles = [Tile(c, tileset.group_value) for c in replaced_tile_colours]
                            added_tiles = [Tile(c, tileset.group_value) for c in added_tile_colours]

                            # Handle the special case where the order of an unambiguous "2-1" tileset such as (J a1 J) matters.
                            if len(existing_tiles + replaced_tiles + added_tiles) == 1:
                                new_tileset = tileset
                            else:
                                new_tileset = Tileset(existing_tiles + jokers + replaced_tiles + added_tiles)

                            tilesets_with_jokers.append(new_tileset)

                tilesets_by_k_set[K].extend(tilesets_with_jokers)

        seen_joker_count += tileset.number_of_jokers

    set_tile_matrices = {}
    for K in tilesets_by_k_set.keys():
        set_tile_matrix = np.zeros((len(TILES), len(tilesets_by_k_set[K])), int)
        for j, tiles in enumerate(tilesets_by_k_set[K]):
            for t in tiles:
                set_tile_matrix[t.index(), j] += 1

        set_tile_matrices[K] = set_tile_matrix

    return TilesetModelParams(set_tile_matrices, substitution_tiles_by_k, tilesets_by_k_set, seen_joker_count)


def solve_cp_model(
        table_tiles_array,
        rack_tiles_array,
        table_sets_array,
        tileset_params: TilesetModelParams,
        config: Config
):
    model = cp_model.CpModel()

    placed_tiles_var = model.new_int_var_series("y", pd.Index(range(len(TILES))), 0, 2)
    unmodified_sets_var = model.new_int_var_series("w", pd.Index(range(len(SETS))), 0, 2)

    joker_placed_sets_vars: dict[tuple[int, ...], Series] = {}
    for k_set, sets in tileset_params.tilesets_by_k_set.items():
        name = f"x_{'_'.join(str(k) for k in k_set)}"
        joker_placed_sets_vars[k_set] = model.new_int_var_series(name, pd.Index(range(len(sets))), 0, 2)

    # true iff joker_k uses tile_i as its substitution
    joker_k_takes_tile_i: dict[tuple[int, int], IntVar] = {}
    for k, substitution_tiles in enumerate(tileset_params.substitution_tiles_by_k):
        for i in substitution_tiles:
            joker_k_takes_tile_i[k, int(i)] = model.new_bool_var(f"z_{k}_{i}")

    for i in range(len(TILES)):
        # The first constraint ensures that you can only make sets of the tiles that are on your rack or on the
        # table. The right-hand side of this constraint denotes the number of tile i that are already on the table
        # plus that are placed from the rack onto the table. The left-hand side adds up the number of tile i present
        # in the sets that are finally on the table.

        count_of_tiles_in_sets = sum(
            sum(
                set_tile_matrix[i, l] * joker_placed_sets_vars[k_set][l]
                for l in np.nonzero(set_tile_matrix[i])[0]
            )
            for k_set, set_tile_matrix in tileset_params.set_tile_matrices_by_k_set.items()
        )
        model.add(count_of_tiles_in_sets == table_tiles_array[i] + placed_tiles_var[i])

        # The second constraint states that the tiles you can place from your rack onto the table cannot be more than
        # the tiles that are on your rack.
        model.add(placed_tiles_var[i] <= rack_tiles_array[i])

    # Enforce that enough tiles are placed to satisfy the claims
    for i in range(len(TILES)):
        claimers = [joker_k_takes_tile_i[k, i] for k in range(tileset_params.joker_count) if
                    (k, i) in joker_k_takes_tile_i]
        if claimers:
            model.add(sum(claimers) <= placed_tiles_var[i])

    for k in range(tileset_params.joker_count):
        joker_sets_used = sum(
            sum(joker_placed_sets_vars[k_set]) for k_set in tileset_params.tilesets_by_k_set.keys() if k in k_set)
        tiles_subbing_for_joker = sum(
            joker_k_takes_tile_i[k, int(i)] for i in tileset_params.substitution_tiles_by_k[k])

        # Either use exactly one joker-set OR substitute with exactly one tile
        model.add(joker_sets_used + tiles_subbing_for_joker == 1)

    for j in range(len(SETS)):
        # Unmodified sets are maximized in the optimization step, so it chooses the highest value <= both the table
        # set and placed set, in other words unmodified_sets_var[j] = min(table_sets_array[j], placed_sets_var[j])
        model.add_min_equality(unmodified_sets_var[j], [table_sets_array[j], joker_placed_sets_vars[()][j]])

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

    tiles_solution = solver.values(placed_tiles_var).to_numpy()

    tiles_placed = nonzero_pairs(tiles_solution)

    sets_on_tables_by_k_set: dict[tuple[int, ...], np.ndarray] = {}
    for k_set in tileset_params.tilesets_by_k_set.keys():
        sets_on_tables_by_k_set[k_set] = nonzero_pairs(solver.values(joker_placed_sets_vars[k_set]))

    return sets_on_tables_by_k_set, tiles_placed


def find_best_move_strings(table_set_strings: list[str], rack_string: str, config: Config) -> RummiResult:
    return find_best_move(
        [Tileset.from_str(s) for s in table_set_strings],
        Tile.from_str(rack_string),
        config
    )


def nonzero_pairs(arr):
    idx = np.where(arr > 0)[0]
    return np.column_stack((idx, arr[idx]))


def non_empty_powerset(super_set):
    return chain.from_iterable(combinations(super_set, r) for r in range(1, len(super_set) + 1))
