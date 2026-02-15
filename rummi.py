import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

COLOURS = "brya"


@dataclass(frozen=True)
class Tile:
    colour: str
    value: int

    def __repr__(self):
        return f'{self.colour}{self.value}'

    def __lt__(self, other):
        if self.colour == other.colour:
            return self.value < other.value
        else:
            return self.colour < other.colour

    def get_i(self):
        return 52 if self.colour == "J" else 13 * COLOURS.index(self.colour) + self.value - 1

    @staticmethod
    def from_str(s: str) -> list['Tile']:
        if not s:
            return []
        return [
            Tile(tile[0], int(tile[1:]) if len(tile) > 1 else 0)
            for tile in s.split(" ")
        ]


class MaximizeMode(Enum):
    TILES_PLACED = "tiles_placed"
    VALUE_PLACED = "value_placed"


class JokerMode(Enum):
    LOCKING = "locking"
    FREE = "free"


@dataclass
class Config:
    joker_mode: JokerMode
    maximize_mode: MaximizeMode
    joker_value: Optional[int] = None
    rearrange_value: float = 1 / 40

    def __post_init__(self):
        if self.maximize_mode == MaximizeMode.VALUE_PLACED and self.joker_value is None:
            raise Exception("Joker value must be set")


@dataclass
class RummiResult:
    table: list[tuple[Tile, ...]]
    placed: list[Tile]
    remaining: list[Tile]


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
                        Tile("J", 0) if j == joker_pos else Tile(c, j)
                        for j in range(i, i + run_length)
                    ))

            # Special case to put the joker in the 1st position for the 11, 12, 13 run
            runs.append(tuple(
                Tile("J", 0) if j == 14 - run_length else Tile(c, j)
                for j in range(14 - run_length, 14)
            ))

            # Two jokers

            for i in range(1, 15 - run_length):
                for joker_pos_1 in range(i + 1, i + run_length - 1):
                    for joker_pos_2 in range(joker_pos_1 + 1, i + run_length):
                        runs.append(tuple(
                            Tile("J", 0) if j == joker_pos_1 or j == joker_pos_2 else Tile(c, j)
                            for j in range(i, i + run_length)
                        ))

            joker_pos_1 = 14 - run_length
            for joker_pos_2 in range(joker_pos_1 + 1, 14):
                runs.append(tuple(
                    Tile("J", 0) if j == joker_pos_1 or j == joker_pos_2 else Tile(c, j) for
                    j in range(14 - run_length, 14)
                ))
    return runs


def create_groups():
    groups: list[tuple[Tile, ...]] = []

    for v in range(1, 14):
        for cs in itertools.combinations(COLOURS, 3):
            groups.append(tuple(Tile(c, v) for c in cs))

        for cs in itertools.combinations(COLOURS, 2):
            groups.append(tuple(Tile(c, v) for c in cs) + tuple([Tile("J", v)]))

        # Groups of 3 tiles with 2 jokers already covered by runs

        for cs in itertools.combinations(COLOURS, 4):
            groups.append(tuple(Tile(c, v) for c in cs))

        for cs in itertools.combinations(COLOURS, 3):
            groups.append(tuple(Tile(c, v) for c in cs) + tuple([Tile("J", v)]))

        for cs in itertools.combinations(COLOURS, 2):
            groups.append(tuple(Tile(c, v) for c in cs) + tuple([Tile("J", v)] * 2))

    return groups


TILES = [Tile(c, v) for c in COLOURS for v in range(1, 14)] + [Tile("J", 0)]
TILE_VALUES = np.array([t.value for t in TILES])
JOKER_INDEX = len(TILES) - 1

RUNS = create_runs()
GROUPS = create_groups()
SETS = RUNS + GROUPS

SET_TO_INDEX_MAP = {s: i for i, s in enumerate(SETS)}
SET_TILE_MATRIX = np.zeros((len(TILES), len(SETS)), int)

for tiles, j in SET_TO_INDEX_MAP.items():
    for t in tiles:
        SET_TILE_MATRIX[t.get_i(), j] += 1


def solve_cp_model(table_tiles_matrix, rack_tiles_matrix, table_sets_matrix, config: Config):
    model = cp_model.CpModel()

    placed_sets_var = model.new_int_var_series("x", pd.Index(range(len(SETS))), 0, 2)
    placed_tiles_var = model.new_int_var_series("y", pd.Index(range(len(TILES))), 0, 2)
    unmodified_sets_var = model.new_int_var_series("z", pd.Index(range(len(SETS))), 0, 2)

    for i in range(len(TILES)):
        model.add(
            sum(SET_TILE_MATRIX[i, j] * placed_sets_var[j] for j in np.nonzero(SET_TILE_MATRIX[i])[0]) ==
            table_tiles_matrix[i] +
            placed_tiles_var[i])

        # Placed tiles cannot be more than rack tiles
        model.add(placed_tiles_var[i] <= rack_tiles_matrix[i])

    for j in range(len(SETS)):
        # Unmodified sets are maximized in the optimization step, so it chooses the highest value <= both the
        # table set and placed set, in other words unmodified_sets_var[j] = min(table_sets_matrix[j], placed_sets_var[j])
        model.add_min_equality(unmodified_sets_var[j], [table_sets_matrix[j], placed_sets_var[j]])

    if config.maximize_mode == MaximizeMode.VALUE_PLACED:
        TILE_VALUES[JOKER_INDEX] = config.joker_value
        model.maximize((placed_tiles_var * TILE_VALUES).sum() + config.rearrange_value * unmodified_sets_var.sum())
    else:
        # The paper does not include the change-minimization term in this version, but we add it anyway
        model.maximize(placed_tiles_var.sum() + config.rearrange_value * unmodified_sets_var.sum())

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0

    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Invalid state")

    sets_solution = solver.values(placed_sets_var).to_numpy()
    tiles_solution = solver.values(placed_tiles_var).to_numpy()

    sets_on_table = np.column_stack((sets_solution.nonzero()[0], sets_solution[sets_solution.nonzero()]))
    tiles_placed = np.column_stack((tiles_solution.nonzero()[0], tiles_solution[tiles_solution.nonzero()]))

    return sets_on_table, tiles_placed


def find_best_move(
        table_sets: list[tuple[Tile, ...]],
        rack_tiles: list[Tile],
        config: Config
) -> RummiResult:
    table_tiles_matrix = np.zeros((len(TILES),), int)
    for ts in table_sets:
        for tile in ts:
            table_tiles_matrix[tile.get_i()] += 1

    rack_tiles_matrix = np.zeros((len(TILES),), int)
    for tile in rack_tiles:
        rack_tiles_matrix[tile.get_i()] += 1

    table_sets_matrix = np.zeros((len(SETS),), int)
    for ts in table_sets:
        # Incoming sets might have jokers at the start like "J a2 a3" which is not in our/the paper's sets, since they
        # are made redundant by "a2 a3 J". In such cases we automatically rearrange them to our preferred format, since that is allowed.
        if ts not in SET_TO_INDEX_MAP:
            if ts[0].colour != "J":
                raise RuntimeError("Invalid set")
            ts = ts[1:] + tuple([ts[0]])

            # Could be 2 jokers at the front so do it again
            if ts[0].colour == "J":
                ts = ts[1:] + tuple([ts[0]])

        table_sets_matrix[SET_TO_INDEX_MAP[ts]] += 1

    result_sets_matrix, result_tiles_matrix = solve_cp_model(
        table_tiles_matrix,
        rack_tiles_matrix,
        table_sets_matrix,
        config,
    )

    result_table_sets = []
    for s in result_sets_matrix:
        j, n = s[0], s[1]
        for _ in range(n):
            result_table_sets.append(SETS[j])

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


def find_best_move_strings(table_set_strings: list[str], rack_string: str, config: Config) -> RummiResult:
    return find_best_move(
        [tuple(Tile.from_str(s)) for s in table_set_strings],
        Tile.from_str(rack_string),
        config
    )
