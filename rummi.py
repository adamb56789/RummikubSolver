import itertools
from dataclasses import dataclass

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


def create_runs():
    runs = []
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
    groups = []

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

# TODO value modes for jokers: at the start it is the value it represents, at the end it is 30. Strategy considerations mean trying to hold into it -> 0 or even negative value.
TILE_VALUES = [t.value for t in TILES]

RUNS = create_runs()
GROUPS = create_groups()
SETS = RUNS + GROUPS

I = len(TILES)
J = len(SETS)

SET_TO_J_MAP = {s: i for i, s in enumerate(SETS)}


def create_sets():
    create_runs()
    sets = np.zeros((I, J), int)

    for tiles, j in SET_TO_J_MAP.items():
        for t in tiles:
            sets[t.get_i(), j] += 1

    return sets


SET_TILE_MATRIX = create_sets()


def solve_cp_model(t, r, w, maximise_value):
    s = SET_TILE_MATRIX
    v = TILE_VALUES

    model = cp_model.CpModel()

    x = model.NewIntVarSeries("x", pd.Index(range(J)), 0, 2)
    y = model.NewIntVarSeries("y", pd.Index(range(I)), 0, 2)
    z = model.NewIntVarSeries("z", pd.Index(range(J)), 0, 2)

    for i in range(I):
        model.Add(sum(s[i, j] * x[j] for j in np.nonzero(s[i])[0]) == t[i] + y[i])

        # Placed tiles (y) cannot be more than rack tiles (r)
        model.Add(y[i] <= r[i])

    # z is the count of sets that are unmodified which we maximize in the optimization step
    # z_j = min(x_j, w_j)
    for j in range(J):
        model.Add(z[j] <= x[j])
        model.Add(z[j] <= w[j])

    if maximise_value:
        model.Maximize((y * v).sum() + (1 / 40) * z.sum())
    else:
        # The paper does not include the change-minimization term in this version, but we add it anyway
        model.Maximize(y.sum() + (1 / 40) * z.sum())

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Invalid state")

    x_solution = solver.Values(x).to_numpy()
    y_solution = solver.Values(y).to_numpy()

    sets_on_table = np.column_stack((x_solution.nonzero()[0], x_solution[x_solution.nonzero()]))
    tiles_placed = np.column_stack((y_solution.nonzero()[0], y_solution[y_solution.nonzero()]))

    return sets_on_table, tiles_placed


def find_best_move(table_sets: list[tuple[Tile, ...]], rack_tiles: list[Tile], maximize_value=False):
    table_tiles_matrix = np.zeros((I,), int)
    for ts in table_sets:
        for tile in ts:
            table_tiles_matrix[tile.get_i()] += 1

    rack_tiles_matrix = np.zeros((I,), int)
    for tile in rack_tiles:
        rack_tiles_matrix[tile.get_i()] += 1

    table_sets_matrix = np.zeros((J,), int)
    for ts in table_sets:
        # Incoming sets might have jokers at the start like "J a2 a3" which is not in our/the paper's sets, since they
        # are made redundant by "a2 a3 J". In such cases we automatically rearrange them to our preferred format, since that is allowed.
        if ts not in SET_TO_J_MAP:
            if ts[0].colour != "J":
                raise RuntimeError("Invalid set")
            ts = ts[1:] + tuple([ts[0]])

            # Could be 2 jokers at the front so do it again
            if ts[0].colour == "J":
                ts = ts[1:] + tuple([ts[0]])

        table_sets_matrix[SET_TO_J_MAP[ts]] += 1

    result_sets_matrix, result_tiles_matrix = solve_cp_model(
        table_tiles_matrix,
        rack_tiles_matrix,
        table_sets_matrix,
        maximize_value
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

    return sorted(result_table_sets), sorted(placed_tiles), sorted(remaining_tiles)
