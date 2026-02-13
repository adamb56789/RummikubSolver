import itertools
from dataclasses import dataclass
from enum import Enum

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
    TILES_PLACED = 1
    VALUE_PLACED = 2


class JokerMode(Enum):
    LOCKING = 1
    FREE = 2


TILES = [Tile(c, v) for c in COLOURS for v in range(1, 14)] + [Tile("J", 0)]
TILE_VALUES = np.array([t.value for t in TILES])
JOKER_INDEX = len(TILES) - 1


class Rummi:
    def __init__(self, joker_mode: JokerMode):
        self.joker_mode = joker_mode
        self.sets = self._create_runs() + self._create_groups()
        self.set_to_index_map = {s: i for i, s in enumerate(self.sets)}

        self.set_tile_matrix = np.zeros((len(TILES), len(self.sets)), int)

        for tiles, j in self.set_to_index_map.items():
            for t in tiles:
                self.set_tile_matrix[t.get_i(), j] += 1

    def _create_runs(self):
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

    def _create_groups(self):
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

    def _solve_cp_model(self, table_tiles_matrix, rack_tiles_matrix, table_sets_matrix, maximize_mode, joker_value):
        set_tile_matrix = self.set_tile_matrix

        tile_values = TILE_VALUES
        tile_values[JOKER_INDEX] = joker_value

        model = cp_model.CpModel()

        x = model.NewIntVarSeries("x", pd.Index(range(len(self.sets))), 0, 2)
        y = model.NewIntVarSeries("y", pd.Index(range(len(TILES))), 0, 2)
        z = model.NewIntVarSeries("z", pd.Index(range(len(self.sets))), 0, 2)

        for i in range(len(TILES)):
            model.Add(
                sum(set_tile_matrix[i, j] * x[j] for j in np.nonzero(set_tile_matrix[i])[0]) == table_tiles_matrix[i] +
                y[i])

            # Placed tiles (y) cannot be more than rack tiles (r)
            model.Add(y[i] <= rack_tiles_matrix[i])

        # z is the count of sets that are unmodified which we maximize in the optimization step
        # z_j = min(x_j, w_j)
        for j in range(len(self.sets)):
            model.Add(z[j] <= x[j])
            model.Add(z[j] <= table_sets_matrix[j])

        if maximize_mode == MaximizeMode.VALUE_PLACED:
            model.Maximize((y * tile_values).sum() + (1 / 40) * z.sum())
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

    def find_best_move_with_parameters(
            self,
            table_sets: list[tuple[Tile, ...]],
            rack_tiles: list[Tile],
            maximize_mode=MaximizeMode.TILES_PLACED,
            joker_value=0
    ):
        table_tiles_matrix = np.zeros((len(TILES),), int)
        for ts in table_sets:
            for tile in ts:
                table_tiles_matrix[tile.get_i()] += 1

        rack_tiles_matrix = np.zeros((len(TILES),), int)
        for tile in rack_tiles:
            rack_tiles_matrix[tile.get_i()] += 1

        table_sets_matrix = np.zeros((len(self.sets),), int)
        for ts in table_sets:
            # Incoming sets might have jokers at the start like "J a2 a3" which is not in our/the paper's sets, since they
            # are made redundant by "a2 a3 J". In such cases we automatically rearrange them to our preferred format, since that is allowed.
            if ts not in self.set_to_index_map:
                if ts[0].colour != "J":
                    raise RuntimeError("Invalid set")
                ts = ts[1:] + tuple([ts[0]])

                # Could be 2 jokers at the front so do it again
                if ts[0].colour == "J":
                    ts = ts[1:] + tuple([ts[0]])

            table_sets_matrix[self.set_to_index_map[ts]] += 1

        result_sets_matrix, result_tiles_matrix = self._solve_cp_model(
            table_tiles_matrix,
            rack_tiles_matrix,
            table_sets_matrix,
            maximize_mode,
            joker_value
        )

        result_table_sets = []
        for s in result_sets_matrix:
            j, n = s[0], s[1]
            for _ in range(n):
                result_table_sets.append(self.sets[j])

        placed_tiles = []
        for t in result_tiles_matrix:
            i, n = t[0], t[1]
            for _ in range(n):
                placed_tiles.append(TILES[i])

        remaining_tiles = rack_tiles.copy()
        for tile in placed_tiles:
            remaining_tiles.remove(tile)

        return sorted(result_table_sets), sorted(placed_tiles), sorted(remaining_tiles)
