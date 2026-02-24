from itertools import combinations

from rummi_cube.structs import Tileset, COLOURS, Tile

JOKER = Tile("J", 0)


def generate_all_runs():
    runs: list[Tileset] = []
    for c in COLOURS:
        for run_length in [3, 4, 5]:
            for number_of_jokers in [0, 1, 2]:
                for run_start in range(1, 15 - run_length):
                    for joker_values in combinations(range(run_start, run_start + run_length), number_of_jokers):
                        tiles = [
                            JOKER if value in joker_values else Tile(c, value)
                            for value in range(run_start, run_start + run_length)
                        ]
                        runs.append(Tileset(tiles))
    return runs


def generate_all_groups():
    groups: list[Tileset] = []

    for value in range(1, 14):
        for group_size in [3, 4]:
            for number_of_jokers in [0, 1, 2]:
                for colours in combinations(COLOURS, group_size - number_of_jokers):
                    groups.append(Tileset([Tile(c, value) for c in sorted(colours)] + [JOKER] * number_of_jokers))

    return groups


def generate_all_sets():
    sets: set[Tileset] = set()
    sets.update(generate_all_runs())
    sets.update(generate_all_groups())

    return sorted(sets)
