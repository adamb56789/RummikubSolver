from itertools import combinations

from rummi_cube.structs import Tileset, COLOURS, Tile


def create_runs():
    runs: list[Tileset] = []
    for c in COLOURS:
        for run_length in [3, 4, 5]:

            # No Jokers
            for i in range(1, 15 - run_length):
                runs.append(Tileset(Tile(c, j) for j in range(i, i + run_length)))

            # One joker
            for i in range(1, 15 - run_length):
                for joker_pos in range(i, i + run_length):
                    runs.append(Tileset(
                        JOKER if j == joker_pos else Tile(c, j)
                        for j in range(i, i + run_length)
                    ))

            # Two jokers
            for i in range(1, 15 - run_length):
                for joker_pos_1 in range(i, i + run_length - 1):
                    for joker_pos_2 in range(joker_pos_1 + 1, i + run_length):
                        runs.append(Tileset(
                            JOKER if j == joker_pos_1 or j == joker_pos_2 else Tile(c, j)
                            for j in range(i, i + run_length)
                        ))
    return runs


def create_groups():
    groups: list[Tileset] = []

    for v in range(1, 14):
        for cs in combinations(COLOURS, 3):
            groups.append(Tileset(Tile(c, v) for c in sorted(cs)))

        for cs in combinations(COLOURS, 2):
            groups.append(Tileset([Tile(c, v) for c in sorted(cs)] + [JOKER]))

        # Groups of 3 tiles with 2 jokers already covered by runs

        for cs in combinations(COLOURS, 4):
            groups.append(Tileset(Tile(c, v) for c in sorted(cs)))

        for cs in combinations(COLOURS, 3):
            groups.append(Tileset([Tile(c, v) for c in sorted(cs)] + [JOKER]))

        for cs in combinations(COLOURS, 2):
            groups.append(Tileset([Tile(c, v) for c in sorted(cs)] + [JOKER] * 2))

    return groups


JOKER = Tile("J", 0)
