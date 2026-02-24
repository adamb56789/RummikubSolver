from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Iterable

from numpy import ndarray


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

    def index(self):
        return 52 if self.is_joker() else 13 * COLOURS.index(self.colour) + self.value - 1

    def is_joker(self):
        return self.colour == "J"

    @staticmethod
    def from_str(s: str) -> list['Tile']:
        if not s:
            return []
        return [
            Tile(tile[0], int(tile[1:]) if len(tile) > 1 else 0)
            for tile in s.split(" ")
        ]


class Tileset:
    is_run: bool
    is_group: bool
    run_colour: str
    group_value: int
    group_colours: list[str]
    number_of_jokers: int
    contains_joker: bool

    tiles: tuple[Tile, ...]

    def __init__(self, tiles: Iterable[Tile]):
        tiles = list(tiles)

        colour_count = Counter(t.colour for t in tiles)
        number_of_jokers = colour_count.get("J", 0)

        is_ambiguous = False
        if len(tiles) - number_of_jokers < 2:
            is_ambiguous = True
            self.is_group = True

            # Can only be a run if there is space for it, e.g. (J a1 J) cannot.
            first_normal_tile_index, first_normal_tile_value = next(
                (i, t.value) for i, t in enumerate(tiles) if not t.is_joker()
            )
            first_tile_value = first_normal_tile_value - first_normal_tile_index
            self.is_run = 1 <= first_tile_value and (first_tile_value + len(tiles)) <= 13
        elif len(colour_count) <= 2:
            self.is_run = True
            self.is_group = False
        else:
            self.is_run = False
            self.is_group = True

        set_colours = list(colour_count.keys())
        if "J" in set_colours:
            set_colours.remove("J")
        if self.is_run:
            self.run_colour = set_colours[0]

        if self.is_group:
            self.group_value = next(t for t in tiles if not t.is_joker()).value
            self.group_colours = set_colours

        if self.is_group and not is_ambiguous:
            self.tiles = tuple(sorted(tiles))
        else:
            self.tiles = tuple(tiles)

        self.number_of_jokers = number_of_jokers
        self.contains_joker = number_of_jokers > 0

    def __len__(self):
        return len(self.tiles)

    def __lt__(self, other):
        return self.tiles < other.tiles

    def __eq__(self, other):
        return self.tiles == other.tiles

    def __hash__(self):
        return hash(self.tiles)

    def __iter__(self):
        return iter(self.tiles)

    def __getitem__(self, item):
        return self.tiles[item]

    def __repr__(self):
        return "(" + " ".join(str(t) for t in self.tiles) + ")"

    def split_tileset(self) -> list['Tileset']:
        n = len(self)
        assert n >= 6

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
                result.append(Tileset(self.tiles[i:i + s]))
                i += s
                n -= s

        return result

    @staticmethod
    def from_str(s: str) -> 'Tileset':
        return Tileset([
            Tile(tile[0], int(tile[1:]) if len(tile) > 1 else 0)
            for tile in s.split(" ")
        ])


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
class TilesetModelParams:
    set_tile_matrices_by_k_set: dict[tuple, ndarray]
    """
    2d arrays where each column corresponds to a set, and each value in the column is the count of the corresponding tile in that set 
    """

    substitution_tiles_by_k: list[list[int]]
    """
    List of tile indexes that can substitute for joker k
    """

    tilesets_by_k_set: dict[tuple, list[Tileset]]
    joker_count: int


@dataclass
class RummiResult:
    table: list[Tileset]
    placed: list[Tile]
    remaining: list[Tile]

    @staticmethod
    def from_strings(table: list[str], placed: str, remaining: str) -> 'RummiResult':
        return RummiResult(
            [Tileset.from_str(s) for s in table],
            Tile.from_str(placed),
            Tile.from_str(remaining),
        )


COLOURS = "brya"
