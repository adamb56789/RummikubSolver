from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import islice
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

    tiles: tuple[Tile, ...]

    def __init__(self, tiles: Iterable[Tile]):
        self.tiles = tuple(tiles)

        # The order sometimes matters for Tilesets with only one real tile, even if they cannot not be a run.
        # For example (J a1 J) cannot be switched with (a1 J J) as then it could become a run.
        # It can however be switched with (J J a1). We use for example (J J a1) or (a12 J J) as the canonical form.
        if self.is_run:
            self.tiles = tuple(self.tiles)
        else:
            if self.has_only_one_real_tile:
                # If the real tile's value is 12 or 13 then reverse sort puts the J on the end like (a12 J J)
                self.tiles = tuple(sorted(self.tiles, reverse=self.run_first_tile_value > 11))
            self.tiles = tuple(sorted(self.tiles))

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

    @cached_property
    def is_run(self):
        if self.has_only_one_real_tile:
            return 1 <= self.run_first_tile_value and self.run_first_tile_value + len(self.tiles) <= 13

        return len(self.colours) == 1

    @cached_property
    def is_group(self):
        return len(self.colours) > 1 or self.has_only_one_real_tile

    @cached_property
    def has_only_one_real_tile(self):
        return len(self.tiles) - self.number_of_jokers < 2

    @cached_property
    def number_of_jokers(self) -> int:
        return len([t for t in self.tiles if t.is_joker()])

    @cached_property
    def contains_joker(self) -> bool:
        return self.number_of_jokers > 0

    @cached_property
    def run_first_tile_value(self) -> int:
        first_real_tile = next(t for t in self.tiles if not t.is_joker())
        return first_real_tile.value - self.tiles.index(first_real_tile)

    @cached_property
    def run_colour(self) -> str:
        if not self.is_run:
            raise ValueError("Tileset must be a run")

        return next(t.colour for t in self.tiles if not t.is_joker())

    @cached_property
    def group_value(self) -> int:
        if not self.is_group:
            raise ValueError("Tileset must be a run")

        return next(t for t in self.tiles if not t.is_joker()).value

    @cached_property
    def colours(self) -> list[str]:
        return list({t.colour for t in self.tiles if not t.is_joker()})

    def split_tileset(self) -> list['Tileset']:
        it = iter(self.tiles)
        n = len(self.tiles)
        result = []

        while n:
            s = 5 if n > 7 else 4 if n == 7 else 3 if n > 5 else n
            result.append(Tileset(list(islice(it, s))))
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
JOKER = Tile("J", 0)
