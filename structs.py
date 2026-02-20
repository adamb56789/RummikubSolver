from dataclasses import dataclass
from enum import Enum
from typing import Optional

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
class JokerParams:
    set_to_index_map: dict[tuple[Tile, ...], int]
    """
    Mapping from tuple tile sets to their "set index" in the other data structures
    """

    table_sets_array: ndarray
    """
    1d array where each element is the count of sets of that index on the table
    """

    set_tile_matrix: ndarray
    """
    2d array where each column corresponds to a set, and each value in the column is the count of the corresponding tile in that set 
    """

    substitution_tile_array: ndarray
    """
    1d array where each element is 1 if that tile can substitute the joker, 0 otherwise
    """

    sets: list[tuple[Tile, ...]]


@dataclass
class RummiResult:
    table: list[tuple[Tile, ...]]
    placed: list[Tile]
    remaining: list[Tile]

    @staticmethod
    def from_strings(table: list[str], placed: str, remaining: str) -> 'RummiResult':
        return RummiResult(
            [tuple(Tile.from_str(s)) for s in table],
            Tile.from_str(placed),
            Tile.from_str(remaining),
        )


COLOURS = "brya"
