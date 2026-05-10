import argparse
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

from rummi_cube.rummi import find_best_move, InfeasibleSolutionException
from rummi_cube.structs import ALL_TILES_STRINGS_WITH_JOKERS, Tile, Tileset, RummiResult, Config, JokerMode, \
    MaximizeMode, COLOURS


class EmptyBagException(Exception):
    pass


class IllegalMoveException(Exception):
    pass


def draw_from_bag(bag: list[Tile], number_of_tiles: int) -> list[Tile]:
    if len(bag) == 1:
        raise EmptyBagException()

    return [bag.pop() for _ in range(number_of_tiles)]


class Player:

    def __init__(self, name: str, starting_tiles: list[Tile],
                 strategy_function: Callable[[list[Tileset], list[Tile], bool], RummiResult]):
        self.name = name
        self.rack = starting_tiles
        self.strategy_function = strategy_function
        self.entered_game = False  # We have not entered the game until we play a move with new tiles of value 30

    def decide_move(self, table: list[Tileset]) -> RummiResult:
        return self.strategy_function(table, self.rack, self.entered_game)

    def pick_up_from_bag(self, bag: list[Tile]):
        self.rack.append(draw_from_bag(bag, 1)[0])

    def __repr__(self):
        return f"{self.name} {self.strategy_function.__name__}: {self.rack}"


def run_game(player_strategies: list[Callable[[list[Tileset], list[Tile], bool], RummiResult]], random_seed=0) -> list[
    Player]:
    random.seed(random_seed)

    bag = Tile.list_from_str(" ".join(ALL_TILES_STRINGS_WITH_JOKERS))
    random.shuffle(bag)

    players = [Player(f"Player {i}", draw_from_bag(bag, 14), strategy) for i, strategy in enumerate(player_strategies)]

    table: list[Tileset] = []

    turn = 0
    while True:
        start_time = time.time()

        player_index = turn % len(players)
        active_player = players[player_index]

        just_entered_the_game = False
        if not active_player.entered_game:
            # Before entering the game we can't manipulate or add to tiles on the board
            move_result = active_player.decide_move([])

            if len(move_result.placed) == 0:
                active_player.pick_up_from_bag(bag)
                # print(f"Does not enter the game and picks up a tile: {active_player}")
            elif sum(ts.numerical_value_to_enter_game() for ts in move_result.table) < 30:
                raise IllegalMoveException(
                    f"Need to place a value of 30 to empty the game. Player: {active_player.name}, move: {move_result}")
            else:
                # print(f"{active_player.name} enters the game by making move: {move_result}")

                just_entered_the_game = True

                active_player.entered_game = True
                active_player.rack = move_result.remaining
                table.extend(move_result.table)

                # Check if the player won
                if not active_player.rack:
                    return players

        # If the player just placed 30 to enter the game they can keep playing and place more tiles including manipulation
        if active_player.entered_game:
            move_result = active_player.decide_move(table)
            if len(move_result.placed) == 0:
                if not just_entered_the_game:
                    active_player.pick_up_from_bag(bag)

                    # print(f"Picks up a tile: {active_player}")
                else:
                    pass
                    # print(f"{active_player.name} takes no further action after entering the game")
            else:
                # print(f"{active_player.name} makes move: {move_result}")

                active_player.rack = move_result.remaining
                table = move_result.table

            # Check if the player won
            if not active_player.rack:
                return players

        # print(f"Turn: {turn}. Time: {time.time() - start_time:.3f}")
        turn += 1


joker_mode = JokerMode.LOCKING


def enter_asap(rack: list[Tile], table: list[Tileset]) -> RummiResult:
    # Enter the game ASAP, avoid placing jokers

    config = Config(joker_mode, maximize_mode=MaximizeMode.VALUE_PLACED, joker_value=-100)
    result = find_best_move(table, rack, config)
    if sum(ts.numerical_value_to_enter_game() for ts in result.table) >= 30:
        return result
    else:
        return RummiResult(table, [], rack)


def hoarder(table: list[Tileset], rack: list[Tile], entered_game: bool) -> RummiResult:
    if not entered_game:
        return enter_asap(rack, table)
    else:
        # Do nothing unless we can place every tile in our rack

        config = Config(joker_mode, maximize_mode=MaximizeMode.TILES_PLACED)
        result = find_best_move(table, rack, config)

        if len(result.remaining) == 0:
            return result
        else:
            return RummiResult(table, [], rack)


def max_tiles_every_time(table: list[Tileset], rack: list[Tile], entered_game: bool) -> RummiResult:
    if not entered_game:
        return enter_asap(rack, table)
    else:
        config = Config(joker_mode, maximize_mode=MaximizeMode.TILES_PLACED)
        return find_best_move(table, rack, config)


def maximize_value_joker_value_one(table: list[Tileset], rack: list[Tile], entered_game: bool) -> RummiResult:
    if not entered_game:
        return enter_asap(rack, table)
    else:
        config = Config(joker_mode, maximize_mode=MaximizeMode.VALUE_PLACED, joker_value=1)
        return find_best_move(table, rack, config)


def maximize_value_always_saves_joker(table: list[Tileset], rack: list[Tile], entered_game: bool) -> RummiResult:
    if not entered_game:
        return enter_asap(rack, table)
    else:
        # Always try to win by placing jokers
        tiles_config = Config(joker_mode, maximize_mode=MaximizeMode.TILES_PLACED)
        max_tiles_result = find_best_move(table, rack, tiles_config)

        if not max_tiles_result.remaining:
            return max_tiles_result

        rack_tiles_to_play = [t for t in rack if not t.is_joker()]
        value_config = Config(joker_mode, maximize_mode=MaximizeMode.VALUE_PLACED, joker_value=-1)
        return find_best_move(table, rack_tiles_to_play, value_config)


def remove_jokers_and_substitutions_from_rack(rack: list[Tile], table: list[Tileset]) -> list[Tile]:
    # Find all tiles which could be used to substitute a joker on the table
    substitution_tiles: set[Tile] = set()
    for tileset in table:
        if tileset.number_of_jokers == 0:
            continue

        if tileset.is_run:
            joker_indexes = [i for i, t in enumerate(tileset) if t.is_joker()]
            joker_values = [tileset.run_first_tile_value + i for i in joker_indexes]
            substitution_tiles.update(Tile(tileset.run_colour, v) for v in joker_values)

        if tileset.is_group:
            missing_colours = [c for c in COLOURS if c not in tileset.colours]
            for i in range(tileset.number_of_jokers):
                for c in missing_colours:
                    substitution_tiles.add(Tile(c, tileset.group_value).index())

    rack_tiles_to_play = [t for t in rack if not t.is_joker() and t not in substitution_tiles]
    return rack_tiles_to_play


def maximize_value_always_saves_joker_and_joker_substitutes(table: list[Tileset], rack: list[Tile],
                                                            entered_game: bool) -> RummiResult:
    if not entered_game:
        return enter_asap(rack, table)
    else:
        # Always try to win by placing jokers
        tiles_config = Config(joker_mode, maximize_mode=MaximizeMode.TILES_PLACED)
        max_tiles_result = find_best_move(table, rack, tiles_config)

        if not max_tiles_result.remaining:
            return max_tiles_result

        rack_tiles_to_play = remove_jokers_and_substitutions_from_rack(rack, table)

        config = Config(joker_mode, maximize_mode=MaximizeMode.VALUE_PLACED, joker_value=-1)
        result = find_best_move(table, rack_tiles_to_play, config)
        result.remaining.extend(t for t in rack if t not in rack_tiles_to_play)  # Add the skipped tiles back in
        return result


def minimum_non_zero_placed_always_saves_joker_and_joker_substitutes(table: list[Tileset], rack: list[Tile],
                                                                     entered_game: bool) -> RummiResult:
    if not entered_game:
        return enter_asap(rack, table)
    else:
        # Always try to win by placing jokers
        tiles_config = Config(joker_mode, maximize_mode=MaximizeMode.TILES_PLACED)
        max_tiles_result = find_best_move(table, rack, tiles_config)

        if not max_tiles_result.remaining:
            return max_tiles_result

        rack_tiles_to_play = remove_jokers_and_substitutions_from_rack(rack, table)

        config = Config(joker_mode, maximize_mode=MaximizeMode.MINIMUM_NON_ZERO_PLACED, joker_value=-1)
        try:
            result = find_best_move(table, rack_tiles_to_play, config)
            result.remaining.extend(t for t in rack if t not in rack_tiles_to_play) # Add the skipped tiles back in
            return result
        except InfeasibleSolutionException:
            return RummiResult(table, [], rack)


def simulate_game(seed: int):
    strategies = [
        maximize_value_always_saves_joker_and_joker_substitutes,
        maximize_value_always_saves_joker_and_joker_substitutes,
        minimum_non_zero_placed_always_saves_joker_and_joker_substitutes,
        minimum_non_zero_placed_always_saves_joker_and_joker_substitutes,
    ]

    random.seed(seed)
    random.shuffle(strategies)

    print(f"[START] Seed={seed} Order={[s.__name__ for s in strategies]}")

    try:
        game_result = run_game(strategies, seed)
        winner = next(player for player in game_result if not player.rack)

        return "win", winner.strategy_function.__name__

    except EmptyBagException:
        return "empty_bag", None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to simulate",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting seed",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker processes",
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    scoreboard = Counter()
    empty_bags = 0

    seeds = range(args.seed, args.seed + args.games)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(simulate_game, seed): seed
            for seed in seeds
        }

        for future in as_completed(futures):
            result_type, winner_name = future.result()

            if result_type == "win":
                scoreboard[winner_name] += 1

            elif result_type == "empty_bag":
                empty_bags += 1

    total_time = time.perf_counter() - start_time

    print("\n=== FINAL SCOREBOARD ===")
    for strategy_name, wins in scoreboard.most_common():
        print(f"{strategy_name}: {wins}")

    print(f"\nEmpty bag games: {empty_bags}")
    print(f"Total simulations: {args.games}")
    print(f"Workers: {args.workers}")
    print(f"Total runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
