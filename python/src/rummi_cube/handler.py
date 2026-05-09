import re
from typing import Iterable

from aws_lambda_powertools.utilities.data_classes import LambdaFunctionUrlEvent, event_source

from rummi_cube.rummi import TILES
from rummi_cube.strategies import remove_jokers_and_substitutions_from_rack, enter_asap, \
    maximize_value_always_saves_joker_and_joker_substitutes, \
    minimum_non_zero_placed_always_saves_joker_and_joker_substitutes
from rummi_cube.structs import Tile, Tileset
from rummi_cube.tileset_generation import generate_all_groups


@event_source(data_class=LambdaFunctionUrlEvent)
def lambda_handler(event: LambdaFunctionUrlEvent, context):
    if event.request_context.http.method != "GET":
        return {"statusCode": 400, "body": "Invalid method"}

    table_string = event.query_string_parameters.get("table", [])
    rack_string = event.query_string_parameters.get("rack").lower()

    if not rack_string:
        return {"statusCode": 400, "body": "Missing rack parameter"}

    if not tile_string_is_valid(rack_string):
        return {"statusCode": 400, "body": "Rack must be of form 'a4 b4 r4 y4'"}

    rack = Tile.list_from_str(rack_string)

    if len(rack) > len(TILES) * 2:
        return {"statusCode": 400, "body": "Too many rack tiles"}

    if not tile_values_in_range(rack):
        return {"statusCode": 400, "body": "Tile values must be between 1 and 13"}

    tileset_strings = [s.lower() for s in table_string.split(",") if s]  # Filter out empty strings

    # Definitely not bulletproof input validation but good enough to catch typos
    table = []
    for tileset_string in tileset_strings:
        if not tile_string_is_valid(tileset_string):
            return {"statusCode": 400, "body": "Table must be of form 'a1 a2 a3 J,a4 b4 r4'"}

        tileset = Tileset.from_str(tileset_string)

        if len(tileset) > 13:
            return {"statusCode": 400, "body": "Tileset is too long"}

        if not tile_values_in_range(tileset):
            return {"statusCode": 400, "body": "Tile values must be between 1 and 13"}

        if tileset.is_group and tileset not in generate_all_groups():
            return {"statusCode": 400, "body": f"{tileset} is not a valid tileset"}

        if tileset.is_run:
            for i in range(len(tileset)):
                if tileset[i].value != tileset.run_first_tile_value + i and not tileset[i].is_joker():
                    return {"statusCode": 400, "body": f"{tileset} is not a valid tileset"}

        if not tileset.is_group and not tileset.is_run:
            return {"statusCode": 400, "body": f"{tileset} is not a valid tileset"}

        table.append(tileset)

    if event.raw_path == "/entry":
        rack_to_play = remove_jokers_and_substitutions_from_rack(rack, table)

        result = enter_asap(rack_to_play, table)

        if not result.placed:
            return {"statusCode": 200, "body": "Pick up a tile"}

        return {"statusCode": 200, "body": result.display(table)}
    elif event.raw_path == "/maximize-value":
        result = maximize_value_always_saves_joker_and_joker_substitutes(table, rack, True)

        if not result.placed:
            return {"statusCode": 200, "body": "Pick up a tile"}

        return {"statusCode": 200, "body": result.display(table)}
    elif event.raw_path == "/place-minimum":
        result = minimum_non_zero_placed_always_saves_joker_and_joker_substitutes(table, rack, True)

        if not result.placed:
            return {"statusCode": 200, "body": "Pick up a tile"}

        return {"statusCode": 200, "body": result.display(table)}
    else:
        return {"statusCode": 400, "body": f"{event.raw_path} is not a valid path"}


def tile_values_in_range(tiles: Iterable[Tile]) -> bool:
    return all(t.is_joker() or (1 <= t.value <= 13) for t in tiles)


def tile_string_is_valid(rack_string: str) -> bool:
    if re.match(r"^([abryj][0-9]* )*[abryj][0-9]*$", rack_string):
        return True
    return False
