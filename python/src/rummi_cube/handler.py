import re
from typing import Iterable

from aws_lambda_powertools.utilities.data_classes import LambdaFunctionUrlEvent, event_source

from rummi_cube.rummi import TILES
from rummi_cube.strategies import remove_jokers_and_substitutions_from_rack, enter_asap, \
    maximize_value_always_saves_joker_and_joker_substitutes, \
    minimum_non_zero_placed_always_saves_joker_and_joker_substitutes
from rummi_cube.structs import Tile, Tileset
from rummi_cube.tileset_generation import generate_all_groups
from rummi_cube.website import home_page, display_result, error_page


class ClientError(Exception):
    pass


@event_source(data_class=LambdaFunctionUrlEvent)
def lambda_handler(event: LambdaFunctionUrlEvent, context):
    try:
        return {"statusCode": 200, "body": handle_event(event), "headers": {
            "Content-Type": "text/html"
        }}
    except ClientError as e:
        return {"statusCode": 400, "body": error_page(str(e)), "headers": {
            "Content-Type": "text/html"
        }}


def handle_event(event: LambdaFunctionUrlEvent):
    print(event)

    if event.request_context.http.method != "GET":
        raise ClientError("Invalid method")

    if event.raw_path == "/":
        return home_page()
    elif event.raw_path != "/solve":
        raise ClientError(f"{event.raw_path} is not a valid path")

    strategy = event.query_string_parameters.get("strategy")
    table_string = event.query_string_parameters.get("table", "")
    rack_string = event.query_string_parameters.get("rack")

    if not rack_string:
        raise ClientError("Missing rack parameter")

    rack_string = rack_string.lower().replace("j", "J")

    if not tile_string_is_valid(rack_string):
        raise ClientError("Rack must be of form 'a4 b4 r4 y4'")

    rack = Tile.list_from_str(rack_string)

    if len(rack) > len(TILES) * 2:
        raise ClientError("Too many rack tiles")

    if not tile_values_in_range(rack):
        raise ClientError("Tile values must be between 1 and 13")

    tileset_strings = [
        line.strip()
        for line in table_string.splitlines()
        if line.strip()
    ]

    # Definitely not bulletproof input validation but good enough to catch typos
    table = []
    for tileset_string in tileset_strings:
        if not tile_string_is_valid(tileset_string):
            raise ClientError(f"Invalid tileset: {tileset_string}")

        tileset = Tileset.from_str(tileset_string)

        if len(tileset) > 13:
            raise ClientError("Tileset is too long")

        if not tile_values_in_range(tileset):
            raise ClientError("Tile values must be between 1 and 13")

        if tileset.is_group and tileset not in generate_all_groups():
            raise ClientError(f"{tileset} is not a valid tileset")

        if tileset.is_run:
            for i in range(len(tileset)):
                if tileset[i].value != tileset.run_first_tile_value + i and not tileset[i].is_joker():
                    raise ClientError(f"{tileset} is not a valid tileset")

        if not tileset.is_group and not tileset.is_run:
            raise ClientError(f"{tileset} is not a valid tileset")

        table.append(tileset)

    if strategy == "entry":
        rack_to_play = remove_jokers_and_substitutions_from_rack(rack, table)

        result = enter_asap(rack_to_play, [])

        if not result.placed:
            return "Pick up a tile"

        result.table.extend(table)
        result.remaining.extend(t for t in rack if t not in rack_to_play)  # Add the skipped tiles back in

        return display_result(result, table)
    elif strategy == "maximize_value":
        result = maximize_value_always_saves_joker_and_joker_substitutes(table, rack, True)

        if not result.placed:
            return "Pick up a tile"

        return display_result(result, table)
    elif strategy == "minimum_tiles":
        result = minimum_non_zero_placed_always_saves_joker_and_joker_substitutes(table, rack, True)

        if not result.placed:
            return "Pick up a tile"

        return display_result(result, table)
    else:
        raise ClientError(f"{strategy} is not a strategy")


def tile_values_in_range(tiles: Iterable[Tile]) -> bool:
    return all(t.is_joker() or (1 <= t.value <= 13) for t in tiles)


def tile_string_is_valid(rack_string: str) -> bool:
    if re.match(r"^([abryJ][0-9]* )*[abryJ][0-9]*$", rack_string):
        return True
    return False
