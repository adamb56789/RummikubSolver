from rummi_cube.structs import RummiResult, Tileset

# Notice: this file is 90% vibe-coded.

HEADER = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Rummi Solver</title>

<link rel="icon" href="data:,">

<style>
body {
    background: #f5f5f5;
    color: #222;
    font-family: Consolas, Monaco, monospace;
    margin: 0;
    padding: 40px;
}

.container {
    max-width: 900px;
    margin: 0 auto;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

h1 {
    margin-top: 0;
}

.section-title {
    margin-top: 24px;
    margin-bottom: 12px;
    font-size: 1.1rem;
    font-weight: bold;
}

.tileset {
    padding: 8px 12px;
    margin: 6px 0;
    background: #fafafa;
    border-left: 4px solid #888;
    border-radius: 6px;
}

.remaining {
    color: #666;
}

.placed {
    color: #0a7a2f;
    font-weight: bold;
}

label {
    display: block;
    margin-top: 18px;
    margin-bottom: 8px;
    font-weight: bold;
}

input[type="text"],
textarea {
    width: 100%;
    box-sizing: border-box;
    padding: 12px;
    border: 1px solid #ccc;
    border-radius: 8px;
    font-family: inherit;
    font-size: 14px;
    background: #fafafa;
}

textarea {
    min-height: 140px;
    resize: vertical;
}

.radio-group {
    margin-top: 12px;
}

.radio-option {
    margin-bottom: 10px;
}

button {
    margin-top: 24px;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 15px;
    font-family: inherit;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.help {
    color: #666;
    font-size: 13px;
    margin-top: 6px;
}

.error-box {{
    background: #fff1f2;
    border-left: 4px solid #dc2626;
    color: #7f1d1d;
    padding: 16px;
    border-radius: 8px;
    white-space: pre-wrap;
    line-height: 1.5;
}}

</style>
</head>
"""


def home_page():
    return f"""
    {HEADER}

    <body>
    <div class="container">

        <div class="card">
            <h1>Rummi Solver</h1>

            <form
                action="/solve"
                method="get"
                target="_blank"
                autocomplete="off"
            >

                <label for="rack">Rack</label>

                <input
                    type="text"
                    id="rack"
                    name="rack"
                    placeholder="Example: r1 r2 r3 b5 a5 y5 J"
                    autocomplete="off"
                >

                <div class="help">
                    Separate tiles with spaces. 'a' is black
                </div>

                <label for="table">Table</label>

                <textarea
                    id="table"
                    name="table"
                    placeholder="One set per line&#10;Example:&#10;r1 r2 r3&#10;b7 a7 r7"
                    autocomplete="off"
                ></textarea>

                <div class="help">
                    One tileset per line.
                </div>

                <div class="section-title">
                    Strategy
                </div>

                <div class="radio-group">

                    <div class="radio-option">
                        <label>
                            <input
                                type="radio"
                                name="strategy"
                                value="entry"
                                checked
                            >
                            30-point entry
                        </label>
                    </div>

                    <div class="radio-option">
                        <label>
                            <input
                                type="radio"
                                name="strategy"
                                value="maximize_value"
                            >
                            Maximize value
                        </label>
                    </div>

                    <div class="radio-option">
                        <label>
                            <input
                                type="radio"
                                name="strategy"
                                value="minimum_tiles"
                            >
                            Place minimum tiles
                        </label>
                    </div>

                </div>

                <button type="submit">
                    Solve
                </button>

            </form>

        </div>

    </div>
    </body>
    </html>
    """


def display_result(result: RummiResult, previous_table: list[Tileset]):
    previous_sets = [ts for ts in result.table if ts in previous_table]
    new_sets = [ts for ts in result.table if ts not in previous_table]

    untouched_html = "".join(
        f"<div class='tileset'>{str(ts).strip('()')}</div>"
        for ts in previous_sets
    )

    updated_html = "".join(
        f"<div class='tileset'>{str(ts).strip('()')}</div>"
        for ts in new_sets
    )

    return f"""
        {HEADER}

        <body>
        <div class="container">

        <div class="card">
            <h1>Rummi Solver</h1>

            <div class="remaining">
                <strong>Tiles remaining:</strong><br>
                {str(result.remaining).strip('[]').replace(',', '') or '(none)'}
            </div>

            <br>

            <div class="placed">
                <strong>Tiles placed:</strong><br>
                {str(result.placed).strip('[]').replace(',', '') or '(none)'}
            </div>

            <div class="section-title">Table untouched</div>
            {untouched_html or "<i>None</i>"}

            <div class="section-title">Table updated</div>
            {updated_html or "<i>None</i>"}

        </div>
        </div>
        </body>
        </html>
        """


def error_page(message: str):
    return f"""
    {HEADER}

    <body>
    <div class="container">

        <div class="card">

            <h1>Error</h1>

            <div class="error-box">
                {message}
            </div>

        </div>

    </div>

    </body>
    </html>
    """
