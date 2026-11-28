import typer

from covid_common.ingest.split_data import split_data

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _common() -> None:
    """Prepare datasets for the COVID-EDA workflows."""


app.command("split", help="Split the dataset into training and test sets.")(split_data)
