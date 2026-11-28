import typer

from covid_common.data.prepare import prepare_train_test_split

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _common() -> None:
    """Prepare datasets for the COVID-EDA workflows."""


app.command("split", help="Split the dataset into training and test sets.")(
    prepare_train_test_split
)
