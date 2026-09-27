"""Command-line interface: `padel <command>`."""

from pathlib import Path

import typer

from padel.video import VideoError, probe

MIN_RECOMMENDED_FPS = 60

app = typer.Typer(help="Analiza meczów padla z wideo.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Analiza meczów padla z wideo."""


@app.command()
def info(video: Path) -> None:
    """Wypisz rozdzielczość, klatkaż i długość nagrania."""
    try:
        vi = probe(video)
    except VideoError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e

    minutes, seconds = divmod(vi.duration_s, 60)
    typer.echo(f"Plik:          {vi.path}")
    typer.echo(f"Rozdzielczość: {vi.width}x{vi.height}")
    typer.echo(f"Klatkaż:       {vi.fps:.2f} fps")
    typer.echo(f"Klatki:        {vi.frame_count}")
    typer.echo(f"Długość:       {int(minutes)} min {seconds:04.1f} s")
    if vi.fps < MIN_RECOMMENDED_FPS:
        typer.echo(
            f"Uwaga: klatkaż poniżej {MIN_RECOMMENDED_FPS} fps, szybka piłka będzie rozmyta.",
            err=True,
        )
