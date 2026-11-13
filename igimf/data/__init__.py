"""Reference observations distributed with igimf."""

from importlib.resources import files


def open_data(filename):
    """Open a bundled table as a binary stream; use as a context manager."""
    return files(__package__).joinpath(filename).open("rb")
