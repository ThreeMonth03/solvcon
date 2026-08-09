# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

from . import data
from . import environment
from . import measurement
from . import sampling
from . import collect
from . import model
from . import codegen
from . import policy
from . import cli


_MODULES = (
    data,
    sampling,
    environment,
    measurement,
    collect,
    model,
    codegen,
    policy,
    cli,
)


def __getattr__(name):
    for module in _MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = set(globals())
    for module in _MODULES:
        names.update(dir(module))
    return sorted(names)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
