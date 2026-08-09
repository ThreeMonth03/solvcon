# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

if __package__:
    from . import matmul_dispatch as _implementation
else:
    import matmul_dispatch as _implementation


main = _implementation.main


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == "__main__":
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
