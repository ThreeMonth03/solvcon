# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import hashlib
import io
from pathlib import Path
import sys
import zipfile


VERSION = "2.2.1"
SHA256 = (
    "76ddbf406e9d9b772ec73af2bf925b38"
    "b290b4390cc4064720a08d4b4bca0aa9"
)


def main():
    payload = Path(sys.argv[1]).read_bytes()
    root = Path(sys.argv[2])
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise RuntimeError(
            f"half {VERSION} SHA-256 is {digest}, expected {SHA256}")

    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in ("LICENSE.txt", "include/half.hpp"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(archive.read(name))
            temporary.replace(target)


if __name__ == "__main__":
    main()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
