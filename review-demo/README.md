# PR #1497 standalone review demo

`demo.py` is a single portable script. It uses the unchanged worker control
from PR #1497 and adds a fixed-input summary, Run button, and result table.
All files in this review bundle can live outside the solvcon repository.

## Obtain the script

Save the complete Python code from the PR comment as `demo.py`, or extract
this review ZIP. The script does not depend on the capture harness, a local
launcher, or the author's filesystem paths.

## Reproduce

Prerequisites: an existing solvcon clone and its configured build dependency
environment, including NumPy and PySide6. Use the same Python to build the
extension and run this script. A desktop display is required.

Start in the directory containing `demo.py`. Replace `/path/to/solvcon` with
your existing clone:

```bash
review_demo_dir="$PWD"
git -C /path/to/solvcon fetch https://github.com/solvcon/solvcon.git refs/pull/1497/head
git -C /path/to/solvcon worktree add --detach "$review_demo_dir/checkout" c2eeba9fe4441d1501c47d5b1aae5c5a56855e5e
cd "$review_demo_dir/checkout"
make CMAKE_PRESET=dev-noqt
python3 "$review_demo_dir/demo.py" --repo "$PWD" --output-dir "$review_demo_dir/output"
```

Set dependency paths through the usual make variables if your environment
needs them, for example `CMAKE_PREFIX_PATH` and `CMAKE_ARGS=-Dpybind11_DIR=...`.
No dependency installation is performed by the script.

Once built, launch from any directory:

```bash
python3 /path/to/demo.py --repo /path/to/checkout
```

For a shorter completion run, add `--rounds 2 --repetitions 1`.
Use `--help` to list options. Output defaults to `pr1497-demo-output` under
where you launched the script; `--output-dir` can select another directory.
Paths containing spaces must be quoted in shell commands.

The script selects the checkout for both the UI and worker, changes its
working directory to that checkout, and sets OpenBLAS, OpenMP, MKL, and
Accelerate thread variables to one before importing the numerical modules.
The script does not fetch code, modify the checkout, or build automatically.

`dev-noqt` builds the numerical extension. PySide6 supplies the standalone
window. The C++ Pilot executable and its embedded interpreter are not part
of this reproduction. The reproduction was exercised on WSL2 with Python
3.12.7 and Qt 6.8.0; other platforms were not tested here.

## Interaction and results

1. Click Run. The real worker runs float64 C-contiguous `(256, 256)` operands
   with Naive and NumPy, one warmup, 10 repetitions, and 400 rounds.
2. Observe the active kernel and elapsed time. Run is disabled during work.
3. Click Stop. The worker exits and the demo states that this run has no
   completed results. Run becomes available again.
4. Click Run and let it finish. The table displays kernel, status, median
   milliseconds per call, maximum absolute difference, and relative
   difference. The original artifact path can be selected and copied.
5. Run again. Old rows and the old artifact path are cleared immediately.
   Closing the window during a run stops the worker before the window exits.

The time shown is `median(round_elapsed_ns) / repetitions / 1e6`.
Differences and statuses come from the validated artifact. Missing values
are shown as `-`, not zero; unavailable-result reasons appear in tooltips.
Worker failures are displayed in the demo. An artifact read or validation
failure is reported separately as a result-display error.

The result table is a review helper in the external demo. The formal
Inspector input editor, result view, and Pilot integration remain Tasks
7 through 9. Benchmark times in screenshots are UI evidence, not performance
claims.

## Review materials

- `PR-COMMENT.md`: complete comment draft with the portable script embedded.
  Replace the image attachment placeholders before posting.
- `running.png`, `stopped.png`, `completed.png`, `failed.png`: actual captures.
- `completed-artifact.json`: artifact corresponding to the completed image.
- `verification.json`, `portable-verification.json`: verification records.
- `capture_demo.py`: optional local verification and capture harness.
- `SHA256SUMS`: hashes of the supplied review files.

Screenshots are actual window-content captures without window decorations.
Temporary verification directories and workers were cleaned up after capture,
so screenshot paths identify that run and are not permanent download links.
The preserved artifact is included beside the screenshots.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
