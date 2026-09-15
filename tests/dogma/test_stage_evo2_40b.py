"""The staging script's checks, exercised without a cluster or 82 GB of weights."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stage_evo2_40b.sh"
PART_BYTES = 41126745847


def _run(*args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True)


def test_truncated_part_is_refused(tmp_path: Path):
    """A short part must fail, not be accepted: a partial 41 GB shard still
    looks like a file, and the model would load garbage."""
    for name in ("evo2_40b.pt.part0", "evo2_40b.pt.part1"):
        (tmp_path / name).write_bytes(b"short")
    result = _run("--verify-only", str(tmp_path))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "size mismatch" in result.stdout + result.stderr


def test_missing_part_is_refused(tmp_path: Path):
    result = _run("--verify-only", str(tmp_path))
    assert result.returncode == 2
    assert "missing evo2_40b.pt.part0" in result.stdout + result.stderr


def test_merged_file_of_the_wrong_size_is_refused(tmp_path: Path):
    """A merge interrupted partway leaves a plausible-looking .pt behind."""
    for name in ("evo2_40b.pt.part0", "evo2_40b.pt.part1"):
        path = tmp_path / name
        with path.open("wb") as handle:
            handle.truncate(PART_BYTES)
    with (tmp_path / "evo2_40b.pt").open("wb") as handle:
        handle.truncate(PART_BYTES)          # one shard's worth, not two
    result = _run("--verify-only", str(tmp_path))
    assert result.returncode == 2
    assert "evo2_40b.pt" in result.stdout + result.stderr


def test_correctly_staged_tree_passes(tmp_path: Path):
    for name in ("evo2_40b.pt.part0", "evo2_40b.pt.part1"):
        with (tmp_path / name).open("wb") as handle:
            handle.truncate(PART_BYTES)
    with (tmp_path / "evo2_40b.pt").open("wb") as handle:
        handle.truncate(2 * PART_BYTES)
    result = _run("--verify-only", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
