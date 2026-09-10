"""Phase 7: the published figures are a contract, and so is the prose.

`scripts/reproduce.py` re-derives every number the repository states and fails
when one moves. That guards the code path. It does not guard the *sentences* --
a figure can be corrected in `PUBLISHED` while a paragraph three files away
still quotes the superseded one, which is exactly how the refusal conclusion
in `docs/generation.md` survived as long as it did.

These tests close that gap without running the eval: they assert the contract
is complete, that every claim it makes is present in the documents, and that
the ladder indices `reproduce.py` reads by position still exist.

The last class covers the reporting scripts the README points readers at,
because a command that crashes is a broken claim too.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gtrag.ablation import ABLATION_LADDER, CHUNKING_SWEEP, GENERATION_LADDER

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a script by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `@dataclass` resolves annotations against `sys.modules[cls.__module__]`,
    # so the module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reproduce():
    return _load_script("reproduce")


class TestTheContractIsComplete:
    def test_every_published_figure_says_how_it_is_written_down(self, reproduce):
        """A figure with no documented rendering is unguarded prose."""
        undocumented = set(reproduce.PUBLISHED) - set(reproduce.DOC_CLAIMS)
        assert not undocumented, f"no DOC_CLAIMS entry for: {sorted(undocumented)}"

    def test_no_claim_describes_a_figure_that_is_not_measured(self, reproduce):
        """The reverse: a claim about a figure nothing re-derives can drift."""
        unmeasured = set(reproduce.DOC_CLAIMS) - set(reproduce.PUBLISHED)
        assert not unmeasured, f"claimed but never reproduced: {sorted(unmeasured)}"

    def test_every_named_document_exists(self, reproduce):
        for name in reproduce.DOCS:
            assert (ROOT / name).is_file(), name


class TestTheProseIsCurrent:
    def test_every_figure_is_quoted_as_measured(self, reproduce):
        """The check that would have caught the superseded refusal conclusion.

        Runs in milliseconds because it reads the documents rather than the
        eval -- so it is a test, not a script somebody remembers to invoke.
        """
        assert reproduce.check_docs() == []

    def test_the_check_actually_fails_on_a_stale_claim(self, reproduce, monkeypatch):
        """A guard that cannot fail is not a guard."""
        monkeypatch.setitem(
            reproduce.DOC_CLAIMS, "ladder.baseline.ndcg@10", ("0.7409 (superseded)",)
        )
        assert reproduce.check_docs()


class TestLadderIndicesAreStillValid:
    """`reproduce.py` reads ladder rungs by position. Reordering the ladder
    would silently publish a different rung's number under the old name."""

    def test_retrieval_ladder_has_the_five_rungs_read_by_position(self):
        assert len(ABLATION_LADDER) == 5
        assert [c.label for c in ABLATION_LADDER][0].startswith("baseline")

    def test_chunking_sweep_covers_every_published_strategy(self, reproduce):
        swept = {c.chunker for c in CHUNKING_SWEEP}
        published = {
            key.split(".")[1] for key in reproduce.PUBLISHED if key.startswith("chunking.")
        }
        assert published <= swept

    def test_generation_ladder_ends_on_the_refusal_rung(self):
        assert GENERATION_LADDER[-1].refusal_signal == "top_score"
        assert not GENERATION_LADDER[0].refusal_signal


class TestTheAblationTableSurvivesAStrayFile:
    """`make ablation` globs every `*.json` in the results directory.

    It used to load them all as run results and crash on the first one that
    was not -- which `make reproduce` triggered the moment it started writing
    its report there. Baselines, calibration exports and editor backups land
    in that directory too, so the glob has to tolerate them: a stray file must
    not take down a command the README tells people to run.
    """

    def _run(self, results_dir: Path):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_ablation_table.py"),
                "--results",
                str(results_dir),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

    def test_a_non_run_json_is_skipped_not_fatal(self, tmp_path):
        real = next(
            path
            for path in (ROOT / "evals" / "results").glob("*.json")
            if "run_id" in json.loads(path.read_text(encoding="utf-8"))
        )
        (tmp_path / "run.json").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
        (tmp_path / "reproduce.json").write_text(json.dumps({"figures": []}), encoding="utf-8")

        result = self._run(tmp_path)

        assert result.returncode == 0, result.stderr
        assert "reproduce.json" in result.stderr  # named, not silently dropped
        assert "| Configuration |" in result.stdout

    def test_a_directory_of_only_stray_files_fails_loudly(self, tmp_path):
        (tmp_path / "reproduce.json").write_text(json.dumps({"figures": []}), encoding="utf-8")

        result = self._run(tmp_path)

        assert result.returncode == 1
        assert "no run result files" in result.stdout
