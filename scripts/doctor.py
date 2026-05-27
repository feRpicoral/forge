"""Diagnose Forge's environment against what the harness, server, and eval need.

``make doctor`` parses the constraint files, inspects the active venv, and
prints a green/yellow/red report per check. Exits 0 when everything's healthy,
1 when at least one required component is missing or mispinned.

This exists because ``uv sync`` and the out-of-band ``uv pip install -c
constraints/...`` flow can drift — running ``uv run`` after a fresh ``uv sync``
silently unwinds the GPU-coupled installs and the next ``make rehearse`` fails
deep into vLLM imports. Catch that here in one second instead.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str | None = None

    def emit(self) -> None:
        glyph = "✓" if self.ok else "✗"
        color_start = "\033[32m" if self.ok else "\033[31m"
        color_end = "\033[0m"
        if sys.stdout.isatty():
            print(f"{color_start}{glyph}{color_end} {self.name:<42s} {self.detail}")
        else:
            print(f"{glyph} {self.name:<42s} {self.detail}")
        if not self.ok and self.fix:
            print(f"     ↳ fix: {self.fix}")


def _installed_version(distribution: str) -> Version | None:
    try:
        return Version(md.version(distribution))
    except md.PackageNotFoundError:
        return None


def _parse_constraint_file(path: Path) -> Iterator[Requirement]:
    """Yield each requirement line from a uv-style constraints file.

    Skips blank lines and comments. Tolerates trailing comments.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            yield Requirement(line)
        except Exception as exc:
            # Permissive on purpose: a malformed constraint shouldn't take down
            # the diagnostic; just warn and move on so the rest of the checks run.
            print(f"WARNING: could not parse '{line}' in {path}: {exc}", file=sys.stderr)


def check_constraint_file(path: Path, label: str) -> list[Check]:
    """For each requirement in ``path``, check the installed version satisfies it."""
    checks: list[Check] = []
    for req in _parse_constraint_file(path):
        installed = _installed_version(req.name)
        if installed is None:
            checks.append(
                Check(
                    name=f"{label}: {req.name}",
                    ok=False,
                    detail="not installed",
                    fix=_install_hint(path, req.name),
                )
            )
            continue
        if req.specifier and installed not in req.specifier:
            checks.append(
                Check(
                    name=f"{label}: {req.name}",
                    ok=False,
                    detail=f"{installed} does not satisfy {req.specifier}",
                    fix=_install_hint(path, req.name),
                )
            )
            continue
        spec = str(req.specifier) if req.specifier else "any"
        checks.append(Check(name=f"{label}: {req.name}", ok=True, detail=f"{installed}  ({spec})"))
    return checks


def _install_hint(constraint_path: Path, package: str) -> str:
    return f'uv pip install -c {constraint_path.as_posix()} "{package}"'


def check_executable(name: str, hint: str) -> Check:
    found = shutil.which(name)
    return Check(
        name=f"executable: {name}",
        ok=found is not None,
        detail=found or "not on PATH",
        fix=None if found else hint,
    )


def check_python_version() -> Check:
    required = (3, 12)
    current = sys.version_info[:2]
    ok = current == required
    return Check(
        name="python version",
        ok=ok,
        detail=f"{sys.version.split()[0]}",
        fix=None if ok else "uv python install 3.12  &&  uv venv --python 3.12",
    )


def check_huggingface_hub_for_transformers() -> Check:
    """transformers 4.x has a runtime guard on huggingface-hub<1.0."""
    transformers = _installed_version("transformers")
    if transformers is None:
        return Check(
            name="transformers ↔ huggingface-hub",
            ok=False,
            detail="transformers not installed; skipping",
        )
    hub = _installed_version("huggingface-hub")
    if hub is None:
        return Check(
            name="transformers ↔ huggingface-hub",
            ok=False,
            detail="huggingface-hub not installed",
        )
    transformers_4x = transformers.major == 4
    hub_lt_1 = hub.major == 0
    ok = (not transformers_4x) or hub_lt_1
    return Check(
        name="transformers ↔ huggingface-hub",
        ok=ok,
        detail=f"transformers={transformers}, huggingface-hub={hub}",
        fix=(None if ok else "uv pip install -c constraints/serve.txt vllm transformers"),
    )


def run_checks(*, eval_too: bool) -> list[Check]:
    repo_root = Path(__file__).resolve().parents[1]
    checks: list[Check] = [check_python_version()]

    # The serve constraint family — vllm + transformers + the bits they pull in.
    serve_constraint = repo_root / "constraints" / "serve.txt"
    checks.extend(check_constraint_file(serve_constraint, "serve"))
    checks.append(check_huggingface_hub_for_transformers())
    checks.append(
        check_executable(
            "vllm",
            f"uv pip install -c {serve_constraint.as_posix()} vllm",
        )
    )

    if eval_too:
        eval_constraint = repo_root / "constraints" / "eval.txt"
        checks.extend(check_constraint_file(eval_constraint, "eval"))
        checks.append(
            check_executable(
                "lm-eval",
                f'uv pip install -c {eval_constraint.as_posix()} "lm-eval[api]"',
            )
        )

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the constraints/eval.txt + lm-eval checks (use for serve-only environments).",
    )
    args = parser.parse_args(argv)

    checks = run_checks(eval_too=not args.no_eval)
    for check in checks:
        check.emit()

    failed = [c for c in checks if not c.ok]
    print()
    if failed:
        print(f"{len(failed)} of {len(checks)} checks failed.", file=sys.stderr)
        return 1
    print(f"all {len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
