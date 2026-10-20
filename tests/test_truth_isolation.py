"""The truth is hidden from the twin: calibration, forecasting, likelihood and treatment code (and
everything they import inside twin) must not import twin.truth / twin.patients / twin.evaluate or
refer to the truth/ directory. Checked on the syntax tree, not by string search of the source.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "twin"
PROTECTED = ["calibrate", "treat", "forecast", "likelihood", "ga", "library", "observe"]
FORBIDDEN_MODULES = {"twin.truth", "twin.patients", "twin.evaluate"}
FORBIDDEN_NAMES = {"TRUTH_DIR", "read_truth", "list_patients", "PatientTruth", "write_truth"}


def imports_of(tree: ast.AST) -> set[str]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # relative import
                base = "twin" + ("." + base if base else "")
            found.add(base)
            found |= {f"{base}.{a.name}" for a in node.names}
    return found


def violations(source: str) -> list[str]:
    tree = ast.parse(source)
    bad = [m for m in imports_of(tree) if m in FORBIDDEN_MODULES]
    for node in ast.walk(tree):
        if isinstance(node, ast.alias) and node.name in FORBIDDEN_NAMES:
            bad.append(node.name)
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            bad.append(node.id)
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            bad.append(node.attr)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "truth/" in node.value or node.value.strip("/") == "truth":
                bad.append(f"path literal {node.value!r}")
    return bad


def closure(start: str) -> set[str]:
    """Modules of the twin package reachable from `start` through imports."""
    seen, todo = set(), [f"twin.{start}"]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = SRC / (mod.split(".", 1)[1] + ".py")
        if not path.exists():
            continue
        todo += [m for m in imports_of(ast.parse(path.read_text())) if m.startswith("twin.")]
    return {m for m in seen if (SRC / (m.split(".", 1)[1] + ".py")).exists()}


def test_checker_detects_violations():
    assert violations("import twin.truth")
    assert violations("from twin import truth") == ["twin.truth"]
    assert violations("from twin.truth import read_truth")
    assert violations("x = open('truth/dev-001.json')")
    assert violations("from twin.config import TRUTH_DIR")
    assert not violations("from twin.observe import Observations")


def test_protected_modules_and_their_imports_never_touch_the_truth():
    checked = 0
    for name in PROTECTED:
        if not (SRC / f"{name}.py").exists():
            continue
        for mod in closure(name):
            if (
                mod == "twin.config"
            ):  # defines the directory constants; protected code only imports it
                continue
            path = SRC / (mod.split(".", 1)[1] + ".py")
            assert not violations(path.read_text()), (
                f"{mod} (reached from twin.{name}): {violations(path.read_text())}"
            )
            checked += 1
    assert checked > 0
