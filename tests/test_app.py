"""Static checks on the Streamlit app (no Streamlit install needed).

Streamlit's "magic" renders any bare expression statement. A tuple such as ``st.subheader("x"), st.pyplot(fig)``
evaluates to a tuple of DeltaGenerators and is drawn as a help box, so every render call must be its own statement
on its own line.
"""

import ast
from collections import Counter
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
RENDER_CALLS = {"subheader", "pyplot", "dataframe"}


@pytest.fixture(scope="module")
def tree():
    return ast.parse(APP.read_text())


def test_app_parses_and_has_five_tabs(tree):
    src = APP.read_text()
    assert "st.tabs(" in src and src.count("with tab") == 5
    assert "5 | Living Annuity drawdown" in src


def test_every_bare_expression_statement_is_a_single_call_or_docstring(tree):
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr):
            v = node.value
            is_docstring = isinstance(v, ast.Constant) and isinstance(v.value, str)
            if not (isinstance(v, ast.Call) or is_docstring):
                bad.append((node.lineno, type(v).__name__))
    assert bad == [], f"bare non-call expression statements would be rendered by Streamlit magic: {bad}"


def test_no_statement_level_tuple_ternary_or_boolean_expressions_with_render_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Tuple, ast.IfExp, ast.BoolOp)):
            pytest.fail(f"line {node.lineno}: expression statement of type {type(node.value).__name__}")


def test_each_subheader_pyplot_dataframe_call_is_on_its_own_line(tree):
    per_line = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in RENDER_CALLS:
            per_line[node.lineno] += 1
    assert per_line, "expected render calls in the app"
    assert max(per_line.values()) == 1, f"several render calls share a line: {[l for l, n in per_line.items() if n > 1]}"


def test_render_calls_are_never_nested_inside_other_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in list(node.args) + [k.value for k in node.keywords]:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in RENDER_CALLS:
                        if isinstance(sub.func.value, ast.Name) and sub.func.value.id in {"st", "sb"}:
                            pytest.fail(f"line {sub.lineno}: st.{sub.func.attr} nested inside another call")


def test_render_calls_are_statements(tree):
    """st.subheader / st.pyplot / st.dataframe are only ever used as standalone statements."""
    stmt_calls = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in RENDER_CALLS:
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "st":
                assert id(node) in stmt_calls, f"line {node.lineno}: st.{node.func.attr} result is used inside an expression"


def test_new_features_are_wired_in():
    src = APP.read_text()
    for needle in (
        "Gross annual income (ZAR, optional)",
        "Smart Contribution Split",
        "plot_smart_split",
        "Initial retirement capital (ZAR)",
        "Desired real monthly income",
        "Retirement horizon (years)",
        "plot_drawdown_fan",
        "ceiling warning",
    ):
        assert needle in src, needle
