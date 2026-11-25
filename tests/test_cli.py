import argparse

import pytest

from sa_wealth import __version__
from sa_wealth.cli import build_parser, main, parse_weights


def test_parse_weights_normalises():
    w = parse_weights("JSE_EQ=30,SA_BOND=70")
    assert w == pytest.approx({"JSE_EQ": 0.3, "SA_BOND": 0.7})
    with pytest.raises(argparse.ArgumentTypeError):
        parse_weights("BITCOIN=1")


def test_commands_registered():
    parser = build_parser()
    for cmd in ("simulate", "optimize", "tax-compare", "charts", "stress"):
        assert parser.parse_args([cmd]).command == cmd


def test_simulate_command(capsys, tmp_path):
    report = tmp_path / "plan.md"
    code = main(["simulate", "--paths", "400", "--years", "6", "--target", "800000", "--report", str(report)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Probability of reaching" in out and "CVaR 5%" in out and "Regulation 28 COMPLIANT" in out
    text = report.read_text(encoding="utf-8")
    assert "# Client Wealth Plan" in text and "Tax wrappers compared" in text and "stress tests" in text


def test_simulate_flags_non_compliant_custom_weights(capsys):
    code = main(["simulate", "--paths", "300", "--years", "5", "--weights", "GLB_EQ=90,JSE_EQ=10"])
    out = capsys.readouterr().out
    assert code == 0 and "NOT Regulation 28" in out and "closest compliant" in out


def test_optimize_command(capsys):
    assert main(["optimize", "--target-return", "11"]) == 0
    out = capsys.readouterr().out
    assert "Regulation 28" in out and "offshore cap" in out


def test_tax_compare_command(capsys):
    assert main(["tax-compare", "--paths", "300", "--years", "8", "--tax-year", "2026/27"]) == 0
    out = capsys.readouterr().out
    assert "Tax-Free Savings" in out and "R46,000/yr" in out


def test_stress_command(capsys):
    assert main(["stress", "--paths", "300", "--years", "10", "--wrapper", "DISC"]) == 0
    assert "Sequence-of-Returns Crash" in capsys.readouterr().out


def test_charts_command_writes_four_pngs(tmp_path, capsys):
    assert main(["charts", "--out", str(tmp_path), "--dpi", "40", "--paths", "300", "--years", "10"]) == 0
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == ["fan_chart.png", "reg28_frontier.png", "stress_scenarios.png", "tax_wrappers.png"]
    assert all(p.stat().st_size > 1_000 for p in tmp_path.glob("*.png"))


def test_bad_input_returns_error_code(capsys):
    assert main(["simulate", "--years", "3", "--paths", "300"]) == 2
    assert "horizon" in capsys.readouterr().err


def test_version():
    assert __version__ == "1.0.0"
