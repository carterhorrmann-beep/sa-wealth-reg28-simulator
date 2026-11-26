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
    for cmd in ("simulate", "optimize", "tax-compare", "charts", "stress", "smart-split", "drawdown"):
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


def test_charts_command_writes_six_pngs(tmp_path, capsys):
    assert main(["charts", "--out", str(tmp_path), "--dpi", "40", "--paths", "300", "--years", "10"]) == 0
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == [
        "fan_chart.png",
        "living_annuity.png",
        "reg28_frontier.png",
        "smart_split.png",
        "stress_scenarios.png",
        "tax_wrappers.png",
    ]
    assert all(p.stat().st_size > 1_000 for p in tmp_path.glob("*.png"))


def test_bad_input_returns_error_code(capsys):
    assert main(["simulate", "--years", "3", "--paths", "300"]) == 2
    assert "horizon" in capsys.readouterr().err


def test_version():
    assert __version__ == "2.0.0"


def test_smart_split_command(capsys):
    assert main(["smart-split", "--budget", "20000", "--income", "800000", "--years", "10", "--paths", "300"]) == 0
    out = capsys.readouterr().out
    assert "SARS 2026/27 Smart Contribution Split" in out
    assert "Marginal rate 39%" in out
    assert "Tax-optimal monthly split" in out
    assert "Blended vs Discretionary only" in out


def test_smart_split_no_reinvest_and_tax_year(capsys):
    assert main(["smart-split", "--income", "500000", "--years", "8", "--paths", "200", "--no-reinvest", "--tax-year", "2025/26"]) == 0
    out = capsys.readouterr().out
    assert "SARS 2025/26" in out
    assert "refunds are spent" in out


def test_drawdown_command(capsys):
    assert main(["drawdown", "--capital", "8000000", "--monthly-income", "40000", "--years", "25", "--paths", "300"]) == 0
    out = capsys.readouterr().out
    assert "Living Annuity drawdown" in out
    assert "Portfolio survival to age 90" in out
    assert "Median terminal bequest" in out
    assert "Year-1 Retirement Equity Crash" in out


def test_drawdown_command_flags_income_above_ceiling(capsys):
    assert main(["drawdown", "--capital", "2000000", "--monthly-income", "50000", "--paths", "200", "--portfolio", "aggressive-offshore"]) == 0
    out = capsys.readouterr().out
    assert "exceeds the 17.5% ceiling" in out
    assert "do not apply to a Living Annuity" in out


def test_drawdown_bad_portfolio_returns_error_code(capsys):
    assert main(["drawdown", "--portfolio", "no-such-portfolio", "--paths", "200"]) != 0
