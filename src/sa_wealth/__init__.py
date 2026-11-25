"""sa-wealth: South African Regulation 28 private wealth & tax-aware Monte Carlo allocator."""

from .assets import (
    ASSETS,
    ASSET_NAMES,
    CMA_TABLE,
    MarketAssumptions,
    default_assumptions,
    load_assumptions,
    sanitise_zac_series,
)
from .montecarlo import (
    DEFAULT_SCENARIOS,
    SimulationConfig,
    SimulationResult,
    StressScenario,
    generate_paths,
    run_stress,
    run_tax_comparison,
    simulate,
)
from .reg28_tax import (
    REG28,
    TAX_2025_26,
    TAX_2026_27,
    Reg28Limits,
    TaxParameters,
    apply_tfsa_limits,
    check_reg28,
    optimize_portfolio,
    project_to_reg28,
)

__version__ = "1.0.0"

__all__ = [
    "ASSETS",
    "ASSET_NAMES",
    "CMA_TABLE",
    "DEFAULT_SCENARIOS",
    "MarketAssumptions",
    "REG28",
    "Reg28Limits",
    "SimulationConfig",
    "SimulationResult",
    "StressScenario",
    "TAX_2025_26",
    "TAX_2026_27",
    "TaxParameters",
    "apply_tfsa_limits",
    "check_reg28",
    "default_assumptions",
    "generate_paths",
    "load_assumptions",
    "optimize_portfolio",
    "project_to_reg28",
    "run_stress",
    "run_tax_comparison",
    "sanitise_zac_series",
    "simulate",
    "__version__",
]
