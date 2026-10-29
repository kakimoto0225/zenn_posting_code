from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "results" / "simulation"
FIGURES = OUTPUTS / "figures"


def ensure_directories() -> None:
    for path in (OUTPUTS, OUTPUTS / "runs", FIGURES):
        path.mkdir(parents=True, exist_ok=True)
