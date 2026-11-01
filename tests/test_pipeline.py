"""End-to-end smoke test on synthetic data. Run: python tests/test_pipeline.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

with tempfile.TemporaryDirectory() as tmp:
    data, out = Path(tmp) / "data", Path(tmp) / "outputs"
    data.mkdir()
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_fake_data.py"), str(data / "fake.csv")], check=True)
    subprocess.run([sys.executable, str(ROOT / "src" / "run.py"), "--data", str(data), "--out", str(out),
                    "--sims", "500"], check=True, cwd=ROOT / "src")
    expected = ["worlds2026_odds.csv", "quarterfinal_odds.csv", "model_scores.csv", "this_worlds_form.csv",
                "feature_importance.csv", "calibration.csv", "run_info.json"]
    missing = [f for f in expected if not (out / f).exists()]
    assert not missing, missing
    import pandas as pd
    odds = pd.read_csv(out / "worlds2026_odds.csv", index_col=0)
    assert len(odds) == 8 and abs(odds["champion"].sum() - 1) < 1e-6
    json.loads((out / "run_info.json").read_text())
print("pipeline smoke test passed")
