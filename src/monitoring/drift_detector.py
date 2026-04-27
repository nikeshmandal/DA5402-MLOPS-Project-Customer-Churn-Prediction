"""
Data Drift Detection
====================
Compares live inference statistics against training baseline.
Triggers alerts when distributions shift significantly.

Author: Nikesh Kumar Mandal (ID25M805)
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASELINE_PATH = Path("config/baseline_stats.json")
DRIFT_THRESHOLD = 0.05  # p-value threshold for KS test


class DriftDetector:
    """
    Detects data drift by comparing live feature distributions
    against the training baseline using the Kolmogorov-Smirnov test.

    How it works
    ------------
    During training we record the mean, std, and percentiles of every
    numeric feature.  At inference time we collect a window of recent
    predictions and compare their feature distributions against the
    baseline.  If the KS test p-value drops below 0.05 it means the
    distributions are statistically different — a sign of drift.

    Why does drift matter?
    ----------------------
    A model trained in January may become stale by June if customer
    behaviour changes.  Drift detection is the early-warning system.
    """

    def __init__(self, baseline_path: Path = BASELINE_PATH):
        self.baseline_path = baseline_path
        self.baseline: dict = {}
        self._load_baseline()

    def _load_baseline(self):
        if self.baseline_path.exists():
            with open(self.baseline_path) as f:
                self.baseline = json.load(f)
            logger.info("Baseline statistics loaded.")
        else:
            logger.warning("Baseline file not found — drift detection disabled.")

    def detect(self, live_data: dict[str, list]) -> dict:
        """
        Compare live feature windows against baseline.

        Parameters
        ----------
        live_data : dict mapping feature_name → list of recent values.

        Returns
        -------
        dict with per-feature drift status and overall flag.
        """
        if not self.baseline:
            return {"drift_detected": False, "reason": "No baseline available."}

        results = {}
        drift_flags = []

        numeric_baseline = self.baseline.get("numeric", {})
        for feature, values in live_data.items():
            if feature not in numeric_baseline:
                continue
            bl = numeric_baseline[feature]
            # Reconstruct a synthetic baseline sample from recorded stats
            baseline_sample = np.random.normal(
                loc=bl["mean"], scale=max(bl["std"], 1e-6), size=1000
            )
            ks_stat, p_value = stats.ks_2samp(baseline_sample, values)
            drifted = p_value < DRIFT_THRESHOLD
            drift_flags.append(drifted)
            results[feature] = {
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": drifted,
            }
            if drifted:
                logger.warning("DRIFT detected for feature '%s' (p=%.4f)", feature, p_value)

        overall = any(drift_flags)
        return {"drift_detected": overall, "features": results}


if __name__ == "__main__":
    # Quick smoke test
    detector = DriftDetector()
    fake_live = {"tenure": list(np.random.normal(20, 5, 200)),
                 "monthly_charges": list(np.random.normal(80, 15, 200))}
    report = detector.detect(fake_live)
    print(json.dumps(report, indent=2))
