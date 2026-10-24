"""Targeted invariance/monotonicity checks for the separate payment-only profile."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_model import (BEHAVIOR_FEATURES, CONSTRAINTS, STABILITY, behavior_values,
                            counterfactual_checks, matrix, parameters, pool)


def row(late=0.0, failed=0.0, count=4, successful=4):
    return {"features": {"days_to_expiry": 3.0, "renewal_cycles_90d": count,
                         "successful_renewals_90d": successful, "failed_cycle_share_90d": failed,
                         "late_cycle_share_90d": late, "mean_paid_delay_hours_90d": 24*late,
                         "max_paid_delay_hours_90d": 48*late, "days_since_successful_renewal": 27,
                         "recoveries_90d": int(late > 0), "has_renewal_history": int(successful > 0),
                         "history_90d_complete": 1}, "label": {"value": int(late >= .5)},
            "cycleWeight": 1.0}


class BehaviorFeatureTests(unittest.TestCase):
    def test_whitelist_does_not_include_money_tariff_or_age_proxy(self):
        original = row()
        changed = copy.deepcopy(original)
        changed["features"].update(tariff_code="EXPENSIVE", price_minor=999999,
                                   monthly_revenue_minor=888888, portfolio_monthly_revenue_minor=777777,
                                   portfolio_change_30d_minor=123456, account_age_days=12345,
                                   service_age_days=12345, live_vps_count=999)
        self.assertEqual(matrix([original]), matrix([changed]))
        self.assertEqual(set(BEHAVIOR_FEATURES), set(behavior_values(original)))

    def test_verified_stability_only_when_all_observed_cycles_are_paid(self):
        self.assertEqual(behavior_values(row(.25))[STABILITY], .75)
        self.assertIsNone(behavior_values(row(.25, count=4, successful=3))[STABILITY])
        self.assertIsNone(behavior_values(self.no_history())[STABILITY])

    @staticmethod
    def no_history():
        result = row(count=0, successful=0)
        for name in ("late_cycle_share_90d", "failed_cycle_share_90d", "mean_paid_delay_hours_90d",
                     "max_paid_delay_hours_90d", "days_since_successful_renewal"):
            result["features"][name] = None
        return result

    def test_missing_history_remains_unknown(self):
        values = matrix([self.no_history()])[0]
        for name in ("late_cycle_share_90d", "failed_cycle_share_90d", STABILITY):
            self.assertTrue(math.isnan(values[BEHAVIOR_FEATURES.index(name)]))

    def test_constraints_do_not_declare_any_history_automatically_safe(self):
        self.assertNotIn("has_renewal_history", CONSTRAINTS)
        self.assertEqual(CONSTRAINTS[STABILITY], -1)
        for name in ("failed_cycle_share_90d", "late_cycle_share_90d",
                     "mean_paid_delay_hours_90d", "max_paid_delay_hours_90d"):
            self.assertEqual(CONSTRAINTS[name], 1)


@unittest.skipUnless(importlib.util.find_spec("catboost"), "CatBoost not installed")
class BehaviorModelSmokeTests(unittest.TestCase):
    def test_fixed_model_preserves_price_invariance_and_monotone_known_coordinates(self):
        from catboost import CatBoostClassifier
        rows = [row(late=i/4, failed=j/4) for i in range(5) for j in range(5)]
        model = CatBoostClassifier(**{**parameters(), "iterations": 30, "depth": 3,
                                     "early_stopping_rounds": None, "use_best_model": False})
        model.fit(pool(rows), verbose=False)
        result = counterfactual_checks(model, rows)
        self.assertEqual(result["excludedAttributesMaximumProbabilityDifference"], 0)
        self.assertTrue(all(v >= -1e-12 for v in result["minimumRiskChangeWhenWorseningOneKnownCoordinate"].values()))


if __name__ == "__main__":
    unittest.main()
