import pytest
import pandas as pd
import numpy as np

from drift_monitor.metrics import (
    PSICalculator,
    KSCalculator,
    ChiSquareCalculator,
    classify_psi_risk,
    classify_ks_risk,
    classify_chisq_risk,
    RiskLevel,
    PSI_THRESHOLDS,
)


class TestPSICalculator:
    def test_calculate_for_numerical_identical_distributions(self):
        calculator = PSICalculator(n_bins=10)
        np.random.seed(42)
        expected = np.random.normal(0, 1, 10000)
        actual = np.random.normal(0, 1, 10000)

        psi, details = calculator.calculate_for_numerical(expected, actual)

        assert psi >= 0
        assert len(details) == 10

    def test_calculate_for_numerical_different_distributions(self):
        calculator = PSICalculator(n_bins=10)
        np.random.seed(42)
        expected = np.random.normal(0, 1, 10000)
        actual = np.random.normal(3, 1, 10000)

        psi, details = calculator.calculate_for_numerical(expected, actual)

        assert psi > 0.1

    def test_calculate_for_numerical_constant_value(self):
        calculator = PSICalculator(n_bins=10)
        expected = np.ones(1000) * 5.0
        actual = np.ones(1000) * 5.0

        psi, details = calculator.calculate_for_numerical(expected, actual)

        assert psi >= 0

    def test_calculate_for_categorical_identical_distributions(self):
        calculator = PSICalculator()
        np.random.seed(42)
        expected = pd.Series(np.random.choice(["A", "B", "C"], 10000, p=[0.3, 0.4, 0.3]))
        actual = pd.Series(np.random.choice(["A", "B", "C"], 10000, p=[0.3, 0.4, 0.3]))

        psi, details = calculator.calculate_for_categorical(expected, actual)

        assert psi >= 0
        assert len(details) == 3

    def test_calculate_for_categorical_different_distributions(self):
        calculator = PSICalculator()
        np.random.seed(42)
        expected = pd.Series(np.random.choice(["A", "B", "C"], 10000, p=[0.8, 0.1, 0.1]))
        actual = pd.Series(np.random.choice(["A", "B", "C"], 10000, p=[0.1, 0.8, 0.1]))

        psi, details = calculator.calculate_for_categorical(expected, actual)

        assert psi > 0.1

    def test_calculate_for_numerical_with_nan(self):
        calculator = PSICalculator(n_bins=10)
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        expected[0] = np.nan
        actual[0] = np.nan

        psi, details = calculator.calculate_for_numerical(expected, actual)

        assert psi >= 0

    def test_calculate_for_categorical_with_nan(self):
        calculator = PSICalculator()
        expected = pd.Series(["A", "B", "C", None, "A"])
        actual = pd.Series(["A", "B", "C", "A", "A"])

        psi, details = calculator.calculate_for_categorical(expected, actual)

        assert psi >= 0

    def test_calculate_for_numerical_empty_array_error(self):
        calculator = PSICalculator(n_bins=10)
        expected = np.array([])
        actual = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            calculator.calculate_for_numerical(expected, actual)

    def test_equal_freq_bins(self):
        calculator = PSICalculator(n_bins=5, bucket_type="equal_freq")
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        psi, details = calculator.calculate_for_numerical(data, data)
        assert len(details) == 5


class TestKSCalculator:
    def test_calculate_identical_distributions(self):
        calculator = KSCalculator()
        np.random.seed(42)
        expected = np.random.normal(0, 1, 10000)
        actual = np.random.normal(0, 1, 10000)

        stat, pvalue = calculator.calculate(expected, actual)

        assert 0 <= stat <= 1
        assert 0 <= pvalue <= 1
        assert pvalue > 0.05

    def test_calculate_different_distributions(self):
        calculator = KSCalculator()
        np.random.seed(42)
        expected = np.random.normal(0, 1, 10000)
        actual = np.random.normal(5, 1, 10000)

        stat, pvalue = calculator.calculate(expected, actual)

        assert stat > 0.5
        assert pvalue < 0.05

    def test_calculate_with_nan(self):
        calculator = KSCalculator()
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        expected[0] = np.nan
        actual[0] = np.nan

        stat, pvalue = calculator.calculate(expected, actual)

        assert 0 <= stat <= 1


class TestChiSquareCalculator:
    def test_calculate_identical_distributions(self):
        calculator = ChiSquareCalculator()
        expected = pd.Series(["A"] * 3000 + ["B"] * 4000 + ["C"] * 3000)
        actual = pd.Series(["A"] * 3000 + ["B"] * 4000 + ["C"] * 3000)

        chi2, pvalue, details = calculator.calculate(expected, actual)

        assert chi2 >= 0
        assert 0 <= pvalue <= 1
        assert pvalue > 0.05
        assert len(details) == 3

    def test_calculate_different_distributions(self):
        calculator = ChiSquareCalculator()
        np.random.seed(42)
        expected = pd.Series(np.random.choice(["A", "B", "C"], 10000, p=[0.8, 0.1, 0.1]))
        actual = pd.Series(np.random.choice(["A", "B", "C"], 10000, p=[0.1, 0.8, 0.1]))

        chi2, pvalue, details = calculator.calculate(expected, actual)

        assert pvalue < 0.05

    def test_calculate_with_nan(self):
        calculator = ChiSquareCalculator()
        expected = pd.Series(["A", "B", "C", None, "A", "B"])
        actual = pd.Series(["A", "B", "C", "A", "A", "B"])

        chi2, pvalue, details = calculator.calculate(expected, actual)

        assert chi2 >= 0


class TestRiskClassification:
    def test_classify_psi_risk_no_drift(self):
        assert classify_psi_risk(0.05) == RiskLevel.NO_DRIFT
        assert classify_psi_risk(0.09) == RiskLevel.NO_DRIFT

    def test_classify_psi_risk_slight_drift(self):
        assert classify_psi_risk(0.1) == RiskLevel.SLIGHT_DRIFT
        assert classify_psi_risk(0.2) == RiskLevel.SLIGHT_DRIFT

    def test_classify_psi_risk_moderate_drift(self):
        assert classify_psi_risk(0.25) == RiskLevel.MODERATE_DRIFT
        assert classify_psi_risk(0.4) == RiskLevel.MODERATE_DRIFT

    def test_classify_psi_risk_severe_drift(self):
        assert classify_psi_risk(0.5) == RiskLevel.SEVERE_DRIFT
        assert classify_psi_risk(1.0) == RiskLevel.SEVERE_DRIFT

    def test_classify_ks_risk(self):
        assert classify_ks_risk(0.1) == RiskLevel.NO_DRIFT
        assert classify_ks_risk(0.25) == RiskLevel.SLIGHT_DRIFT
        assert classify_ks_risk(0.4) == RiskLevel.MODERATE_DRIFT
        assert classify_ks_risk(0.6) == RiskLevel.SEVERE_DRIFT

    def test_classify_chisq_risk(self):
        assert classify_chisq_risk(0.1) == RiskLevel.NO_DRIFT
        assert classify_chisq_risk(0.03) == RiskLevel.SLIGHT_DRIFT
        assert classify_chisq_risk(0.005) == RiskLevel.MODERATE_DRIFT
        assert classify_chisq_risk(0.0001) == RiskLevel.SEVERE_DRIFT
