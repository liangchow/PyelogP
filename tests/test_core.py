import warnings
import pytest
import numpy as np
from pyelogp import Data
from pyelogp.find_pc import FindPc

# ---------------------------------------------------------------------------
# Original tests
# ---------------------------------------------------------------------------

def test_data_initialization():
    """Verify that Data object calculates expected columns."""
    pressure = [10, 20, 40, 80, 160]
    void_ratio = [1.2, 1.1, 1.0, 0.9, 0.8]
    d = Data(pressure, void_ratio)

    # Check if dot access works
    assert len(d.log_p) == 5
    assert isinstance(d.dc, np.ndarray)

def test_data_initialization_w_e0():
    """Verify that Data object calculates expected columns."""
    pressure = [10, 20, 40, 80, 160]
    void_ratio = [1.2, 1.1, 1.0, 0.9, 0.8]
    e0 = 1.21
    d = Data(pressure, void_ratio, e0)

    # Check if dot access works
    assert len(d.log_p) == 5
    assert isinstance(d.dc, np.ndarray)


def test_find_pc_logic():
    """Verify find_pc runs without error on valid input."""
    pressure = [10, 20, 40, 80, 160, 320, 640]
    void_ratio = [1.2, 1.12, 1.03, 0.93, 0.81, 0.68, 0.54]

    d = Data(pressure, void_ratio)
    result = d.find_pc()

    # Check if a result object is returned
    assert result.pc > 0
    assert result.e_pc > 0


def test_insufficient_data_raises_error():
    """Verify that too few points raise a ValueError."""
    pressure = [10, 20]
    void_ratio = [1.2, 1.1]

    d = Data(pressure, void_ratio)
    with pytest.raises(ValueError, match="At least 6 valid loading-curve points are required."):
        d.find_pc()


# ---------------------------------------------------------------------------
# Shared fixtures: shape-preserving datasets scaled to span a "small" range
# (0-10) and a "large" range (0-10,000+), each including pressure points
# between 0 and 1 (with decimals), and each starting at exactly 0.
#
# Both datasets are derived from the same known-good e-log(P) shape used in
# test_find_pc_logic above, just rescaled, so the underlying curvature/knee
# behavior is preserved and find_pc() is expected to converge cleanly.
# ---------------------------------------------------------------------------

_BASE_P = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0])
_BASE_E = [1.20, 1.12, 1.03, 0.93, 0.81, 0.68, 0.54]


@pytest.fixture
def small_range_data():
    """Pressure spans 0 -> 10, including sub-1 decimal points."""
    pressure = [0, 0.05] + (_BASE_P / 64.0).tolist()
    void_ratio = [1.205, 1.203] + _BASE_E
    return pressure, void_ratio


@pytest.fixture
def large_range_data():
    """Pressure spans 0 -> ~10,240, including sub-1 decimal points."""
    pressure = [0, 0.5, 1] + (_BASE_P * 16.0).tolist()
    void_ratio = [1.205, 1.203, 1.201] + _BASE_E
    return pressure, void_ratio


# ---------------------------------------------------------------------------
# Small pressure range (0-10), with decimal points between 0 and 1
# ---------------------------------------------------------------------------

class TestSmallPressureRange:

    def test_initializes_and_preserves_point_count(self, small_range_data):
        pressure, void_ratio = small_range_data
        d = Data(pressure, void_ratio)
        assert len(d.dc) == len(pressure)
        assert isinstance(d.dc, np.ndarray)

    def test_decimal_points_below_one_are_finite_in_loading_curve(self, small_range_data):
        pressure, void_ratio = small_range_data
        d = Data(pressure, void_ratio)
        # every pressure < 1 in the input (0.05, 0.15625, 0.3125, 0.625) must
        # survive into the loading curve with a finite log_p
        sub_one = [p for p in pressure if 0 < p < 1]
        assert len(sub_one) > 0
        assert np.all(np.isfinite(d.lc["log_p"]))
        lc_p = set(np.round(d.lc["p"], 6))
        for p in sub_one:
            assert round(p, 6) in lc_p

    def test_zero_pressure_point_is_retained_via_shift_trick(self, small_range_data):
        pressure, void_ratio = small_range_data
        d = Data(pressure, void_ratio)
        # in the "small range" branch (order <= 2), a leading zero pressure
        # is NOT dropped: log_p[0] is computed via a half-step shift
        # (log10(x[1] / 2)), so it survives into the loading curve with a
        # finite log_p -- unlike the large-range branch (see
        # TestLargePressureRange.test_zero_pressure_point_is_dropped_from_loading_curve)
        assert 0.0 in d.lc["p"]
        assert np.isfinite(d.dc["log_p"][0])

    def test_find_pc_runs_and_returns_physical_result(self, small_range_data):
        pressure, void_ratio = small_range_data
        d = Data(pressure, void_ratio)
        result = d.find_pc()
        assert result.pc > 0
        assert result.pc < max(pressure)
        assert not np.isnan(result.e_pc)
        assert 0 < result.e_pc < void_ratio[0]

    def test_work_is_monotonically_nondecreasing(self, small_range_data):
        pressure, void_ratio = small_range_data
        d = Data(pressure, void_ratio)
        # cumulative volumetric strain work should never decrease for a
        # consolidating (compressing) loading curve
        assert np.all(np.diff(d.lc["work"]) >= -1e-9)


# ---------------------------------------------------------------------------
# Large pressure range (0-10,000+), with decimal points between 0 and 1
# ---------------------------------------------------------------------------

class TestLargePressureRange:

    def test_initializes_and_preserves_point_count(self, large_range_data):
        pressure, void_ratio = large_range_data
        d = Data(pressure, void_ratio)
        assert len(d.dc) == len(pressure)
        assert isinstance(d.dc, np.ndarray)

    def test_decimal_points_below_one_are_finite_in_loading_curve(self, large_range_data):
        pressure, void_ratio = large_range_data
        d = Data(pressure, void_ratio)
        sub_one = [p for p in pressure if 0 < p < 1]
        assert len(sub_one) > 0
        assert np.all(np.isfinite(d.lc["log_p"]))
        lc_p = set(np.round(d.lc["p"], 6))
        for p in sub_one:
            assert round(p, 6) in lc_p

    def test_zero_pressure_point_is_dropped_from_loading_curve(self, large_range_data):
        pressure, void_ratio = large_range_data
        d = Data(pressure, void_ratio)
        assert 0.0 not in d.lc["p"]

    def test_find_pc_runs_and_returns_physical_result(self, large_range_data):
        pressure, void_ratio = large_range_data
        d = Data(pressure, void_ratio)
        result = d.find_pc()
        assert result.pc > 0
        assert result.pc < max(pressure)
        assert not np.isnan(result.e_pc)
        assert 0 < result.e_pc < void_ratio[0]

    def test_large_and_small_range_use_different_log_p_strategies(
        self, small_range_data, large_range_data
    ):
        # documents the order > 2 branch split in Data._preprocess: large
        # range data must NOT use the index-0 zero-shift trick the small
        # range branch uses, since order = floor(log10(rx)) > 2 here
        s_pressure, s_void = small_range_data
        l_pressure, l_void = large_range_data
        d_small = Data(s_pressure, s_void)
        d_large = Data(l_pressure, l_void)
        # small range's first (zero) point gets a finite shifted log_p
        assert np.isfinite(d_small.dc["log_p"][0])
        # large range's first (zero) point is NaN by design (mask = x > 0)
        assert np.isnan(d_large.dc["log_p"][0])


# ---------------------------------------------------------------------------
# Cross-cutting / shared behavior across both ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("range_fixture", ["small_range_data", "large_range_data"])
def test_e0_defaults_to_first_void_ratio(range_fixture, request):
    pressure, void_ratio = request.getfixturevalue(range_fixture)
    d = Data(pressure, void_ratio)
    assert d.e0 == pytest.approx(void_ratio[0])


@pytest.mark.parametrize("range_fixture", ["small_range_data", "large_range_data"])
def test_explicit_e0_overrides_default(range_fixture, request):
    pressure, void_ratio = request.getfixturevalue(range_fixture)
    d = Data(pressure, void_ratio, e0=2.0)
    assert d.e0 == 2.0
    assert d.dc["epsilon"][0] != pytest.approx((void_ratio[0] - void_ratio[0]) / (1 + void_ratio[0]))


@pytest.mark.parametrize("range_fixture", ["small_range_data", "large_range_data"])
def test_mismatched_lengths_raise_value_error(range_fixture, request):
    pressure, void_ratio = request.getfixturevalue(range_fixture)
    with pytest.raises(ValueError, match="same length"):
        Data(pressure, void_ratio[:-1])


@pytest.mark.parametrize("range_fixture", ["small_range_data", "large_range_data"])
def test_negative_pressure_raises_value_error(range_fixture, request):
    pressure, void_ratio = request.getfixturevalue(range_fixture)
    bad_pressure = list(pressure)
    bad_pressure[1] = -bad_pressure[1] - 0.01 if bad_pressure[1] != 0 else -1.0
    with pytest.raises(ValueError, match="non-negative"):
        Data(bad_pressure, void_ratio)


# ---------------------------------------------------------------------------
# Fixed issues (regression tests).
#
# These tests previously pinned buggy behavior (see git history / CHANGELOG).
# The underlying bugs have since been fixed in find_pc.py; these tests now
# pin the corrected behavior so any regression is caught.
# ---------------------------------------------------------------------------

class TestFixedIssues:

    def test_knee_at_last_point_raises_value_error_gracefully(self):
        """
        FIXED: when the detected knee lands on (or one point before) the
        final loading-curve sample, segment 2 of the bilinear fit would
        become empty and np.polyfit raised an unhandled TypeError. This is
        now caught early and raised as a clean, documented ValueError.
        """
        x = np.array([1.0, 2, 4, 8, 16, 32])
        y = np.array([0, 0.5, 1.5, 3.5, 8, 20.0])
        knee = 32.0  # forces knee_idx = len(x) - 1
        with pytest.raises(ValueError, match="Not enough points in segment 2"):
            FindPc._bilinear_w_knee(x, y, knee, threshold=None)

    def test_threshold_of_exactly_zero_is_respected(self):
        """
        FIXED: `if threshold else len(x)` treated a legitimate
        threshold of 0.0 the same as no threshold (falsy check instead of
        `is not None`). Now `threshold=0.0` is honored and produces a
        different, tighter segment-2 boundary than `threshold=None`.
        """
        x = np.array([1.0, 2, 4, 8, 16, 32, 64, 128])
        y = np.array([0, 0.5, 1.5, 3.5, 8, 20, 50, 130.0])
        knee = 8.0
        result_zero = FindPc._bilinear_w_knee(x, y, knee, threshold=0.0)
        result_none = FindPc._bilinear_w_knee(x, y, knee, threshold=None)
        assert not np.allclose(result_zero.seg2["x"], result_none.seg2["x"])
        # threshold=0.0 forces the earliest possible split (best_k + 3)
        assert result_zero.seg2["x"][-1] == pytest.approx(64.0)
        assert result_none.seg2["x"][-1] == pytest.approx(128.0)

    def test_negative_pc_is_validated_and_recorded_as_warning(self):
        """
        FIXED: when the bilinear intersection extrapolates to a negative
        pressure, find_pc() now validates `pc` before computing `e_pc`,
        avoiding the invalid `log10` call and recording the issue in
        `result.warnings` instead of silently returning a corrupted,
        unexplained `nan`.
        """
        pressure = [0, 0.5, 1, 10, 100, 500, 1000, 2500, 5000, 10000]
        void_ratio = [1.50, 1.498, 1.495, 1.48, 1.40, 1.10, 0.85, 0.70, 0.60, 0.52]
        d = Data(pressure, void_ratio)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = d.find_pc()
        assert result.pc < 0
        assert np.isnan(result.e_pc)
        assert len(result.warnings) == 1
        assert "non-physical" in result.warnings[0]

