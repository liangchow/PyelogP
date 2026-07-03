"""
pyelogp.find_pc
===============
Preconsolidation pressure estimation using bilinear fitting in work-p
space (Becker et al., 1987) with a knee-point detected in e–log(P) space.

Typical usage (via Data)
------------------------
>>> from pyelogp import Data
>>> d = Data.from_csv("data/example.csv")
>>> result = d.find_pc()
>>> print(result.pc)          # preconsolidation pressure

Direct usage
------------
>>> from pyelogp import Data, FindPc
>>> d = Data(pressure=[...], void_ratio=[...])
>>> result = d.find_pc()
>>> print(result.pc)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from kneed import KneeLocator

class BilinearResult:
    """
    Internal result container for :meth:`FindPc._bilinear_w_knee`.
    """
    def __init__(self, x_int, y_int, seg1, seg2, r2_seg1, r2_seg2):
        self.x_int = x_int
        self.y_int = y_int
        self.seg1 = seg1
        self.seg2 = seg2
        self.r2_seg1 = r2_seg1
        self.r2_seg2 = r2_seg2

class PcResult:
    """
    Final output of :meth:`FindPc.run`.

    Attributes
    ----------
    pc : float
        Estimated preconsolidation pressure.
    e_pc : float
        Void ratio at ``pc``. ``nan`` when ``pc`` could not be validated
        (see ``warnings``).
    seg1 : list of tuple
        ``(x, y_fit)`` pairs describing the fitted segment-1 line.
    seg2 : list of tuple
        ``(x, y_fit)`` pairs describing the fitted segment-2 line.
    r2_seg1 : float
        Coefficient of determination (R2) for segment 1.
    r2_seg2 : float
        Coefficient of determination (R2) for segment 2.
    warnings : list of str
        Non-fatal issues encountered while producing this result (e.g. a
        non-physical ``pc``). Empty when none occurred.
    """
    def __init__(self, pc, e_pc, seg1, seg2, r2_seg1, r2_seg2, warnings=None):
        self.pc = pc
        self.e_pc = e_pc
        self.seg1 = seg1
        self.seg2 = seg2
        self.r2_seg1 = r2_seg1
        self.r2_seg2 = r2_seg2
        self.warnings = warnings if warnings is not None else []

# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------
class FindPc:
    """
    Estimate the preconsolidation pressure.

    Algorithm summary
    -----------------
    1. Fit a cubic spline to the primary loading curve in e–log(P) space.
    2. Locate the knee of the e–log(P) curve using :class:`kneed.KneeLocator`.
    3. Perform a bilinear least-squares fit in work–pressure (W–P) space,
       splitting at the detected knee.
    4. Compute the intersection of the two lines → preconsolidation pressure (yield stress).

    Parameters
    ----------
    data : Data
        Pre-processed consolidation dataset.
    n_spline_points : int, optional
        Number of points used when evaluating the cubic spline for knee
        detection.  Default is 50.
    """
    def __init__(self, data, n_spline_points=50):
        self._data = data
        self._n_spline_points = n_spline_points

    def run(self):
        """
        Execute the full analysis pipeline and return a :class:`PcResult`.

        Raises
        ------
        ValueError
            If the dataset is too small or the fitting fails.
        """
        warnings = []
        lc = self._data.lc

        if len(lc) < 6:
            raise ValueError("At least 6 valid loading-curve points are required.")

        cs_fit, cs_curve, log_p_max_d2y = self._cs_profile(
            lc["log_p"], lc["e"], n_points=self._n_spline_points
        )
        p_threshold = np.floor(np.power(10, log_p_max_d2y))

        kl = self._find_knee(lc, cs_curve)
        if kl.knee is None:
            raise ValueError("Could not detect a knee in the e–log(P) curve.")
        knee = np.power(10, kl.knee)

        fit = self._bilinear_w_knee(lc["p"], lc["work"], knee, threshold=p_threshold)

        pc = fit.x_int

        if not np.isfinite(pc) or pc <= 0:
            warnings.append(
                f"Computed pc={pc} is non-physical (must be > 0); "
                "e_pc could not be determined."
            )
            e_pc = np.nan
        else:
            e_pc = self._find_e(lc, cs_fit, pc)

        def _to_pairs(seg):
            return [(x, y) for x, y in zip(seg["x"], seg["y_fit"])]

        return PcResult(
            pc=pc,
            e_pc=e_pc,
            seg1=_to_pairs(fit.seg1),
            seg2=_to_pairs(fit.seg2),
            r2_seg1=fit.r2_seg1,
            r2_seg2=fit.r2_seg2,
            warnings=warnings,
        )

    @staticmethod
    def _r2(y_actual, y_pred):
        """
        Coefficient of determination (R2).
        """
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        return (1.0 - ss_res / ss_tot) if ss_tot != 0 else 1.0

    @staticmethod
    def _find_e(lc, cs, sigma_v):
        """
        Interpolate void ratio at ``sigma_v`` using the cubic spline ``cs``.

        Falls back to linear extrapolation when ``sigma_v`` is below the
        spline's fitted range.

        Parameters
        ----------
        lc : NDArray
            Loading-curve structured array.
        cs : CubicSpline
            Spline fitted in log(P) space.
        sigma_v : float
            Target pressure.

        Returns
        -------
        float
        """
        x = np.log10(sigma_v)
        x_min, x_max = cs.x[0], cs.x[-1]

        if x < x_min:
            # Linear extrapolation from the first two LC points
            p0, p1 = lc["p"][:2]
            e0, e1 = lc["e"][:2]
            slope  = (e1 - e0) / (p1 - p0)
            e      = e0 + slope * (sigma_v - p0)
        else:
            e = cs(np.clip(x, x_min, x_max))

        return np.round(e, 4)

    @staticmethod
    def _cs_profile(x, y, n_points=50):
        """
        Fit a cubic spline and return curvature information.

        Parameters
        ----------
        x : ndarray
            log(P) values of the loading curve.
        y : ndarray
            Void-ratio values.
        n_points : int
            Evaluation density for the dense curve.

        Returns
        -------
        cs : CubicSpline
        cs_curve : ndarray, shape (n_points, 5)
            Columns: x_dense, y_dense
        log_p_max_d2y : float
            log(P) at the location of maximum second derivative.
        """
        order  = np.argsort(x)
        x_fit, y_fit = x[order], y[order]

        cs       = CubicSpline(x_fit, y_fit)
        x_dense  = np.linspace(x_fit.min(), x_fit.max(), n_points)
        y_dense  = cs(x_dense)
        d2y      = cs(x_dense, 2)

        idx_max_d2y    = int(np.argmax(d2y))
        log_p_max_d2y  = x_dense[idx_max_d2y]

        return cs, np.column_stack((x_dense, y_dense)), log_p_max_d2y

    @staticmethod
    def _find_knee(lc, cs_curve, min_len=10):
        """
        Locate the knee of the e–log(P) curve.

        Uses the dense spline curve for small datasets and the raw loading-curve
        data for larger ones.

        Parameters
        ----------
        lc : NDArray
            Loading-curve structured array.
        cs_curve : ndarray
            Dense curve array from :meth:`_cs_profile`.
        min_len : int
            Dataset-length threshold for switching strategy.

        Returns
        -------
        KneeLocator
        """
        x, y = lc["log_p"], lc["e"]

        if len(lc) <= min_len:
            kl = KneeLocator(
                cs_curve[:, 0], cs_curve[:, 1],
                curve="concave", direction="decreasing", online=True
            )
            if kl.knee is not None and kl.knee != cs_curve[:, 0][-1]:
                return kl

        return KneeLocator(x, y, curve="concave", direction="decreasing", online=True)

    @staticmethod
    def _bilinear_w_knee(x, y, knee, threshold=None):
        """
        Bilinear least-squares fit split at the knee point.

        Parameters
        ----------
        x : ndarray
            Pressure values of the loading curve.
        y : ndarray
            Cumulative work values.
        knee : float
            Pressure value of the detected knee (in native pressure units).
        threshold : float, optional
            Upper pressure limit for segment 2 (uses max curvature location).

        Returns
        -------
        BilinearResult

        Raises
        ------
        ValueError
            When segments contain too few points or the lines are parallel.
        """
        knee_idx = int(np.searchsorted(x, knee, side="right") - 1)

        if knee_idx < 1:
            raise ValueError("Not enough points in segment 1.")
        best_k = knee_idx + 1

        if best_k > len(x) - 2:
            raise ValueError(
                "Not enough points in segment 2; the detected knee is too close "
                "to the end of the dataset."
            )

        end_k = min(len(x), max(int(np.searchsorted(x, threshold, side="right")) if threshold is not None else len(x), best_k + 3))

        coeffs1 = np.polyfit(x[:best_k], y[:best_k], deg=1)
        coeffs2 = np.polyfit(x[best_k:end_k], y[best_k:end_k], deg=1)

        dslope = coeffs1[0] - coeffs2[0]

        if np.isclose(dslope, 0, atol=1e-10):
            raise ValueError("Lines are parallel; cannot determine intersection.")

        x_int = (coeffs2[1] - coeffs1[1]) / dslope
        
        return BilinearResult(
            x_int=np.round(x_int, 2), y_int=np.polyval(coeffs1, x_int),
            seg1={"x": np.array([x[0], x_int]), "y_fit": np.polyval(coeffs1, np.array([x[0], x_int]))},
            seg2={"x": np.array([x_int, x[end_k - 1]]), "y_fit": np.polyval(coeffs2, np.array([x_int, x[end_k - 1]]))},
            r2_seg1=np.round(FindPc._r2(y[:best_k], np.polyval(coeffs1, x[:best_k])), 6),
            r2_seg2=np.round(FindPc._r2(y[best_k:end_k], np.polyval(coeffs2, x[best_k:end_k])), 6),
        )

