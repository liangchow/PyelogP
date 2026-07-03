"""
pyelogp.data
============
Data ingestion and pre-processing for consolidation test analysis.

Typical usage
-------------
>>> from pyelogp import Data
>>> d = Data(
    pressure=[59, 90, 120, 150, 165, 172, 184, 222, 300, 400], 
    void_ratio=[2.115, 2.113, 2.098, 2.083, 2.055, 2, 1.8, 1.5, 1.3, 1.193])
>>> d.log_p          # log10 of pressure (1-D ndarray)
>>> result = d.find_pc()
>>> print(result.pc)
"""

import numpy as np

DATA_DTYPE = np.dtype([
    ("idx",     np.intp),
    ("p",       np.float64),
    ("log_p",   np.float64),
    ("e",       np.float64),
    ("epsilon", np.float64),
    ("dwork",   np.float64),
    ("work",    np.float64),
])

class Data:
    """
    Container and pre-processor for a consolidation test dataset.

    Parameters
    ----------
    pressure : array-like
        Applied vertical effective stress values (any consistent pressure unit).
        Must be non-negative. Zero is allowed and is handled internally.
    void_ratio : array-like
        Measured void ratios corresponding to each pressure step.
    e0 : float, optional
        Initial void ratio at the start of the test.  Defaults to
        ``void_ratio[0]`` when not supplied.

    Attributes (available after construction)
    -----------------------------------------
    dc : NDArray
        Full structured array for the entire dataset (all loading/unloading
        steps), fields: idx, p, log_p, e, epsilon, dwork, work.
    lc : NDArray
        Loading-curve only; unloading–reloading rows removed and rows with
        non-finite log_p dropped.

    Column accessors (dot-access shortcuts on `dc`)
    ------------------------------------------------
    p, log_p, e, epsilon, dwork, work
        Each returns the corresponding column of ``dc`` as a plain ndarray.
    """

    def __init__(self, pressure, void_ratio, e0=None):
        x = np.asarray(pressure, dtype=np.float64)
        y = np.asarray(void_ratio, dtype=np.float64)

        if x.shape != y.shape:
            raise ValueError("pressure and void_ratio must have the same length.")
        if np.any(x < 0):
            raise ValueError("All pressure values must be non-negative.")
        
        self.e0 = float(y[0]) if e0 is None else float(e0)
        self.dc = self._preprocess(x, y, self.e0)
        self.lc = self._remove_loginf_rl_unl_rows(self.dc)

    def __getattr__(self, name):
        if name in DATA_DTYPE.names:
            return self.dc[name]
        raise AttributeError(f"'Data' object has no attribute '{name}'")

    def find_pc(self, **kwargs):
        from .find_pc import FindPc
        return FindPc(self, **kwargs).run()

    @classmethod
    def from_csv(cls, path, pressure_col=0, void_ratio_col=1, delimiter=",", e0=None):
        """
        Construct a Data instance from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file (e.g. ``"data/example.csv"``).
        pressure_col : int, optional
            Column index for pressure values. Default is 0 (first column).
        void_ratio_col : int, optional
            Column index for void-ratio values. Default is 1 (second column).
        delimiter : str, optional
            Field delimiter. Default is ``","``.
        e0 : float, optional
            Initial void ratio. Defaults to ``void_ratio[0]`` when not supplied.

        Returns
        -------
        Data

        Examples
        --------
        >>> d = Data.from_csv("data/example.csv")
        >>> d = Data.from_csv("data/example.csv", pressure_col=0, void_ratio_col=1)
        """
        raw = np.loadtxt(path, delimiter=delimiter)

        if raw.ndim != 2:
            raise ValueError(
                f"Expected a 2-D array from '{path}', got shape {raw.shape}. "
                "Check your delimiter settings."
            )

        n_cols = raw.shape[1]
        for col_name, col_idx in (("pressure_col", pressure_col), ("void_ratio_col", void_ratio_col)):
            if col_idx >= n_cols:
                raise ValueError(
                    f"{col_name}={col_idx} is out of range — file only has {n_cols} column(s)."
                )

        return cls(
            pressure=raw[:, pressure_col],
            void_ratio=raw[:, void_ratio_col],
            e0=e0,
        )

    @staticmethod
    def vol_strain_work(p, epsilon):
        """
        Compute incremental and cumulative volumetric strain work.
        """
        dwork = 0.5 * (epsilon[1:] - epsilon[:-1]) * (p[1:] + p[:-1])
        dwork = np.hstack((0.0, dwork))
        work  = np.cumsum(dwork)
        return dwork, work

    @staticmethod
    def _preprocess(x, y, e0):
        """
        Build the internal structured array from raw pressure / void-ratio arrays.
        """
        if e0 is None:
            e0 = float(y[0])

        n = len(x)
        if n == 0:
            return np.array([], dtype=DATA_DTYPE)

        # Check order of magnitude of p
        rx    = float(np.max(x) - np.min(x))
        order = int(np.floor(np.log10(rx))) if rx > 0 else 0

        log_p = np.full(n, np.nan, dtype=np.float64)

        if order > 2:
            # Large range: treat 0 as NaN (filtered later); keep 0 < x ≤ 1 as-is
            mask = x > 0
            log_p[mask] = np.log10(x[mask])
        else:
            # Small range: shift zero by half the next step
            if x[0] == 0:
                c = np.min(x[x > 0]) / 2
                log_p = np.log10(np.where(x > 0, x, np.nan))
                log_p[0] = np.log10(c)
            else:
                log_p = np.log10(x)

        epsilon        = (e0 - y) / (1.0 + e0)
        dwork, work    = Data.vol_strain_work(x, epsilon)

        data            = np.empty(n, dtype=DATA_DTYPE)
        data["idx"]     = np.arange(n)
        data["p"]       = x
        data["log_p"]   = log_p
        data["e"]       = np.round(y, 4)
        data["epsilon"] = np.round(epsilon, 5)
        data["dwork"]   = dwork
        data["work"]    = work

        return data

    @staticmethod
    def _remove_loginf_rl_unl_rows(data):
        """
        Strip unloading–reloading rows and non-finite log_p entries.

        A row is considered part of the primary loading curve when its
        pressure strictly exceeds all previously seen pressure values.
        """
        max_p   = -np.inf
        loading = []

        for i, row in enumerate(data):
            if row["p"] > max_p:
                max_p = float(row["p"])
                loading.append(i)

        result = data[loading]
        mask   = np.isfinite(result["log_p"])
        return result[mask]
