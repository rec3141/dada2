"""Error model estimation for DADA2."""

import numpy as np

# Transition row names (matching R's ordering)
TRANS_NAMES = [
    "A2A", "A2C", "A2G", "A2T",
    "C2A", "C2C", "C2G", "C2T",
    "G2A", "G2C", "G2G", "G2T",
    "T2A", "T2C", "T2G", "T2T",
]

# Indices of non-self transitions (12 total)
# Order: A2C,A2G,A2T, C2A,C2G,C2T, G2A,G2C,G2T, T2A,T2C,T2G
_NON_SELF = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]
_SELF = [0, 5, 10, 15]  # A2A, C2C, G2G, T2T

# Mapping: for each base (0-3), which rows are its transitions (in order A,C,G,T)
_BASE_ROWS = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15]}


def _lowess_fit(x, y, weights, span=0.75, degree=2):
    """Weighted LOESS (locally weighted scatterplot smoothing).

    Uses local polynomial regression with tricube kernel.
    Matches R's loess() default: degree=2 (local quadratic), span=0.75.
    """
    n = len(x)
    h = max(int(np.ceil(span * n)), degree + 1)
    y_pred = np.empty(n)

    for i in range(n):
        dists = np.abs(x - x[i])
        idx = np.argsort(dists)[:h]
        max_dist = dists[idx[-1]]
        if max_dist == 0:
            max_dist = 1.0

        # Tricube kernel
        u = dists[idx] / (max_dist * 1.0001)
        w = (1.0 - u ** 3) ** 3
        w *= weights[idx]

        if w.sum() == 0:
            y_pred[i] = y[i]
            continue

        # Weighted polynomial regression: y = a0 + a1*(x-xi) + a2*(x-xi)^2 + ...
        xc = x[idx] - x[i]
        yi = y[idx]
        # Build Vandermonde matrix (centered at x[i])
        V = np.column_stack([xc ** p for p in range(degree + 1)])
        W = np.diag(w)
        try:
            VtW = V.T @ W
            beta = np.linalg.solve(VtW @ V, VtW @ yi)
            y_pred[i] = beta[0]  # value at x[i] (xc=0)
        except np.linalg.LinAlgError:
            y_pred[i] = np.average(yi, weights=w)

    return y_pred


def loess_errfun(trans):
    """Estimate error rates from transition counts using LOESS smoothing.

    Mirrors R's loessErrfun from errorModels.R.

    Args:
        trans: numpy array (16, ncol), transition counts. Rows are transitions
               (A2A,A2C,...,T2T), columns are quality scores.

    Returns:
        numpy array (16, ncol), estimated error rates.
    """
    ncol = trans.shape[1]
    qq = np.arange(ncol, dtype=np.float64)
    est = np.zeros((12, ncol))  # 12 non-self transitions

    MIN_ERROR_RATE = 1e-7
    MAX_ERROR_RATE = 0.25

    idx = 0
    for nti in range(4):  # A, C, G, T
        base_rows = _BASE_ROWS[nti]
        tot = trans[base_rows].sum(axis=0).astype(np.float64)

        for ntj in range(4):
            if nti == ntj:
                continue
            row = nti * 4 + ntj
            errs = trans[row].astype(np.float64)

            # Log10 error rate with pseudocount
            with np.errstate(divide="ignore", invalid="ignore"):
                rlogp = np.log10((errs + 1.0) / tot)
            rlogp[~np.isfinite(rlogp)] = np.nan

            # Find valid (non-NaN) positions
            valid = ~np.isnan(rlogp)
            if valid.sum() < 2:
                est[idx] = MIN_ERROR_RATE
                idx += 1
                continue

            # LOESS fit on valid points
            x_valid = qq[valid]
            y_valid = rlogp[valid]
            w_valid = tot[valid]
            w_valid = np.maximum(w_valid, 1.0)

            pred_valid = _lowess_fit(x_valid, y_valid, w_valid, span=0.75)

            # Predict for all quality scores by extending edges
            pred = np.full(ncol, np.nan)
            pred[valid] = pred_valid

            # Fill NaN edges by repeating boundary values
            first_valid = np.where(valid)[0][0]
            last_valid = np.where(valid)[0][-1]
            pred[:first_valid] = pred[first_valid]
            pred[last_valid + 1:] = pred[last_valid]

            # Back-transform from log10
            est[idx] = 10.0 ** pred
            idx += 1

    # Clamp
    est = np.clip(est, MIN_ERROR_RATE, MAX_ERROR_RATE)

    # Build full 16-row matrix with self-transitions as complement
    # Row order: A2A,A2C,A2G,A2T, C2A,C2C,C2G,C2T, G2A,G2C,G2G,G2T, T2A,T2C,T2G,T2T
    # Non-self est order: A2C(0),A2G(1),A2T(2), C2A(3),C2G(4),C2T(5), G2A(6),G2C(7),G2T(8), T2A(9),T2C(10),T2G(11)
    err = np.zeros((16, ncol))
    err[1] = est[0]   # A2C
    err[2] = est[1]   # A2G
    err[3] = est[2]   # A2T
    err[0] = 1.0 - est[0:3].sum(axis=0)  # A2A

    err[4] = est[3]   # C2A
    err[6] = est[4]   # C2G
    err[7] = est[5]   # C2T
    err[5] = 1.0 - est[3:6].sum(axis=0)  # C2C

    err[8] = est[6]   # G2A
    err[9] = est[7]   # G2C
    err[11] = est[8]  # G2T
    err[10] = 1.0 - est[6:9].sum(axis=0)  # G2G

    err[12] = est[9]   # T2A
    err[13] = est[10]  # T2C
    err[14] = est[11]  # T2G
    err[15] = 1.0 - est[9:12].sum(axis=0)  # T2T

    return err


def noqual_errfun(trans):
    """Error estimation ignoring quality scores (constant across Q)."""
    ncol = trans.shape[1]
    err = np.zeros((16, ncol))
    for nti in range(4):
        base_rows = _BASE_ROWS[nti]
        tot = trans[base_rows].sum()
        if tot == 0:
            for ntj in range(4):
                err[nti * 4 + ntj] = 0.25
            continue
        for ntj in range(4):
            row = nti * 4 + ntj
            rate = trans[row].sum() / tot
            err[row] = rate
    return err


def inflate_err(err, inflation):
    """Inflate error rates by a factor (prevents premature convergence).

    Mirrors R's inflateErr.
    """
    out = err * inflation / (1.0 + (inflation - 1.0) * err)
    return np.clip(out, 0.0, 1.0)


def get_initial_err(ncol=41):
    """Get a maximum-possible initial error matrix (all 1.0, matching R)."""
    err = np.full((16, ncol), 1.0)
    return err
