from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence
import numpy as np
import pandas as pd
try:
    from src.features.extractor import BaseTransformer
    from src.utils.eye_utils import _split_dataframe
except Exception:
    BaseTransformer = object

    def _split_dataframe(X: pd.DataFrame, pk: list[str]):
        return [('all', X)]
_EPS = 1e-12
_LN2 = float(np.log(2.0))

def _log_base(x: np.ndarray | float, base: float) -> np.ndarray | float:
    """Logarithm in an arbitrary base (supports scalar/ndarray)."""
    if base == 2.0:
        return np.log(x) / _LN2
    return np.log(x) / np.log(base)

def _safe_prob(p: np.ndarray, eps: float=_EPS) -> np.ndarray:
    """Normalize to probabilities and clip."""
    p = np.asarray(p, dtype=float)
    s = float(np.sum(p))
    if s <= 0:
        return np.array([], dtype=float)
    p = p / (s + eps)
    return np.clip(p, eps, 1.0)

def _as_probabilities_from_counts(counts: np.ndarray, eps: float=_EPS) -> np.ndarray:
    c = np.asarray(counts, dtype=float).ravel()
    total = float(np.sum(c))
    if total <= 0:
        return np.array([], dtype=float)
    p = c / (total + eps)
    p = np.clip(p, eps, 1.0)
    return p

def _q_log(x: np.ndarray | float, q: float, base: float=2.0) -> np.ndarray | float:
    """Tsallis q-logarithm, scaled to requested base."""
    if np.isclose(q, 1.0):
        return _log_base(x, base)
    return (x ** (1.0 - q) - 1.0) / ((1.0 - q) * np.log(base))

def _q_exp(u: np.ndarray | float, q: float) -> np.ndarray | float:
    """Tsallis q-exponential (unscaled)."""
    if np.isclose(q, 1.0):
        return np.exp(u)
    z = 1.0 + (1.0 - q) * u
    return np.where(z > 0, z ** (1.0 / (1.0 - q)), 0.0)

def _format_param(v: float | int) -> str:
    """Stable parameter formatting for feature names."""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if np.isinf(v):
        return 'inf'
    s = f'{float(v):.6g}'
    s = s.replace('-', 'm').replace('.', 'p')
    return s

def renyi_entropy(p: np.ndarray, alpha: float, base: float=2.0, eps: float=_EPS) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > eps]
    if p.size == 0:
        return 0.0
    if np.isclose(alpha, 1.0):
        return float(-np.sum(p * _log_base(p + eps, base)))
    if np.isinf(alpha):
        return float(-_log_base(np.max(p), base))
    s = float(np.sum(p ** alpha))
    return float(1.0 / (1.0 - alpha) * _log_base(s + eps, base))

def tsallis_entropy(p: np.ndarray, q: float, base: float=2.0, eps: float=_EPS) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > eps]
    if p.size == 0:
        return 0.0
    if np.isclose(q, 1.0):
        return float(-np.sum(p * _log_base(p + eps, base)))
    s = float(np.sum(p ** q))
    return float((1.0 - s) / (q - 1.0) / np.log(base))

def kaniadakis_entropy(p: np.ndarray, kappa: float, base: float=2.0, eps: float=_EPS) -> float:
    """Kaniadakis κ-entropy.

    S_kappa(p) = - sum_i p_i ln_kappa(p_i)
    ln_kappa(x) = (x^k - x^{-k}) / (2k)
    """
    p = np.asarray(p, dtype=float)
    p = p[p > eps]
    if p.size == 0:
        return 0.0
    if np.isclose(kappa, 0.0):
        return float(-np.sum(p * _log_base(p + eps, base)))
    x = np.clip(p, eps, 1.0)
    ln_k = (x ** kappa - x ** (-kappa)) / (2.0 * kappa)
    return float(-np.sum(x * ln_k) / np.log(base))

def escort_distribution(p: np.ndarray, gamma: float, eps: float=_EPS) -> np.ndarray:
    """Escort distribution: p_i^gamma / sum p^gamma."""
    p = _safe_prob(p, eps=eps)
    if p.size == 0:
        return p
    w = p ** gamma
    return _safe_prob(w, eps=eps)
BoundsType = Literal['minmax', 'unit'] | tuple[float, float, float, float]

def discretize_spatial_states(coords: np.ndarray, grid_size: int, bounds: BoundsType='minmax') -> np.ndarray:
    """Map (x,y) coords to integer states on a grid_size x grid_size grid."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError('coords must be shape (n, 2)')
    if grid_size <= 1:
        raise ValueError('grid_size must be > 1')
    if bounds == 'unit':
        xmin, xmax, ymin, ymax = (0.0, 1.0, 0.0, 1.0)
    elif bounds == 'minmax':
        xmin, xmax = (float(np.min(coords[:, 0])), float(np.max(coords[:, 0])))
        ymin, ymax = (float(np.min(coords[:, 1])), float(np.max(coords[:, 1])))
    else:
        xmin, xmax, ymin, ymax = map(float, bounds)
    if np.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if np.isclose(ymin, ymax):
        ymax = ymin + 1.0
    x = np.clip(coords[:, 0], xmin, xmax)
    y = np.clip(coords[:, 1], ymin, ymax)
    xn = (x - xmin) / (xmax - xmin)
    yn = (y - ymin) / (ymax - ymin)
    xi = np.minimum((xn * grid_size).astype(int), grid_size - 1)
    yi = np.minimum((yn * grid_size).astype(int), grid_size - 1)
    return xi * grid_size + yi

def states_from_dataframe(X: pd.DataFrame, mode: Literal['spatial', 'aoi'], *, x: Optional[str], y: Optional[str], aoi: Optional[str], grid_size: int, bounds: BoundsType) -> tuple[np.ndarray, int]:
    """Return (states, n_states) from a fixation dataframe."""
    if mode == 'aoi':
        if aoi is None or aoi not in X.columns:
            raise ValueError("AOI column must be provided for mode='aoi'")
        codes, uniques = pd.factorize(X[aoi], sort=False)
        n_states = int(len(uniques))
        return (codes.astype(int), n_states)
    if mode == 'spatial':
        if x is None or y is None:
            raise ValueError("x and y must be provided for mode='spatial'")
        coords = X[[x, y]].to_numpy(dtype=float)
        states = discretize_spatial_states(coords, grid_size=grid_size, bounds=bounds)
        n_states = int(grid_size * grid_size)
        return (states.astype(int), n_states)
    raise ValueError(f'Unknown mode: {mode}')

def state_counts(states: np.ndarray, n_states: int) -> np.ndarray:
    s = np.asarray(states, dtype=int)
    if s.size == 0:
        return np.zeros((n_states,), dtype=float)
    return np.bincount(s, minlength=int(n_states)).astype(float)

def transition_counts(states: np.ndarray, n_states: int) -> np.ndarray:
    s = np.asarray(states, dtype=int)
    M = np.zeros((n_states, n_states), dtype=float)
    if s.size < 2:
        return M
    for i, j in zip(s[:-1], s[1:]):
        if 0 <= i < n_states and 0 <= j < n_states:
            M[i, j] += 1.0
    return M

def transition_matrix_from_counts(C: np.ndarray, smoothing: float=0.0, eps: float=_EPS) -> np.ndarray:
    """Row-stochastic transition matrix from counts with optional Dirichlet smoothing."""
    C = np.asarray(C, dtype=float)
    if smoothing > 0:
        C = C + float(smoothing)
    row_sum = C.sum(axis=1, keepdims=True)
    P = np.zeros_like(C, dtype=float)
    mask = row_sum.squeeze() > 0
    P[mask] = C[mask] / (row_sum[mask] + eps)
    if smoothing == 0.0:
        for i in range(P.shape[0]):
            if not mask[i]:
                P[i, i] = 1.0
    return np.clip(P, eps, 1.0)

def stationary_distribution(P: np.ndarray, max_iter: int=10000, tol: float=1e-12) -> np.ndarray:
    """Power iteration to approximate stationary distribution of a row-stochastic matrix."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    pi = np.full((n,), 1.0 / n, dtype=float)
    for _ in range(max_iter):
        new = pi @ P
        if np.linalg.norm(new - pi, ord=1) < tol:
            pi = new
            break
        pi = new
    pi = _safe_prob(pi)
    return pi if pi.size else np.full((n,), 1.0 / n, dtype=float)

class EntropyFeatureBase(ABC, BaseTransformer):
    """Small self-contained transformer base (avoids heavy optional deps)."""

    def __init__(self, x: str=None, y: str=None, t: str=None, duration: str=None, aoi: str=None, pk: list[str]=None, return_df: bool=True, ignore_errors: bool=True):
        super().__init__(x=x, y=y, t=t, duration=duration, aoi=aoi, pk=pk, return_df=return_df)
        self.ignore_errors = ignore_errors
        self._feature_names: Optional[list[str]] = None

    def _check_init(self, X_len: int):
        assert X_len != 0, 'Error: there are no fixations'

    def get_feature_names_out(self, input_features=None) -> list[str]:
        if self._feature_names is not None:
            return list(self._feature_names)
        return ['feature']

    @abstractmethod
    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        if X.shape[0] == 0:
            if self.ignore_errors:
                names = self.get_feature_names_out()
                if self.pk is None:
                    out = pd.DataFrame([[np.nan] * len(names)], columns=names, index=['all'])
                else:
                    out = pd.DataFrame(columns=names)
                return out if self.return_df else out.values
            self._check_init(X_len=X.shape[0])
        try:
            self._check_init(X_len=X.shape[0])
        except (AssertionError, RuntimeError):
            if self.ignore_errors:
                names = self.get_feature_names_out()
                if self.pk is None:
                    out = pd.DataFrame([[np.nan] * len(names)], columns=names, index=['all'])
                else:
                    X_split = _split_dataframe(X, self.pk)
                    group_names = [g for g, _ in X_split]
                    out = pd.DataFrame([[np.nan] * len(names) for _ in group_names], columns=names, index=group_names)
                return out if self.return_df else out.values
            raise
        group_names: list[str] = []
        gathered: list[list[float]] = []
        colnames: list[str] = []
        if self.pk is None:
            group_names.append('all')
            try:
                names, vals = self.calculate_features(X)
                colnames = names
                gathered.append(vals)
            except Exception:
                if not self.ignore_errors:
                    raise
                colnames = self.get_feature_names_out()
                gathered.append([np.nan] * len(colnames))
        else:
            X_split = _split_dataframe(X, self.pk)
            for gname, Xi in X_split:
                group_names.append(gname)
                try:
                    names, vals = self.calculate_features(Xi)
                    colnames = names
                    gathered.append(vals)
                except Exception:
                    if not self.ignore_errors:
                        raise
                    names = self.get_feature_names_out()
                    colnames = names
                    gathered.append([np.nan] * len(names))
        out_df = pd.DataFrame(gathered, columns=colnames, index=group_names)
        return out_df if self.return_df else out_df.values
EntropyFamily = Literal['shannon', 'renyi', 'tsallis', 'kaniadakis']

def _entropy_of_distribution(p: np.ndarray, family: EntropyFamily, param: float, base: float) -> float:
    if family == 'shannon':
        return renyi_entropy(p, alpha=1.0, base=base)
    if family == 'renyi':
        return renyi_entropy(p, alpha=float(param), base=base)
    if family == 'tsallis':
        return tsallis_entropy(p, q=float(param), base=base)
    if family == 'kaniadakis':
        return kaniadakis_entropy(p, kappa=float(param), base=base)
    raise ValueError(f'Unknown family: {family}')

class StateDistributionEntropy(EntropyFeatureBase):
    """Entropy of a state distribution p(state) from scanpath.

    Parameters
    ----------
    mode: "spatial" uses (x,y) discretized to grid; "aoi" uses AOI categorical states.
    family: "shannon" | "renyi" | "tsallis" | "kaniadakis"
    spectrum: list of params for the family (alphas/qs/kappas). Ignored for shannon.
    smoothing: additive smoothing on state counts before normalization.
    escort_gamma: if set, applies escort reweighting p -> p^(gamma) / sum p^gamma before entropy.
    """

    def __init__(self, *, mode: Literal['spatial', 'aoi']='spatial', family: EntropyFamily='shannon', params: Sequence[float]=(1.0,), escort_gamma: Optional[float]=None, grid_size: int=10, bounds: BoundsType='minmax', smoothing: float=0.0, base: float=2.0, x: str=None, y: str=None, aoi: str=None, pk: list[str]=None, return_df: bool=True, ignore_errors: bool=True, feature_prefix: str='state_entropy'):
        super().__init__(x=x, y=y, aoi=aoi, pk=pk, return_df=return_df, ignore_errors=ignore_errors)
        self.mode = mode
        self.family = family
        self.params = tuple((float(p) for p in params))
        self.escort_gamma = escort_gamma
        self.grid_size = int(grid_size)
        self.bounds = bounds
        self.smoothing = float(smoothing)
        self.base = float(base)
        self.feature_prefix = feature_prefix
        if family == 'shannon':
            names = [f'{feature_prefix}_shannon']
        else:
            tag = {'renyi': 'a', 'tsallis': 'q', 'kaniadakis': 'k'}[family]
            names = [f'{feature_prefix}_{family}_{tag}{_format_param(p)}' for p in self.params]
        if escort_gamma is not None:
            names = [n + f'_esc{_format_param(escort_gamma)}' for n in names]
        if self.mode == 'spatial':
            names = [n + f'_g{self.grid_size}' for n in names]
        self._feature_names = names

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        states, n_states = states_from_dataframe(X, self.mode, x=self.x, y=self.y, aoi=self.aoi, grid_size=self.grid_size, bounds=self.bounds)
        c = state_counts(states, n_states=n_states)
        if self.smoothing > 0:
            c = c + self.smoothing
        p = _as_probabilities_from_counts(c)
        if self.escort_gamma is not None:
            p = escort_distribution(p, gamma=float(self.escort_gamma))
        vals: list[float] = []
        if self.family == 'shannon':
            vals.append(_entropy_of_distribution(p, 'shannon', 1.0, self.base))
        else:
            for param in self.params:
                vals.append(_entropy_of_distribution(p, self.family, param, self.base))
        return (self.get_feature_names_out(), vals)

def _energy_series(X: pd.DataFrame, *, kind: Literal['step_length', 'turning', 'speed', 'duration_inv', 'surprisal_state', 'surprisal_transition'], x: Optional[str], y: Optional[str], t: Optional[str], duration: Optional[str], state_mode: Literal['spatial', 'aoi']='spatial', aoi: Optional[str]=None, grid_size: int=10, bounds: BoundsType='minmax', smoothing: float=0.5) -> np.ndarray:
    """Compute a per-step/per-fixation energy series. Local by design (not cumulative)."""
    if kind == 'duration_inv':
        if duration is None:
            raise ValueError('duration column required')
        d = X[duration].to_numpy(dtype=float)
        return 1.0 / (d + _EPS)
    if kind in ('step_length', 'turning', 'speed'):
        if x is None or y is None:
            raise ValueError('x and y columns required')
        coords = X[[x, y]].to_numpy(dtype=float)
        if coords.shape[0] < 2:
            return np.array([], dtype=float)
        v = np.diff(coords, axis=0)
        step = np.linalg.norm(v, axis=1) + _EPS
        if kind == 'step_length':
            return step
        if kind == 'speed':
            if t is None:
                return step
            tt = X[t].to_numpy(dtype=float)
            dt = np.diff(tt) + _EPS
            return step / dt
        if v.shape[0] < 2:
            return np.array([], dtype=float)
        v1 = v[:-1]
        v2 = v[1:]
        dot = np.sum(v1 * v2, axis=1)
        n1 = np.linalg.norm(v1, axis=1) + _EPS
        n2 = np.linalg.norm(v2, axis=1) + _EPS
        cosang = np.clip(dot / (n1 * n2), -1.0, 1.0)
        ang = np.arccos(cosang)
        return ang + _EPS
    if kind == 'surprisal_state':
        states, n_states = states_from_dataframe(X, state_mode, x=x, y=y, aoi=aoi, grid_size=grid_size, bounds=bounds)
        c = state_counts(states, n_states) + smoothing
        p = _as_probabilities_from_counts(c)
        s = states.astype(int)
        E = -np.log(np.clip(p[s], _EPS, 1.0))
        return E
    if kind == 'surprisal_transition':
        states, n_states = states_from_dataframe(X, state_mode, x=x, y=y, aoi=aoi, grid_size=grid_size, bounds=bounds)
        C = transition_counts(states, n_states)
        P = transition_matrix_from_counts(C, smoothing=smoothing)
        s = states.astype(int)
        if s.size < 2:
            return np.array([], dtype=float)
        pij = P[s[:-1], s[1:]]
        E = -np.log(np.clip(pij, _EPS, 1.0))
        return E
    raise ValueError(f'Unknown energy kind: {kind}')

class ThermodynamicObservables(EntropyFeatureBase):
    """Compute thermodynamic observables from an energy series.

    For each beta:
      - free energy: F = -(1/beta) log Z
      - mean energy: <E>
      - entropy: S = log Z + beta <E>
      - heat capacity: C = beta^2 Var(E)

    Energy can be kinematic (step length, turning, speed) or surprisal-based.
    """

    def __init__(self, *, energy: Literal['step_length', 'turning', 'speed', 'duration_inv', 'surprisal_state', 'surprisal_transition']='step_length', betas: Sequence[float]=(0.5, 1.0, 2.0), base: float=2.0, state_mode: Literal['spatial', 'aoi']='spatial', grid_size: int=10, bounds: BoundsType='minmax', smoothing: float=0.5, x: str=None, y: str=None, t: str=None, duration: str=None, aoi: str=None, pk: list[str]=None, return_df: bool=True, ignore_errors: bool=True, feature_prefix: str='thermo'):
        super().__init__(x=x, y=y, t=t, duration=duration, aoi=aoi, pk=pk, return_df=return_df, ignore_errors=ignore_errors)
        self.energy = energy
        self.betas = tuple((float(b) for b in betas))
        self.base = float(base)
        self.state_mode = state_mode
        self.grid_size = int(grid_size)
        self.bounds = bounds
        self.smoothing = float(smoothing)
        self.feature_prefix = feature_prefix
        names: list[str] = []
        for b in self.betas:
            bb = _format_param(b)
            names.extend([f'{feature_prefix}_F_b{bb}', f'{feature_prefix}_E_b{bb}', f'{feature_prefix}_S_b{bb}', f'{feature_prefix}_C_b{bb}'])
        grid_suffix = ''
        if energy in ('surprisal_state', 'surprisal_transition'):
            grid_suffix = f'_g{self.grid_size}'
        names = [n + f'_{energy}{grid_suffix}' for n in names]
        self._feature_names = names

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        E = _energy_series(X, kind=self.energy, x=self.x, y=self.y, t=getattr(self, 't', None), duration=self.duration, state_mode=self.state_mode, aoi=self.aoi, grid_size=self.grid_size, bounds=self.bounds, smoothing=self.smoothing)
        if E.size == 0:
            return (self.get_feature_names_out(), [0.0 for _ in self.get_feature_names_out()])
        Emin, Emax = (float(np.min(E)), float(np.max(E)))
        if np.isclose(Emin, Emax):
            E = E - Emin
        else:
            E = (E - Emin) / (Emax - Emin)
        vals: list[float] = []
        for beta in self.betas:
            beta = float(beta)
            w = np.exp(-beta * E)
            Z = float(np.sum(w)) + _EPS
            q = w / Z
            meanE = float(np.sum(q * E))
            F = -(1.0 / beta) * _log_base(Z, self.base)
            S = _log_base(Z, self.base) + beta * meanE
            varE = float(np.sum(q * (E - meanE) ** 2))
            C = beta ** 2 * varE
            vals.extend([float(F), float(meanE), float(S), float(C)])
        return (self.get_feature_names_out(), vals)

def _grid_centroids(grid_size: int, bounds: BoundsType='unit') -> np.ndarray:
    if bounds == 'unit':
        xmin, xmax, ymin, ymax = (0.0, 1.0, 0.0, 1.0)
    elif bounds == 'minmax':
        xmin, xmax, ymin, ymax = (0.0, 1.0, 0.0, 1.0)
    else:
        xmin, xmax, ymin, ymax = map(float, bounds)
    xs = np.linspace(xmin, xmax, grid_size + 1)
    ys = np.linspace(ymin, ymax, grid_size + 1)
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    centroids = np.zeros((grid_size * grid_size, 2), dtype=float)
    for xi in range(grid_size):
        for yi in range(grid_size):
            idx = xi * grid_size + yi
            centroids[idx] = [xc[xi], yc[yi]]
    return centroids

class ThermodynamicTransitionDivergence(EntropyFeatureBase):
    """Divergence between empirical transitions P(.|i) and a Boltzmann cost model Q_beta(.|i).

    For each beta:
      Q_beta(j|i) ∝ exp(-beta * c(i,j))

    We report a weighted average over states:
      sum_i pi_i * D_alpha(P_i || Q_i)

    cost:
      - "spatial": Euclidean distance between grid-cell centroids
      - "aoi_uniform": cost 0 on same-state, 1 otherwise (minimal baseline)

    divergence: "renyi" (alpha) or "kl" (alpha ignored)
    """

    def __init__(self, *, mode: Literal['spatial', 'aoi']='spatial', cost: Literal['spatial', 'aoi_uniform']='spatial', divergence: Literal['renyi', 'kl']='renyi', alpha: float=2.0, betas: Sequence[float]=(0.5, 1.0, 2.0), grid_size: int=10, bounds: BoundsType='unit', smoothing: float=0.5, base: float=2.0, pi_mode: Literal['stationary', 'row_mass']='stationary', x: str=None, y: str=None, aoi: str=None, pk: list[str]=None, return_df: bool=True, ignore_errors: bool=True, feature_prefix: str='thermo_trans_div'):
        super().__init__(x=x, y=y, aoi=aoi, pk=pk, return_df=return_df, ignore_errors=ignore_errors)
        self.mode = mode
        self.cost = cost
        self.divergence = divergence
        self.alpha = float(alpha)
        self.betas = tuple((float(b) for b in betas))
        self.grid_size = int(grid_size)
        self.bounds = bounds
        self.smoothing = float(smoothing)
        self.base = float(base)
        self.pi_mode = pi_mode
        self.feature_prefix = feature_prefix
        names: list[str] = []
        for b in self.betas:
            bb = _format_param(b)
            if divergence == 'kl':
                names.append(f'{feature_prefix}_kl_b{bb}')
            else:
                names.append(f'{feature_prefix}_renyi_a{_format_param(alpha)}_b{bb}')
        grid_suffix = ''
        if mode == 'spatial':
            grid_suffix = f'_g{self.grid_size}'
        names = [n + f'_{mode}_{cost}{grid_suffix}' for n in names]
        self._feature_names = names

    def _cost_matrix(self, n_states: int) -> np.ndarray:
        if self.cost == 'aoi_uniform':
            C = np.ones((n_states, n_states), dtype=float)
            np.fill_diagonal(C, 0.0)
            return C
        g = self.grid_size
        cent = _grid_centroids(g, bounds=self.bounds)
        diff = cent[:, None, :] - cent[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        states, n_states = states_from_dataframe(X, self.mode, x=self.x, y=self.y, aoi=self.aoi, grid_size=self.grid_size, bounds=self.bounds)
        Cnt = transition_counts(states, n_states=n_states)
        P = transition_matrix_from_counts(Cnt, smoothing=self.smoothing)
        if self.pi_mode == 'stationary':
            pi = stationary_distribution(P)
        else:
            pi = _safe_prob(Cnt.sum(axis=1))
        cost = self._cost_matrix(n_states=n_states)
        vals: list[float] = []
        for beta in self.betas:
            beta = float(beta)
            W = np.exp(-beta * cost)
            Q = W / (np.sum(W, axis=1, keepdims=True) + _EPS)
            if self.divergence == 'kl':
                D_rows = np.sum(P * _log_base(P / (Q + _EPS), self.base), axis=1)
            else:
                a = self.alpha
                if np.isclose(a, 1.0):
                    D_rows = np.sum(P * _log_base(P / (Q + _EPS), self.base), axis=1)
                else:
                    s = np.sum(P ** a * (Q + _EPS) ** (1.0 - a), axis=1)
                    D_rows = 1.0 / (a - 1.0) * _log_base(s + _EPS, self.base)
            val = float(np.sum(pi * D_rows))
            vals.append(val)
        return (self.get_feature_names_out(), vals)
import math
from itertools import islice

def _get_weights_from_column(X: pd.DataFrame, col: Optional[str]) -> Optional[np.ndarray]:
    if col is None:
        return None
    if col not in X.columns:
        raise ValueError(f"weights column '{col}' not found in dataframe")
    w = X[col].to_numpy(dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return w

def state_counts_weighted(states: np.ndarray, n_states: int, weights: Optional[np.ndarray]=None) -> np.ndarray:
    s = np.asarray(states, dtype=int)
    out = np.zeros((int(n_states),), dtype=float)
    if s.size == 0:
        return out
    if weights is None:
        return np.bincount(s, minlength=int(n_states)).astype(float)
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != s.shape[0]:
        raise ValueError('weights must have same length as states')
    for idx, ww in zip(s, w):
        if 0 <= idx < n_states:
            out[idx] += float(ww)
    return out

def transition_counts_weighted(states: np.ndarray, n_states: int, weights: Optional[np.ndarray]=None) -> np.ndarray:
    s = np.asarray(states, dtype=int)
    M = np.zeros((int(n_states), int(n_states)), dtype=float)
    if s.size < 2:
        return M
    if weights is None:
        for i, j in zip(s[:-1], s[1:]):
            if 0 <= i < n_states and 0 <= j < n_states:
                M[i, j] += 1.0
        return M
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != s.shape[0]:
        raise ValueError('weights must have same length as states')
    for i, j, ww in zip(s[:-1], s[1:], w[:-1]):
        if 0 <= i < n_states and 0 <= j < n_states:
            M[i, j] += float(ww)
    return M

class WeightedStateDistributionEntropy(EntropyFeatureBase):
    """State distribution entropy where counts are weighted (e.g., by fixation duration).

    This implements the "duration-weighted occupancy" suggestion without changing
    the original StateDistributionEntropy.

    Parameters
    ----------
    weights: column name containing nonnegative weights (e.g., duration).
    Other parameters match StateDistributionEntropy.
    """

    def __init__(self, *, mode: Literal['spatial', 'aoi']='spatial', weights: str, family: EntropyFamily='shannon', params: Sequence[float]=(1.0,), escort_gamma: Optional[float]=None, grid_size: int=10, bounds: BoundsType='minmax', smoothing: float=0.0, base: float=2.0, x: str=None, y: str=None, aoi: str=None, pk: list[str]=None, return_df: bool=True, ignore_errors: bool=True, feature_prefix: str='state_entropy_w'):
        super().__init__(x=x, y=y, aoi=aoi, pk=pk, return_df=return_df, ignore_errors=ignore_errors)
        self.mode = mode
        self.weights = weights
        self.family = family
        self.params = tuple((float(p) for p in params))
        self.escort_gamma = escort_gamma
        self.grid_size = int(grid_size)
        self.bounds = bounds
        self.smoothing = float(smoothing)
        self.base = float(base)
        self.feature_prefix = feature_prefix
        if family == 'shannon':
            names = [f'{feature_prefix}_shannon_{weights}']
        else:
            tag = {'renyi': 'a', 'tsallis': 'q', 'kaniadakis': 'k'}[family]
            names = [f'{feature_prefix}_{family}_{tag}{_format_param(p)}_{weights}' for p in self.params]
        if escort_gamma is not None:
            names = [n + f'_esc{_format_param(escort_gamma)}' for n in names]
        if self.mode == 'spatial':
            names = [n + f'_g{self.grid_size}' for n in names]
        self._feature_names = names

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        states, n_states = states_from_dataframe(X, self.mode, x=self.x, y=self.y, aoi=self.aoi, grid_size=self.grid_size, bounds=self.bounds)
        w = _get_weights_from_column(X, self.weights)
        c = state_counts_weighted(states, n_states=n_states, weights=w)
        if self.smoothing > 0:
            c = c + self.smoothing
        p = _as_probabilities_from_counts(c)
        if self.escort_gamma is not None:
            p = escort_distribution(p, gamma=float(self.escort_gamma))
        vals: list[float] = []
        if self.family == 'shannon':
            vals.append(_entropy_of_distribution(p, 'shannon', 1.0, self.base))
        else:
            for param in self.params:
                vals.append(_entropy_of_distribution(p, self.family, param, self.base))
        return (self.get_feature_names_out(), vals)

class EdgeDistributionEntropy(EntropyFeatureBase):
    """Entropy of the edge (i->j) distribution formed by observed transitions.

    This is simpler than entropy rate: it ignores row-conditioning, treating all
    transitions as a flat categorical distribution over edges.

    Supports Shannon/Rényi/Tsallis/Kaniadakis.
    """

    def __init__(self, *, mode: Literal['spatial', 'aoi']='spatial', family: EntropyFamily='shannon', params: Sequence[float]=(2.0,), escort_gamma: Optional[float]=None, grid_size: int=10, bounds: BoundsType='minmax', smoothing: float=0.0, base: float=2.0, weights: Optional[str]=None, x: str=None, y: str=None, aoi: str=None, pk: list[str]=None, return_df: bool=True, ignore_errors: bool=True, feature_prefix: str='edge_entropy'):
        super().__init__(x=x, y=y, aoi=aoi, pk=pk, return_df=return_df, ignore_errors=ignore_errors)
        self.mode = mode
        self.family = family
        self.params = tuple((float(p) for p in params))
        self.escort_gamma = escort_gamma
        self.grid_size = int(grid_size)
        self.bounds = bounds
        self.smoothing = float(smoothing)
        self.base = float(base)
        self.weights = weights
        self.feature_prefix = feature_prefix
        if family == 'shannon':
            names = [f'{feature_prefix}_shannon']
        else:
            tag = {'renyi': 'a', 'tsallis': 'q', 'kaniadakis': 'k'}[family]
            names = [f'{feature_prefix}_{family}_{tag}{_format_param(p)}' for p in self.params]
        if weights is not None:
            names = [n + f'_{weights}' for n in names]
        if escort_gamma is not None:
            names = [n + f'_esc{_format_param(escort_gamma)}' for n in names]
        if self.mode == 'spatial':
            names = [n + f'_g{self.grid_size}' for n in names]
        self._feature_names = names

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        states, n_states = states_from_dataframe(X, self.mode, x=self.x, y=self.y, aoi=self.aoi, grid_size=self.grid_size, bounds=self.bounds)
        w = _get_weights_from_column(X, self.weights)
        C = transition_counts_weighted(states, n_states=n_states, weights=w)
        if self.smoothing > 0:
            C = C + self.smoothing
        p = _as_probabilities_from_counts(C.reshape(-1))
        if self.escort_gamma is not None:
            p = escort_distribution(p, gamma=float(self.escort_gamma))
        vals: list[float] = []
        if self.family == 'shannon':
            vals.append(_entropy_of_distribution(p, 'shannon', 1.0, self.base))
        else:
            for param in self.params:
                vals.append(_entropy_of_distribution(p, self.family, param, self.base))
        return (self.get_feature_names_out(), vals)

def _silverman_bandwidth_2d(coords: np.ndarray) -> float:
    n = coords.shape[0]
    if n < 2:
        return 1.0
    std = float(np.mean(np.std(coords, axis=0, ddof=1)))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    h = 1.06 * std * n ** (-1.0 / 6.0)
    return float(max(h, 1e-06))

def _kde2d_gaussian(coords: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, bandwidth: float, max_points: Optional[int]=None) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0:
        return np.zeros((grid_y.size, grid_x.size), dtype=float)
    if max_points is not None and coords.shape[0] > int(max_points):
        idx = np.linspace(0, coords.shape[0] - 1, int(max_points)).astype(int)
        coords = coords[idx]
    h = float(bandwidth)
    inv = 1.0 / (h + _EPS)
    gx, gy = np.meshgrid(grid_x, grid_y)
    dens = np.zeros_like(gx, dtype=float)
    for xi, yi in coords:
        dx = (gx - xi) * inv
        dy = (gy - yi) * inv
        dens += np.exp(-0.5 * (dx * dx + dy * dy))
    dens *= 1.0 / (coords.shape[0] * (2.0 * math.pi) * (h * h))
    return dens

def _normalize_density_grid(dens: np.ndarray, dx: float, dy: float) -> np.ndarray:
    Z = float(np.sum(dens) * dx * dy)
    if not np.isfinite(Z) or Z <= 0:
        return np.full_like(dens, 1.0 / max(dens.size, 1), dtype=float) / (dx * dy)
    return dens / Z

def _q_partition_weights(E: np.ndarray, beta: float, q: float) -> np.ndarray:
    w = _q_exp(-float(beta) * E, q=float(q))
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    Z = float(np.sum(w))
    if Z <= 0:
        return np.full_like(w, 1.0 / max(w.size, 1), dtype=float)
    return w / Z

def _boltzmann_weights(E: np.ndarray, beta: float) -> np.ndarray:
    w = np.exp(-float(beta) * E)
    w = np.where(np.isfinite(w), w, 0.0)
    Z = float(np.sum(w))
    if Z <= 0:
        return np.full_like(w, 1.0 / max(w.size, 1), dtype=float)
    return w / Z

def _thermo_observables_from_weights(E: np.ndarray, w: np.ndarray, beta: float, base: float, ensemble: str, q_ens: float) -> tuple[float, float, float, float]:
    Em = float(np.sum(w * E))
    Var = float(np.sum(w * (E - Em) ** 2))
    C = float(beta) ** 2 * Var
    if ensemble == 'tsallis':
        S = _entropy_of_distribution(w, 'tsallis', float(q_ens), base=base)
        u = _q_exp(-float(beta) * E, q=float(q_ens))
        u = np.where(np.isfinite(u) & (u > 0), u, 0.0)
        Z = float(np.sum(u))
        F = -(1.0 / max(float(beta), _EPS)) * float(_q_log(Z, q=float(q_ens)))
        return (float(F), float(Em), float(S), float(C))
    u = np.exp(-float(beta) * E)
    Z = float(np.sum(u))
    logZ = math.log(max(Z, _EPS))
    F = -(1.0 / max(float(beta), _EPS)) * (logZ / math.log(base))
    S = (logZ + float(beta) * Em) / math.log(base)
    return (float(F), float(Em), float(S), float(C))

class ThermodynamicObservablesExtended(EntropyFeatureBase):
    """Thermodynamic observables with:
    - optional cumulative energy (time-integrated)
    - ensemble: 'boltzmann' or 'tsallis' (q-exponential weights)
    - energy includes 'kde_surprisal' (continuous occupancy via KDE)
    """

    def __init__(self, *, energy: Literal['step_length', 'turning', 'speed', 'duration_inv', 'surprisal_state', 'surprisal_transition', 'kde_surprisal']='step_length', betas: Sequence[float]=(0.5, 1.0, 2.0), ensemble: Literal['boltzmann', 'tsallis']='boltzmann', q_ensemble: float=1.5, cumulative: bool=False, base: float=2.0, state_mode: Literal['spatial', 'aoi']='spatial', grid_size: int=10, bounds: BoundsType='minmax', smoothing: float=0.5, kde_grid_points: int=48, kde_bandwidth: Optional[float]=None, kde_max_points: Optional[int]=512, x: str=None, y: str=None, t: str=None, duration: str=None, aoi: str=None, pk: list[str]=None, return_df: bool=True, ignore_errors: bool=True, feature_prefix: str='thermo2'):
        super().__init__(x=x, y=y, t=t, duration=duration, aoi=aoi, pk=pk, return_df=return_df, ignore_errors=ignore_errors)
        self.energy = energy
        self.betas = tuple((float(b) for b in betas))
        self.ensemble = ensemble
        self.q_ensemble = float(q_ensemble)
        self.cumulative = bool(cumulative)
        self.base = float(base)
        self.state_mode = state_mode
        self.grid_size = int(grid_size)
        self.bounds = bounds
        self.smoothing = float(smoothing)
        self.kde_grid_points = int(kde_grid_points)
        self.kde_bandwidth = kde_bandwidth
        self.kde_max_points = kde_max_points
        self.feature_prefix = feature_prefix
        names: list[str] = []
        for b in self.betas:
            bb = _format_param(b)
            names.extend([f'{feature_prefix}_F_b{bb}', f'{feature_prefix}_E_b{bb}', f'{feature_prefix}_S_b{bb}', f'{feature_prefix}_C_b{bb}'])
        names = [n + f'_{energy}_{ensemble}' + (f'_q{_format_param(self.q_ensemble)}' if ensemble == 'tsallis' else '') + ('_cum' if cumulative else '') for n in names]
        self._feature_names = names

    def _energy(self, X: pd.DataFrame) -> np.ndarray:
        if self.energy == 'kde_surprisal':
            if self.x is None or self.y is None:
                raise ValueError('x,y must be provided for kde_surprisal energy')
            coords = X[[self.x, self.y]].to_numpy(dtype=float)
            if coords.shape[0] < 3:
                return np.array([], dtype=float)
            if self.bounds == 'unit':
                xmin, xmax, ymin, ymax = (0.0, 1.0, 0.0, 1.0)
            elif self.bounds == 'minmax':
                xmin, xmax = (float(np.min(coords[:, 0])), float(np.max(coords[:, 0])))
                ymin, ymax = (float(np.min(coords[:, 1])), float(np.max(coords[:, 1])))
            else:
                xmin, xmax, ymin, ymax = map(float, self.bounds)
            if np.isclose(xmin, xmax):
                xmax = xmin + 1.0
            if np.isclose(ymin, ymax):
                ymax = ymin + 1.0
            gx = np.linspace(xmin, xmax, self.kde_grid_points)
            gy = np.linspace(ymin, ymax, self.kde_grid_points)
            dx = float(gx[1] - gx[0]) if gx.size > 1 else 1.0
            dy = float(gy[1] - gy[0]) if gy.size > 1 else 1.0
            bw = float(self.kde_bandwidth) if self.kde_bandwidth is not None else _silverman_bandwidth_2d(coords)
            dens_grid = _kde2d_gaussian(coords, gx, gy, bandwidth=bw, max_points=self.kde_max_points)
            dens_grid = _normalize_density_grid(dens_grid, dx=dx, dy=dy)
            xi = np.clip(((coords[:, 0] - xmin) / (xmax - xmin + _EPS) * (gx.size - 1)).astype(int), 0, gx.size - 1)
            yi = np.clip(((coords[:, 1] - ymin) / (ymax - ymin + _EPS) * (gy.size - 1)).astype(int), 0, gy.size - 1)
            dens = dens_grid[yi, xi]
            E = -np.log(np.clip(dens, _EPS, None))
            return E
        return _energy_series(X, kind=self.energy, x=self.x, y=self.y, t=getattr(self, 't', None), duration=self.duration, state_mode=self.state_mode, aoi=self.aoi, grid_size=self.grid_size, bounds=self.bounds, smoothing=self.smoothing)

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        E = self._energy(X)
        if E.size == 0:
            return (self.get_feature_names_out(), [0.0 for _ in self.get_feature_names_out()])
        if self.cumulative:
            E = np.cumsum(E)
        Emin, Emax = (float(np.min(E)), float(np.max(E)))
        if np.isclose(Emin, Emax):
            En = E - Emin
        else:
            En = (E - Emin) / (Emax - Emin)
        vals: list[float] = []
        for beta in self.betas:
            if self.ensemble == 'tsallis':
                w = _q_partition_weights(En, beta=beta, q=self.q_ensemble)
                F, Em, S, C = _thermo_observables_from_weights(En, w, beta=beta, base=self.base, ensemble='tsallis', q_ens=self.q_ensemble)
            else:
                w = _boltzmann_weights(En, beta=beta)
                F, Em, S, C = _thermo_observables_from_weights(En, w, beta=beta, base=self.base, ensemble='boltzmann', q_ens=1.0)
            vals.extend([F, Em, S, C])
        return (self.get_feature_names_out(), vals)
