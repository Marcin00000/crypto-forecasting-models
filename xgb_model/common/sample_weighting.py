"""
Sample weighting and mis-direction penalty utilities for XGBoost model.

Etap 1 -- podejście natywne (A):
    compute_sample_weights()  -- oblicza per-sample wagi z |log_return|
    apply_misdir_penalty()    -- aktualizuje wagi po pierwszym treningu

Etap 2 -- custom objective (B):
    make_custom_objective()   -- factory function zwracajaca custom obj
                                 dla xgb.train(obj=...)

Signal Threshold:
    resolve_signal_threshold() -- resolves threshold from config (fixed/empirical)
    make_dead_zone_objective() -- alternative objective with dead-zone below threshold
"""
import numpy as np


# =============================================================================
# SIGNAL THRESHOLD RESOLUTION
# =============================================================================

def resolve_signal_threshold(y_raw, mode='fixed', fixed_value=0.001, percentile=50.0):
    """
    Resolve signal threshold from configuration.

    Args:
        y_raw:       np.ndarray — raw (unscaled) log-returns
        mode:        'fixed' | 'empirical'
        fixed_value: threshold for 'fixed' mode
        percentile:  percentile of |y_raw| for 'empirical' mode

    Returns:
        float — resolved threshold value
    """
    if mode == 'fixed':
        return float(fixed_value)
    elif mode == 'empirical':
        return float(np.percentile(np.abs(y_raw), percentile))
    else:
        raise ValueError(f"Unknown threshold mode: '{mode}'. Use 'fixed' or 'empirical'.")


def _threshold_weights(abs_y, threshold, smooth=False, smooth_width=2.0, low_w=0.1):
    """
    Compute per-sample weight multiplier based on signal threshold.

    Args:
        abs_y:       np.ndarray — |log_return| values
        threshold:   float — signal threshold
        smooth:      bool — use smooth sigmoid transition
        smooth_width: float — sigmoid width (higher = smoother transition)
        low_w:       float — weight for below-threshold samples

    Returns:
        np.ndarray — weight multiplier in [low_w, 1.0]
    """
    if smooth:
        scale = threshold * smooth_width if threshold > 0 else smooth_width
        x = (abs_y - threshold) / max(scale, 1e-8)
        sigmoid = 1.0 / (1.0 + np.exp(np.clip(-x * 5, -30, 30)))
        return low_w + (1.0 - low_w) * sigmoid
    else:
        return np.where(abs_y >= threshold, 1.0, low_w)


# =============================================================================
# SAMPLE WEIGHTS (podejście A — DMatrix weight)
# =============================================================================

def compute_sample_weights(y_raw, mode, alpha=1.0, min_w=0.1,
                           percentile=70.0, low_w=0.1,
                           use_threshold=False,
                           threshold=0.001,
                           smooth_transition=False,
                           smooth_width=2.0,
                           threshold_low_w=0.1):
    """
    Compute per-sample weights from raw (unscaled) log-return targets.

    Weights are passed to xgb.DMatrix(weight=...) so that XGBoost
    scales each sample's hessian contribution accordingly. Computed
    from raw (pre-scaling) y_train so that magnitude reflects actual
    market moves regardless of target scaler settings.

    Args:
        y_raw:      np.ndarray -- raw log-returns before target scaling.
        mode:       str | None -- weighting scheme:
                      None         -- uniform (no weighting, XGBoost default)
                      'abs'        -- w = |y|^alpha
                      'abs_clip'   -- w = clip(|y|^alpha, floor, inf)
                                      where floor = min_w * max(w)
                      'rank'       -- w = rank(|y|) / N  (scale-free)
                      'percentile' -- w = 1.0 if |y| >= p-th pct else low_w
        alpha:      float -- exponent for 'abs' / 'abs_clip' modes.
        min_w:      float -- floor fraction of max weight for 'abs_clip'.
        percentile: float -- threshold percentile for 'percentile' mode.
        low_w:      float -- weight for below-threshold samples in
                    'percentile' mode.
        use_threshold:      bool -- overlay threshold-based weights on top
        threshold:          float -- signal threshold (absolute value)
        smooth_transition:  bool -- smooth sigmoid around threshold
        smooth_width:       float -- sigmoid width
        threshold_low_w:    float -- weight for below-threshold samples

    Returns:
        np.ndarray of shape (n_samples,) with positive weights, or
        None if mode is None and use_threshold is False.
    """
    if mode is None and not use_threshold:
        return None

    abs_y = np.abs(y_raw).astype(np.float64)

    if mode is None:
        w = np.ones(len(y_raw), dtype=np.float64)
    elif mode == 'abs':
        w = abs_y ** alpha
    elif mode == 'abs_clip':
        raw   = abs_y ** alpha
        floor = min_w * raw.max() if raw.max() > 0 else 1.0
        w     = np.clip(raw, floor, None)
    elif mode == 'rank':
        order        = np.argsort(abs_y)
        ranks        = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(abs_y) + 1)
        w            = ranks / len(abs_y)
    elif mode == 'percentile':
        pct_threshold = np.percentile(abs_y, percentile)
        w         = np.where(abs_y >= pct_threshold, 1.0, low_w)
    else:
        raise ValueError(
            f"Unknown sample_weight_mode: '{mode}'. "
            f"Valid: None, 'abs', 'abs_clip', 'rank', 'percentile'"
        )

    # Threshold-based weight overlay (multiplicative)
    if use_threshold and threshold > 0:
        tw = _threshold_weights(
            abs_y, threshold,
            smooth=smooth_transition,
            smooth_width=smooth_width,
            low_w=threshold_low_w,
        )
        w = w * tw

    # Guard: replace zero / negative weights with a tiny positive value
    w = np.where(w <= 0, 1e-9, w)
    return w.astype(np.float32)


# =============================================================================
# MISDIR PENALTY — podejście A (aktualizacja wag, drugi trening)
# =============================================================================

def apply_misdir_penalty(y_raw, y_pred_raw, weights, penalty, min_quantile):
    """
    Apply a constant multiplier to weights of samples mis-predicted on
    large moves, then pass the updated weights to a second xgb.train().

    'Mis-directed' means sign(y_pred_raw) != sign(y_raw) for samples
    whose |y_raw| >= the min_quantile-th percentile of |y_raw|.

    The penalty only increases existing weights -- it never zeroes
    anything out, so the model does not collapse toward zero predictions.

    Args:
        y_raw:        np.ndarray -- raw (unscaled) training targets.
        y_pred_raw:   np.ndarray -- preliminary predictions in raw space
                      (inverse-transformed if target_scaler was active).
        weights:      np.ndarray | None -- current per-sample weights.
                      If None, starts from uniform weights of 1.0.
        penalty:      float -- multiplier for mis-directed large-move
                      samples. Recommended: 2.0. Values < 1.0 clamped to 1.0.
        min_quantile: float in [0, 1) -- |y_raw| quantile threshold.
                      Only samples at or above this quantile are eligible.
                      Default 0.7 = top 30% by |return|.

    Returns:
        tuple: (updated_weights np.ndarray, n_penalised int)
    """
    penalty = max(1.0, float(penalty))

    if weights is None:
        weights = np.ones(len(y_raw), dtype=np.float32)
    else:
        weights = weights.copy().astype(np.float64)

    abs_y     = np.abs(y_raw)
    threshold = np.quantile(abs_y, min_quantile)

    large_move   = abs_y >= threshold
    wrong_dir    = np.sign(y_pred_raw) != np.sign(y_raw)
    penalty_mask = large_move & wrong_dir

    n_penalised = int(np.sum(penalty_mask))
    weights[penalty_mask] *= penalty

    return weights.astype(np.float32), n_penalised


# =============================================================================
# CUSTOM OBJECTIVE (podejście B — weighted_huber)
# =============================================================================

def _pseudo_huber_grad_hess(y_pred, y_true, delta):
    """
    Compute pseudo-Huber gradient and hessian per sample.

    L = delta^2 * (sqrt(1 + ((y_pred - y_true) / delta)^2) - 1)
    grad = dL/dy_pred = (y_pred - y_true) / sqrt(1 + r^2)
    hess = d^2L/dy_pred^2 = 1/scale - r^2 / (delta^2 * scale^3)
    where r = (y_pred - y_true) / delta,  scale = sqrt(1 + r^2)
    """
    residuals = y_pred - y_true
    r         = residuals / delta
    scale     = np.sqrt(1.0 + r ** 2)
    grad      = residuals / scale
    hess      = (1.0 / scale) - (r ** 2 / (scale ** 3))
    hess      = np.maximum(hess, 1e-6)   # stabilnosc numeryczna
    return grad, hess


def make_custom_objective(
        target_scaler=None,
        use_sample_weights=False,
        weight_mode='percentile',
        weight_alpha=1.0,
        weight_min_w=0.1,
        weight_percentile=70.0,
        weight_low_w=0.1,
        use_misdir_penalty=False,
        misdir_penalty=2.0,
        misdir_min_quantile=0.7,
        huber_delta=1.0,
        # threshold-based penalty overlay
        use_threshold=False,
        threshold=0.001,
        smooth_transition=False,
        smooth_width=2.0,
        threshold_low_w=0.1,
):
    """
    Factory zwracajaca custom objective (weighted_huber) dla xgb.train(obj=...).

    Zastepuje standardowy params['objective'] -- XGBoost ignoruje
    'objective' w params gdy obj= jest podane w xgb.train().

    Parametry use_sample_weights i use_misdir_penalty sa niezalezne.
    use_threshold adds a threshold-based weight overlay (multiplicative)
    on top of existing sample weights.

    Args:
        target_scaler:       TargetScaler | None
        use_sample_weights:  bool
        weight_mode:         str -- tryb wag
        weight_alpha:        float
        weight_min_w:        float
        weight_percentile:   float
        weight_low_w:        float
        use_misdir_penalty:  bool
        misdir_penalty:      float
        misdir_min_quantile: float
        huber_delta:         float | 'auto'
        use_threshold:       bool -- overlay threshold-based weights
        threshold:           float -- signal threshold
        smooth_transition:   bool -- smooth sigmoid around threshold
        smooth_width:        float -- sigmoid width
        threshold_low_w:     float -- weight for below-threshold samples

    Returns:
        Callable (y_pred, dtrain) -> (grad, hess)
    """
    auto_delta = (huber_delta == 'auto')
    if auto_delta:
        effective_delta = [None]
    else:
        effective_delta = [float(huber_delta)]

    def build_static_terms(y_scaled):
        """Label-only terms: identical in every boosting round, so computed once per label vector."""
        y_raw = target_scaler.inverse_transform(y_scaled) if target_scaler is not None else y_scaled
        static = {}

        # Skladnik 1: sample weights z |y_raw|
        base = np.ones(len(y_raw), dtype=np.float64)
        if use_sample_weights:
            w = compute_sample_weights(
                y_raw,
                mode=weight_mode,
                alpha=weight_alpha,
                min_w=weight_min_w,
                percentile=weight_percentile,
                low_w=weight_low_w,
            )
            if w is not None:
                base *= w.astype(np.float64)

        # Skladnik 2 (czesc stala): zbior duzych ruchow i znak etykiety
        if use_misdir_penalty:
            abs_y = np.abs(y_raw)
            mis_threshold = np.quantile(abs_y, misdir_min_quantile)
            static['large_move'] = abs_y >= mis_threshold
            static['sign_y'] = np.sign(y_raw)

        # Skladnik 3: threshold-based weight overlay
        tw = None
        if use_threshold and threshold > 0:
            tw = _threshold_weights(
                np.abs(y_raw), threshold,
                smooth=smooth_transition,
                smooth_width=smooth_width,
                low_w=threshold_low_w,
            ).astype(np.float64)
        static['base'] = base
        static['tw'] = tw
        # Bez kary misdir mnoznik jest stala; kolejnosc mnozen (wagi, potem prog) bez zmian,
        # aby wynik byl bitowo identyczny z obliczeniem w kazdej iteracji.
        if not use_misdir_penalty:
            static['fixed_multiplier'] = base if tw is None else base * tw
        return static

    cache = {'labels': None, 'static': None}

    def objective(y_pred, dtrain):
        y_scaled = dtrain.get_label()

        # Auto-adaptive delta
        if auto_delta and effective_delta[0] is None:
            if target_scaler is None:
                data_std = float(np.std(y_scaled))
                effective_delta[0] = max(data_std, 1e-8)
            else:
                effective_delta[0] = 1.0

        delta = effective_delta[0]

        # Etykiety nie zmieniaja sie miedzy iteracjami; przy innym dtrain cache jest budowany od nowa
        labels = cache['labels']
        if labels is None or labels.shape != y_scaled.shape or not np.array_equal(labels, y_scaled):
            cache['static'] = build_static_terms(y_scaled)
            cache['labels'] = np.array(y_scaled, copy=True)
        static = cache['static']

        # Pseudo-Huber grad/hess w skalowanej przestrzeni
        grad, hess = _pseudo_huber_grad_hess(y_pred, y_scaled, delta=delta)

        if use_misdir_penalty:
            # Skladnik 2 (czesc zalezna od predykcji): kara za mis-directional duze ruchy
            y_pred_raw = (target_scaler.inverse_transform(y_pred)
                          if target_scaler is not None else y_pred)
            wrong_dir = np.sign(y_pred_raw) != static['sign_y']
            multiplier = static['base'].copy()
            multiplier[static['large_move'] & wrong_dir] *= misdir_penalty
            if static['tw'] is not None:
                multiplier *= static['tw']
        else:
            multiplier = static['fixed_multiplier']

        return (grad * multiplier).astype(np.float32), \
               (hess * multiplier).astype(np.float32)

    # Expose effective_delta for external logging
    objective.effective_delta = effective_delta
    objective.auto_delta = auto_delta

    return objective


# =============================================================================
# DEAD-ZONE OBJECTIVE (alternatywna funkcja celu)
# =============================================================================

def make_dead_zone_objective(
        target_scaler=None,
        signal_threshold=0.001,
        dead_zone_weight=0.1,
        smooth_transition=False,
        smooth_width=2.0,
        use_misdir_penalty=False,
        misdir_penalty=2.0,
        huber_delta=1.0,
):
    """
    Factory zwracajaca dead-zone objective dla xgb.train(obj=...).

    Alternatywa do make_custom_objective(). Zamiast wazenia probek
    przez |y|, tlumi gradienty dla |y| < signal_threshold (dead zone).
    Model uczy sie tych wartosci z niskim priorytetem.

    Powyzej progu: pelny gradient + opcjonalna kara misdir
    (stosowana tylko do probek powyzej progu).

    Args:
        target_scaler:       TargetScaler | None
        signal_threshold:    float — próg sygnałowy (log-return)
        dead_zone_weight:    float — mnożnik gradientu w dead zone (0.1 = 10%)
        smooth_transition:   bool — gładkie przejście sigmoid
        smooth_width:        float — szerokość sigmoidy
        use_misdir_penalty:  bool — kara mis-direction powyżej progu
        misdir_penalty:      float — mnożnik kary
        huber_delta:         float | 'auto' — delta Pseudo-Hubera

    Returns:
        Callable (y_pred, dtrain) -> (grad, hess)
    """
    auto_delta = (huber_delta == 'auto')
    if auto_delta:
        effective_delta = [None]
    else:
        effective_delta = [float(huber_delta)]

    def build_static_terms(y_scaled):
        """Label-only terms: identical in every boosting round, so computed once per label vector."""
        y_raw = target_scaler.inverse_transform(y_scaled) if target_scaler is not None else y_scaled
        abs_y = np.abs(y_raw)
        # Dead-zone weights: low for |y_raw| < threshold, 1.0 above
        static = {'base': _threshold_weights(
            abs_y, signal_threshold,
            smooth=smooth_transition,
            smooth_width=smooth_width,
            low_w=dead_zone_weight,
        ).astype(np.float64)}
        if use_misdir_penalty:
            static['above_threshold'] = abs_y >= signal_threshold
            static['sign_y'] = np.sign(y_raw)
        return static

    cache = {'labels': None, 'static': None}

    def objective(y_pred, dtrain):
        y_scaled = dtrain.get_label()

        # Auto-adaptive delta
        if auto_delta and effective_delta[0] is None:
            if target_scaler is None:
                data_std = float(np.std(y_scaled))
                effective_delta[0] = max(data_std, 1e-8)
            else:
                effective_delta[0] = 1.0

        delta = effective_delta[0]

        # Etykiety nie zmieniaja sie miedzy iteracjami; przy innym dtrain cache jest budowany od nowa
        labels = cache['labels']
        if labels is None or labels.shape != y_scaled.shape or not np.array_equal(labels, y_scaled):
            cache['static'] = build_static_terms(y_scaled)
            cache['labels'] = np.array(y_scaled, copy=True)
        static = cache['static']

        # Pseudo-Huber grad/hess
        grad, hess = _pseudo_huber_grad_hess(y_pred, y_scaled, delta=delta)

        # Misdir penalty (only above threshold)
        if use_misdir_penalty:
            y_pred_raw = (target_scaler.inverse_transform(y_pred)
                          if target_scaler is not None else y_pred)
            wrong_dir = np.sign(y_pred_raw) != static['sign_y']
            multiplier = static['base'].copy()
            multiplier[static['above_threshold'] & wrong_dir] *= misdir_penalty
        else:
            multiplier = static['base']

        return (grad * multiplier).astype(np.float32), \
               (hess * multiplier).astype(np.float32)

    objective.effective_delta = effective_delta
    objective.auto_delta = auto_delta

    return objective
