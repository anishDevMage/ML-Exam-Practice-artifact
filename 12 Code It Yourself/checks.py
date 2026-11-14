"""Auto-checker for the '12 Code It Yourself' workbook.

Spoiler warning: this file contains reference implementations.
Don't open it until you've cleared a level on your own.
"""
import json
import os
import time

import numpy as np

LEVELS = [
    ('0.1', 'add_intercept'),
    ('0.2', 'predict  (row · θ)'),
    ('0.3', 'gradient: for-loops vs X.T @ e'),
    ('1', 'standardize from scratch'),
    ('2', 'mse and r2 from scratch'),
    ('3a', 'closed form'),
    ('3b', 'ridge closed form'),
    ('4', 'batch gradient descent'),
    ('5', 'mini-batch and stochastic GD'),
    ('6', 'Ridge, Lasso, Elastic penalty classes'),
    ('7', 'your LinearRegression class'),
    ('8a', 'k-fold indices from scratch'),
    ('8b', 'choose λ by cross-validation'),
    ('9a', 'sklearn LinearRegression = your closed form'),
    ('9b', 'sklearn Ridge = your ridge closed form'),
    ('9c', 'GridSearchCV pipeline'),
    ('10', 'sigmoid, cross-entropy, gradient'),
    ('11a', 'logistic regression by batch GD'),
    ('11b', 'confusion matrix, precision, recall, F1'),
    ('12', 'BOSS: your LogisticRegression class'),
    ('13', 'BONUS: softmax regression'),
]
_PROG = 'progress_12.json'


# ---------------------------------------------------------------- plumbing
def _load():
    try:
        with open(_PROG, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save(d):
    try:
        with open(_PROG, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=1)
    except Exception:
        pass


def _pass(level, note=''):
    d = _load()
    if level not in d:
        d[level] = time.strftime('%Y-%m-%d %H:%M')
        _save(d)
    done = sum(1 for k, _ in LEVELS if k in d)
    print(f'✅ Level {level} cleared' + (f'. {note}' if note else '.'))
    print(f'   [{"█" * done}{"░" * (len(LEVELS) - done)}] {done}/{len(LEVELS)}')
    return True


def _fail(level, msg):
    print(f'❌ Level {level}: {msg}')
    return False


class _Stop(Exception):
    pass


def _call(level, f, *a, **k):
    try:
        return f(*a, **k)
    except NotImplementedError:
        _fail(level, 'not written yet (it still raises NotImplementedError).')
    except _Stop:
        raise
    except Exception as e:
        _fail(level, f'your code crashed with {type(e).__name__}: {e}\n'
                     f'   Tip: call the function on its own in a new cell to see the full traceback.')
    raise _Stop


def _short(a):
    a = np.asarray(a, dtype=float)
    s = np.array2string(a.ravel()[:6], precision=4, suppress_small=True)
    return s + (' …' if a.size > 6 else '')


def _same(level, name, got, exp, rtol=1e-6, atol=1e-8):
    if got is None:
        _fail(level, f'{name} is None. Did you forget to return it?')
        raise _Stop
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    if got.shape != exp.shape:
        _fail(level, f'{name} has shape {got.shape}, expected {exp.shape}.')
        raise _Stop
    if not np.allclose(got, exp, rtol=rtol, atol=atol):
        _fail(level, f'{name} has the right shape {got.shape} but wrong values.\n'
                     f'   expected {_short(exp)}\n   got      {_short(got)}')
        raise _Stop


def _guard(fn):
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except _Stop:
            return False
    wrapper.__doc__ = fn.__doc__
    return wrapper


def scoreboard():
    d = _load()
    for k, name in LEVELS:
        mark = '✅' if k in d else '⬜'
        when = f'  ({d[k]})' if k in d else ''
        print(f'{mark} {k:>4}  {name}{when}')
    done = sum(1 for k, _ in LEVELS if k in d)
    print(f'\n{done}/{len(LEVELS)} cleared')


# ---------------------------------------------------------------- references
def _data(m=40, n=3, seed=0, noise=0.3):
    r = np.random.default_rng(seed)
    X = r.normal(size=(m, n))
    X = (X - X.mean(0)) / X.std(0)
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    true = np.array([3.0, 2.0, -1.0, 0.5, 1.5, -2.0][: n + 1])
    y = X @ true + noise * r.normal(size=m)
    return X, y


def _mse(a, b):
    return float(((a - b) ** 2).sum() / a.shape[0])


def _closed(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


def _ridge_closed(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def _bgd(X, y, lr, max_iter):
    th = np.zeros(X.shape[1])
    m = X.shape[0]
    for _ in range(max_iter):
        th = th - lr * (1 / m) * X.T @ (X @ th - y)
    return th


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _logit_gd(X, y, lr, max_iter):
    th = np.zeros(X.shape[1])
    m = X.shape[0]
    for _ in range(max_iter):
        th = th - lr * (1 / m) * X.T @ (_sig(X @ th) - y)
    return th


# ---------------------------------------------------------------- level 0
@_guard
def check_0_1(add_intercept):
    X = np.arange(15, dtype=float).reshape(5, 3)
    got = _call('0.1', add_intercept, X)
    _same('0.1', 'add_intercept(X)', got, np.c_[np.ones(5), X])
    return _pass('0.1', 'Shape (5, 3) became (5, 4); the first column is all 1s.')


@_guard
def check_0_2(predict):
    X = np.array([[1, 2, 5], [1, -1, 3], [1, 4, 8]], dtype=float)
    th = np.array([1, 0.5, -1])
    got = _call('0.2', predict, X, th)
    _same('0.2', 'predict(X, theta)', got, [-3, -2.5, -5])
    return _pass('0.2', 'ŷ = [−3, −2.5, −5]: each ŷ is one ROW of X dotted with θ.')


@_guard
def check_0_3(grad_loop, grad_vec):
    X = np.array([[1, 2, 5], [1, -1, 3], [1, 4, 8]], dtype=float)
    th = np.array([1, 0.5, -1])
    y = np.array([2, -4, 1], dtype=float)
    exp = np.array([-9.5, -35.5, -68.5])
    _same('0.3', 'grad_loop on the worked example', _call('0.3', grad_loop, X, y, th), exp)
    _same('0.3', 'grad_vec on the worked example', _call('0.3', grad_vec, X, y, th), exp)
    Xr, yr = _data(23, 4, seed=3)
    thr = np.linspace(-1, 1, 5)
    a = _call('0.3', grad_loop, Xr, yr, thr)
    b = _call('0.3', grad_vec, Xr, yr, thr)
    _same('0.3', 'grad_loop on random data', a, Xr.T @ (Xr @ thr - yr))
    _same('0.3', 'grad_vec on random data', b, Xr.T @ (Xr @ thr - yr))
    return _pass('0.3', 'Both give [−9.5, −35.5, −68.5]. X.T @ e IS the double for-loop: each θ_j sums DOWN its column.')


# ---------------------------------------------------------------- level 1-3
@_guard
def check_1(standardize):
    r = np.random.default_rng(1)
    Xtr = r.normal(5, 3, size=(30, 4))
    Xte = r.normal(5, 3, size=(10, 4))
    out = _call('1', standardize, Xtr, Xte)
    if not (isinstance(out, tuple) and len(out) == 2):
        return _fail('1', 'standardize should return TWO arrays: (X_train_scaled, X_test_scaled).')
    a, b = out
    mu, sd = Xtr.mean(0), Xtr.std(0)
    _same('1', 'X_train_scaled', a, (Xtr - mu) / sd)
    leak = (Xte - Xte.mean(0)) / Xte.std(0)
    if b is not None and np.shape(b) == Xte.shape and np.allclose(b, leak):
        return _fail('1', 'X_test_scaled used the TEST mean and std. Always use the TRAIN mean and std '
                          '(that is what scaler.transform(X_test) does). Using test statistics is data leakage.')
    _same('1', 'X_test_scaled', b, (Xte - mu) / sd)
    return _pass('1', 'Same numbers as sklearn StandardScaler (population std, ddof = 0).')


@_guard
def check_2(mse, r2):
    from sklearn.metrics import mean_squared_error, r2_score
    r = np.random.default_rng(2)
    y = r.normal(size=50)
    yh = y + r.normal(scale=0.5, size=50)
    _same('2', 'mse(yhat, y)', _call('2', mse, yh, y), mean_squared_error(y, yh))
    _same('2', 'r2(yhat, y)', _call('2', r2, yh, y), r2_score(y, yh))
    _same('2', 'r2 of a perfect prediction', _call('2', r2, y, y), 1.0)
    _same('2', 'r2 of predicting the mean', _call('2', r2, np.full(50, y.mean()), y), 0.0, atol=1e-9)
    return _pass('2', 'R² = 1 is perfect, 0 means "no better than predicting the mean".')


@_guard
def check_3a(closed_form):
    X = np.array([[1, -1], [1, 2], [1, 4]], dtype=float)
    y = np.array([1, 4, 5], dtype=float)
    _same('3a', 'closed_form on the hand example', _call('3a', closed_form, X, y), _closed(X, y))
    Xr, yr = _data(30, 3, seed=4)
    _same('3a', 'closed_form on random data', _call('3a', closed_form, Xr, yr), _closed(Xr, yr))
    return _pass('3a', f'Hand example: θ = {_short(_closed(X, y))}. Check it against your paper method!')


@_guard
def check_3b(ridge_closed_form):
    Xr, yr = _data(30, 3, seed=5)
    for lam in (0.0, 1.0, 25.0):
        _same('3b', f'ridge_closed_form(lam={lam})', _call('3b', ridge_closed_form, Xr, yr, lam), _ridge_closed(Xr, yr, lam))
    a, b = _closed(Xr, yr), _ridge_closed(Xr, yr, 25.0)
    return _pass('3b', f'|θ| shrank from {np.linalg.norm(a):.3f} (λ = 0) to {np.linalg.norm(b):.3f} (λ = 25). '
                       'That shrinking IS regularization.')


# ---------------------------------------------------------------- level 4-7
@_guard
def check_4(batch_gd):
    X, y = _data(40, 3, seed=6)
    out = _call('4', batch_gd, X, y, 0.1, 300)
    if not (isinstance(out, tuple) and len(out) == 2):
        return _fail('4', 'batch_gd should return (theta, losses).')
    th, losses = out
    _same('4', 'theta after 300 iterations (lr 0.1)', th, _bgd(X, y, 0.1, 300))
    if len(losses) != 300:
        return _fail('4', f'losses has {len(losses)} entries; expected one per iteration (300).')
    ls = np.asarray(losses, dtype=float)
    if not np.all(np.diff(ls) <= 1e-12):
        return _fail('4', 'losses should never go up with lr = 0.1 on this data.')
    _same('4', 'theta vs closed form (should have converged)', th, _closed(X, y), rtol=1e-4, atol=1e-4)
    return _pass('4', f'Loss {ls[0]:.2f} → {ls[-1]:.4f}, and θ matches the closed form.')


@_guard
def check_5(minibatch_gd):
    X, y = _data(40, 3, seed=7)
    full = _call('5', minibatch_gd, X, y, 0.1, 50, 40, 0)
    _same('5', 'minibatch_gd with bs = m (must equal batch GD)', full, _bgd(X, y, 0.1, 50), rtol=1e-8, atol=1e-10)
    mini = _call('5', minibatch_gd, X, y, 0.05, 200, 8, 0)
    _same('5', 'minibatch_gd with bs = 8 (should land near the closed form)', mini, _closed(X, y), rtol=0.05, atol=0.05)
    sgd = _call('5', minibatch_gd, X, y, 0.01, 200, 1, 0)
    _same('5', 'minibatch_gd with bs = 1, i.e. SGD (should land near the closed form)', sgd, _closed(X, y), rtol=0.05, atol=0.08)
    return _pass('5', 'bs = m is batch GD, bs = 1 is SGD, anything between is mini-batch. One function, three methods.')


@_guard
def check_6(Ridge, Lasso, Elastic):
    th = np.array([1.0, -2.0, 0.0, 3.0])
    r = _call('6', Ridge, 0.5)
    _same('6', 'Ridge(0.5)(theta)', _call('6', r, th), 0.5 * 14)
    _same('6', 'Ridge(0.5).derivation(theta)', _call('6', r.derivation, th), [1, -2, 0, 3])
    l = _call('6', Lasso, 0.5)
    _same('6', 'Lasso(0.5)(theta)', _call('6', l, th), 3.0)
    _same('6', 'Lasso(0.5).derivation(theta)', _call('6', l.derivation, th), [0.5, -0.5, 0, 0.5])
    e = _call('6', Elastic, 1.0, 0.5)
    _same('6', 'Elastic(1, 0.5)(theta)', _call('6', e, th), 3 + 7)
    _same('6', 'Elastic(1, 0.5).derivation(theta)', _call('6', e.derivation, th), [1.5, -2.5, 0, 3.5])
    return _pass('6', 'Ridge gradient 2λθ shrinks big θ more; Lasso gradient λ·sign(θ) pushes every θ by the same amount, so small ones hit 0.')


@_guard
def check_7(LinearRegression, Ridge):
    X, y = _data(60, 3, seed=8)
    m = X.shape[0]
    mod = _call('7', LinearRegression, regularization=None, lr=0.1, method='batch', num_epochs=2000, bs=16)
    _call('7', mod.fit, X, y)
    _same('7', 'theta (batch, no regularization) vs closed form', mod.theta, _closed(X, y), rtol=1e-4, atol=1e-4)
    p = _call('7', mod.predict, X)
    _same('7', 'predict(X)', p, X @ mod.theta)
    lam = 0.2
    rid = _call('7', LinearRegression, regularization=_call('7', Ridge, lam), lr=0.1, method='batch', num_epochs=3000, bs=16)
    _call('7', rid.fit, X, y)
    _same('7', 'theta (batch, Ridge λ = 0.2) vs (XᵀX + 2mλI)⁻¹Xᵀy', rid.theta, _ridge_closed(X, y, 2 * m * lam), rtol=1e-4, atol=1e-4)
    for meth, lr in (('mini', 0.05), ('sto', 0.01)):
        mm = _call('7', LinearRegression, regularization=None, lr=lr, method=meth, num_epochs=200, bs=16)
        _call('7', mm.fit, X, y)
        _same('7', f"theta with method='{meth}' (approximately)", mm.theta, _closed(X, y), rtol=0.05, atol=0.08)
    return _pass('7', 'Your class reproduces the closed form, Ridge included. GD Ridge λ ↔ closed-form λ: multiply by 2m.')


# ---------------------------------------------------------------- level 8-9
@_guard
def check_8a(kfold_indices):
    from sklearn.model_selection import KFold
    for m, k in ((23, 5), (10, 3), (12, 4)):
        got = _call('8a', kfold_indices, m, k)
        exp = list(KFold(n_splits=k).split(np.zeros(m)))
        if len(got) != k:
            return _fail('8a', f'kfold_indices({m}, {k}) returned {len(got)} folds, expected {k}.')
        for f, ((gt, gv), (et, ev)) in enumerate(zip(got, exp)):
            if not (np.array_equal(np.sort(gv), ev) and np.array_equal(np.sort(gt), et)):
                return _fail('8a', f'fold {f} of kfold_indices({m}, {k}) differs from sklearn KFold.\n'
                                   f'   expected val {ev.tolist()}\n   got      val {np.sort(np.asarray(gv)).tolist()}')
    return _pass('8a', 'Same folds as sklearn KFold(n_splits=k). Note: when m isn\'t divisible by k, the FIRST folds get one extra sample.')


@_guard
def check_8b(cv_mse_ridge, best_lam, X, y, lams):
    from sklearn.model_selection import KFold
    ref = []
    for lam in lams:
        s = []
        for tr, va in KFold(n_splits=5).split(X):
            th = _ridge_closed(X[tr], y[tr], lam)
            s.append(_mse(X[va] @ th, y[va]))
        ref.append(np.mean(s))
    got = [_call('8b', cv_mse_ridge, X, y, lam, 5) for lam in lams]
    _same('8b', 'cv_mse_ridge for each λ', got, ref, rtol=1e-6)
    exp = lams[int(np.argmin(ref))]
    if best_lam != exp:
        return _fail('8b', f'best_lam = {best_lam}, but the λ with the lowest CV MSE is {exp}.')
    return _pass('8b', f'Best λ = {exp} (CV MSE {min(ref):.1f}). The DATA chose how much to regularize.')


@_guard
def check_9a(sk_model, theta_cf):
    if sk_model is None or theta_cf is None:
        return _fail('9a', 'fit sk_lr and compute theta_cf first.')
    _same('9a', 'sk_lr.intercept_ vs theta_cf[0]', sk_model.intercept_, theta_cf[0], rtol=1e-6, atol=1e-6)
    _same('9a', 'sk_lr.coef_ vs theta_cf[1:]', sk_model.coef_, theta_cf[1:], rtol=1e-6, atol=1e-6)
    return _pass('9a', 'sklearn intercept_ is your θ₀; coef_ is θ₁…θₙ.')


@_guard
def check_9b(sk_ridge, theta_ridge):
    if sk_ridge is None or theta_ridge is None:
        return _fail('9b', 'fit sk_ridge and compute theta_ridge first.')
    if getattr(sk_ridge, 'fit_intercept', True):
        return _fail('9b', 'use Ridge(alpha=10, fit_intercept=False): X already has the column of 1s.')
    _same('9b', 'sk_ridge.coef_ vs theta_ridge', sk_ridge.coef_, theta_ridge, rtol=1e-6, atol=1e-6)
    return _pass('9b', 'sklearn Ridge(alpha) solves (XᵀX + αI)θ = Xᵀy, exactly your ridge closed form.')


@_guard
def check_9c(gs, X_train_raw, y_train):
    from sklearn.linear_model import Ridge as _R
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if gs is None or not hasattr(gs, 'best_params_'):
        return _fail('9c', 'gs must be a FITTED GridSearchCV.')
    steps = getattr(gs.best_estimator_, 'named_steps', {})
    if not any(isinstance(s, StandardScaler) for s in steps.values()):
        return _fail('9c', 'put a StandardScaler inside the Pipeline so each CV fold is scaled with its own training part.')
    ref = GridSearchCV(Pipeline([('scaler', StandardScaler()), ('model', _R())]),
                       {'model__alpha': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='neg_mean_squared_error').fit(X_train_raw, y_train)
    ga = [v for k, v in gs.best_params_.items() if k.endswith('alpha')]
    if not ga or ga[0] != ref.best_params_['model__alpha']:
        return _fail('9c', f'best alpha {ga} differs from the reference {ref.best_params_["model__alpha"]}. '
                           "Check the grid [0.01, 0.1, 1, 10, 100], cv=5 and scoring='neg_mean_squared_error'.")
    return _pass('9c', f'Best alpha = {ref.best_params_["model__alpha"]}. gs.best_score_ is NEGATIVE MSE (sklearn always maximizes).')


# ---------------------------------------------------------------- level 10-13
@_guard
def check_10(sigmoid, bce, logistic_grad):
    _same('10', 'sigmoid(0)', _call('10', sigmoid, 0.0), 0.5)
    z = np.array([-2.0, 0.0, 1.0, 3.0])
    _same('10', 'sigmoid(array)', _call('10', sigmoid, z), _sig(z))
    X = np.array([[1.0, 2.0]])
    y = np.array([1.0])
    th = np.array([0.0, 0.5])
    h = _sig(X @ th)
    _same('10', 'bce on the Drill Lab example (y = 1, h = 0.7311)', _call('10', bce, y, h), 0.31326, rtol=1e-4)
    _same('10', 'logistic_grad on the Drill Lab example', _call('10', logistic_grad, X, y, th), [-0.26894, -0.53788], rtol=1e-4)
    yy = np.array([1, 1, 0, 0, 1.0])
    hh = np.array([0.99, 0.99, 0.01, 0.01, 0.99])
    _same('10', 'bce for good predictions', _call('10', bce, yy, hh), -np.sum(yy * np.log(hh) + (1 - yy) * np.log(1 - hh)))
    Xr, _ = _data(20, 3, seed=9)
    yr = (np.arange(20) % 2).astype(float)
    tr = np.array([0.1, -0.3, 0.2, 0.5])
    _same('10', 'logistic_grad on random data', _call('10', logistic_grad, Xr, yr, tr), Xr.T @ (_sig(Xr @ tr) - yr))
    return _pass('10', 'Same numbers as your Drill Lab screenshot: loss 0.3133, gradient [−0.269, −0.538].')


@_guard
def check_11a(fit_logistic, X, y):
    out = _call('11a', fit_logistic, X, y, 0.5, 300)
    if not (isinstance(out, tuple) and len(out) == 2):
        return _fail('11a', 'fit_logistic should return (theta, losses).')
    th, losses = out
    _same('11a', 'theta after 300 iterations (lr 0.5)', th, _logit_gd(X, y, 0.5, 300), rtol=1e-6, atol=1e-8)
    if len(losses) != 300:
        return _fail('11a', f'losses has {len(losses)} entries, expected 300.')
    acc = np.mean((_sig(X @ th) >= 0.5) == y)
    return _pass('11a', f'Training accuracy {acc:.3f}.')


@_guard
def check_11b(confusion, precision, recall, f1):
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    r = np.random.default_rng(11)
    yt = r.integers(0, 2, 60)
    yp = np.where(r.random(60) < 0.75, yt, 1 - yt)
    _same('11b', 'confusion(y_true, y_pred)  [[TN, FP], [FN, TP]]', _call('11b', confusion, yt, yp), confusion_matrix(yt, yp))
    _same('11b', 'precision', _call('11b', precision, yt, yp), precision_score(yt, yp))
    _same('11b', 'recall', _call('11b', recall, yt, yp), recall_score(yt, yp))
    _same('11b', 'f1', _call('11b', f1, yt, yp), f1_score(yt, yp))
    return _pass('11b', 'Matches sklearn. Layout: rows = TRUE class, columns = PREDICTED class.')


@_guard
def check_12(model, X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression as _SkLR
    if model is None:
        return _fail('12', 'train your model first.')
    p = _call('12', model.predict, X_test)
    p = np.asarray(p)
    if p.shape != y_test.shape:
        return _fail('12', f'predict returned shape {p.shape}, expected {y_test.shape}.')
    if not set(np.unique(p)).issubset({0, 1}):
        return _fail('12', 'predict should return class labels 0/1, not probabilities. (Use predict_proba for those.)')
    pr = np.asarray(_call('12', model.predict_proba, X_test))
    if pr.shape != y_test.shape or pr.min() < 0 or pr.max() > 1:
        return _fail('12', 'predict_proba should return one probability in [0, 1] per sample.')
    acc = np.mean(p == y_test)
    sk = _SkLR(max_iter=5000).fit(X_train[:, 1:], y_train)
    sk_acc = sk.score(X_test[:, 1:], y_test)
    if acc < sk_acc - 0.03:
        return _fail('12', f'test accuracy {acc:.3f}; sklearn gets {sk_acc:.3f}. Get within 0.03: try more iterations, a different lr, or check the gradient.')
    return _pass('12', f'BOSS DOWN. Your accuracy {acc:.3f} vs sklearn {sk_acc:.3f}. You wrote this from scratch.')


@_guard
def check_13a(softmax, one_hot):
    Z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1001.0]])
    with np.errstate(all='ignore'):
        S = _call('13', softmax, Z)
    S = np.asarray(S, dtype=float)
    if S.shape == Z.shape and np.isnan(S).any():
        return _fail('13', 'softmax gives nan for large scores like 1000 (e^1000 overflows). '
                           'Subtract each row\'s max first: Z - Z.max(axis=1, keepdims=True). The answer doesn\'t change, only the overflow does.')
    Zs = Z - Z.max(1, keepdims=True)
    ref = np.exp(Zs) / np.exp(Zs).sum(1, keepdims=True)
    _same('13', 'softmax(Z)', S, ref)
    _same('13', 'one_hot([2, 0, 1], 3)', _call('13', one_hot, np.array([2, 0, 1]), 3), np.eye(3)[[2, 0, 1]])
    print('   softmax and one_hot look right. Now train it (13b).')
    return True


@_guard
def check_13b(y_pred, y_test):
    y_pred = np.asarray(y_pred)
    if y_pred.shape != y_test.shape:
        return _fail('13', f'y_pred has shape {y_pred.shape}, expected {y_test.shape}. Did you take argmax(axis=1)?')
    acc = np.mean(y_pred == y_test)
    if acc < 0.85:
        return _fail('13', f'test accuracy {acc:.3f}; aim for at least 0.85.')
    return _pass('13', f'Softmax regression, 3 classes, test accuracy {acc:.3f}. Gradient was Xᵀ(H − Y) again.')
