"""
Dataset Lie Detector
====================

A small, reusable toolkit that tells you whether a dataset can be trusted
before you spend hours modelling it.

It answers four questions:

1. Signal:      Do the labels carry any learnable pattern, or are they noise?
2. Integrity:   Do the tables link together the way they claim to?
3. Time:        Do events happen in a possible order?
4. Plausibility: Do business rules and expected relationships hold?

Every check adds a finding (PASS / WARN / FAIL) to a scorecard, and the
scorecard rolls up into a single Trust Score out of 100.

Author: Collins Lemeke (github.com/CollinsLemeke, kaggle.com/collinslemeke)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

__version__ = "1.0.0"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_ICON = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}
_POINTS = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}


@dataclass
class Finding:
    group: str      # Signal / Integrity / Time / Plausibility
    check: str      # short name of the check
    status: str     # PASS / WARN / FAIL
    metric: str     # the headline number, as text
    detail: str     # one plain-English sentence


class LieDetector:
    """Collects checks on a dataset and produces a scorecard.

    Parameters
    ----------
    name : str
        Name shown in the report.
    warn_rate, fail_rate : float
        Thresholds for rate-based checks (share of rows breaking a rule).
        Default: above 0.1% is a warning, above 5% is a failure.
    random_state : int
        Seed for sampling, cross-validation and label shuffling.
    """

    def __init__(self, name: str = "dataset", warn_rate: float = 0.001,
                 fail_rate: float = 0.05, random_state: int = 42):
        self.name = name
        self.warn_rate = warn_rate
        self.fail_rate = fail_rate
        self.random_state = random_state
        self.findings: list[Finding] = []
        self.signal_results: list[dict] = []

    # ------------------------------------------------------------------ utils
    def _add(self, group, check, status, metric, detail, verbose=True):
        f = Finding(group, check, status, metric, detail)
        self.findings.append(f)
        if verbose:
            print(f"{_ICON[status]} [{group}] {check}: {metric}  |  {detail}")
        return f

    @staticmethod
    def _pct(rate):
        if 0 < rate < 0.0001:
            return "<0.01%"
        if 0.9999 < rate < 1:
            return ">99.99%"
        return f"{rate:.2%}"

    def _rate_status(self, rate):
        if rate > self.fail_rate:
            return FAIL
        if rate > self.warn_rate:
            return WARN
        return PASS

    @staticmethod
    def _prepare_features(df, features):
        X = df[features].copy()
        for c in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[c]):
                X[c] = X[c].astype("int64") // 86_400_000_000_000  # days since epoch
            elif pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype(int)
            elif not pd.api.types.is_numeric_dtype(X[c]):
                # text or category columns (works for both old object and new str dtypes)
                X[c] = X[c].astype(str).astype("category")
        return X

    def _cv_auc(self, X, y, cv, seed):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = []
        for tr, te in skf.split(X, y):
            model = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.1, categorical_features="from_dtype",
                random_state=seed)
            model.fit(X.iloc[tr], y.iloc[tr])
            scores.append(roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1]))
        return float(np.mean(scores))

    # ============================================================ 1. SIGNAL
    def label_signal(self, df: pd.DataFrame, target: str, features: list[str] | None = None,
                     sample: int = 200_000, n_permutations: int = 5, cv: int = 3,
                     name: str | None = None, verbose: bool = True) -> dict:
        """Can a model learn the target at all?

        Trains a gradient-boosted model with cross-validation and compares its
        ROC AUC against the same model trained on SHUFFLED labels (the "null").
        If the real AUC is not clearly above the null, the labels are noise.

        Verdict: PASS if AUC >= 0.60 and it beats the null by > 3 standard
        deviations; FAIL if AUC < 0.55 or it does not beat the null; WARN between.
        """
        name = name or target
        y_all = df[target]
        if y_all.nunique() != 2:
            raise ValueError("label_signal supports binary targets only")
        if features is None:
            features = [c for c in df.columns
                        if c != target and not c.lower().endswith("_id") and c.lower() != "id"]

        rng = np.random.RandomState(self.random_state)
        if len(df) > sample:
            # stratified sample keeps the positive rate intact
            frac = sample / len(df)
            idx = np.concatenate([
                rng.choice(g.index.values, size=max(1, int(round(len(g) * frac))), replace=False)
                for _, g in df.groupby(target)])
            data = df.loc[idx]
        else:
            data = df
        X = self._prepare_features(data, features)
        y = data[target].astype(int).reset_index(drop=True)
        X = X.reset_index(drop=True)

        auc = self._cv_auc(X, y, cv, self.random_state)
        null = []
        for i in range(n_permutations):
            y_shuffled = pd.Series(rng.permutation(y.values))
            null.append(self._cv_auc(X, y_shuffled, cv, self.random_state + i + 1))
        null = np.array(null)
        null_mean, null_std = float(null.mean()), float(max(null.std(ddof=1), 1e-3))
        z = (auc - null_mean) / null_std

        if auc >= 0.60 and z > 3:
            status, verdict = PASS, "the labels carry a learnable pattern"
        elif auc < 0.55 or z <= 3:
            status, verdict = FAIL, "no better than shuffled labels, so the labels look like noise"
        else:
            status, verdict = WARN, "weak signal, model with care"

        res = dict(name=name, target=target, auc=auc, null_mean=null_mean,
                   null_std=null_std, null=null.tolist(), z=z, n_rows=len(y),
                   positive_rate=float(y.mean()), n_features=len(features), status=status)
        self.signal_results.append(res)
        self._add("Signal", f"Label signal: {name}", status,
                  f"AUC {auc:.3f} vs shuffled {null_mean:.3f}",
                  f"{verdict} ({len(features)} features, {len(y):,} rows).", verbose)
        return res

    def label_spread(self, df: pd.DataFrame, target: str, by: str,
                     verbose: bool = True) -> pd.DataFrame:
        """Does the target rate change across groups of a column?

        Returns the rate per group and runs a chi-square test. Real labels
        usually vary across at least some groups; flat rates are a red flag.
        """
        table = (df.groupby(by, observed=True)[target]
                   .agg(rate="mean", n="size").sort_values("rate", ascending=False))
        ct = pd.crosstab(df[by], df[target])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        # Cramer's V: effect size, 0 = no association, 1 = perfect
        n = ct.values.sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        status = PASS if (p < 0.01 and v >= 0.02) else WARN
        spread = table["rate"].max() - table["rate"].min()
        self._add("Signal", f"{target} by {by}", status,
                  f"Cramer's V {v:.3f}, p={p:.2g}",
                  f"rate ranges {table['rate'].min():.2%} to {table['rate'].max():.2%} "
                  f"(spread {spread:.2%}).", verbose)
        return table

    # ========================================================= 2. INTEGRITY
    def foreign_key(self, child: pd.DataFrame, key: str, parent: pd.DataFrame,
                    parent_key: str | None = None, name: str | None = None,
                    verbose: bool = True) -> float:
        """Share of child rows whose key does not exist in the parent table."""
        parent_key = parent_key or key
        rate = float((~child[key].isin(parent[parent_key])).mean())
        self._add("Integrity", name or f"Foreign key {key}", self._rate_status(rate),
                  f"{self._pct(rate)} orphans",
                  "every ID points to a real record." if rate == 0 else "rows point to a record that does not exist.", verbose)
        return rate

    def agreement(self, name: str, left: pd.Series, right: pd.Series,
                  detail: str = "", verbose: bool = True) -> float:
        """Two columns that should always match (e.g. the owner of a card vs
        the owner of the account it is linked to). Returns the mismatch rate."""
        rate = float((left.values != right.values).mean())
        self._add("Integrity", name, self._rate_status(rate), f"{self._pct(rate)} mismatched",
                  "values agree." if rate == 0 else (detail or "values that should agree do not."), verbose)
        return rate

    def duplicates(self, df: pd.DataFrame, subset: list[str] | None = None,
                   name: str = "Duplicate rows", verbose: bool = True) -> float:
        rate = float(df.duplicated(subset=subset).mean())
        self._add("Integrity", name, self._rate_status(rate), f"{self._pct(rate)} duplicated",
                  "no repeated rows." if rate == 0 else "identical rows repeated.", verbose)
        return rate

    # ============================================================== 3. TIME
    def date_order(self, name: str, earlier: pd.Series, later: pd.Series,
                   detail: str = "", verbose: bool = True) -> float:
        """Share of rows where `later` happens before `earlier`."""
        e, l = pd.to_datetime(earlier).values, pd.to_datetime(later).values
        rate = float((l < e).mean())
        self._add("Time", name, self._rate_status(rate), f"{self._pct(rate)} impossible",
                  "events happen in a possible order." if rate == 0 else (detail or "events happen in the wrong order."), verbose)
        return rate

    # ====================================================== 4. PLAUSIBILITY
    def rule(self, name: str, violations: pd.Series, detail: str = "",
             verbose: bool = True) -> float:
        """Any custom business rule. Pass a boolean Series that is True where
        the rule is BROKEN."""
        rate = float(pd.Series(violations).mean())
        self._add("Plausibility", name, self._rate_status(rate), f"{self._pct(rate)} break the rule",
                  "rule holds." if rate == 0 else (detail or "rows that break a real-world rule."), verbose)
        return rate

    def expected_relationship(self, name: str, x: pd.Series, y: pd.Series,
                              direction: str = "positive", min_abs: float = 0.05,
                              detail: str = "", verbose: bool = True) -> float:
        """A relationship you would expect in real life (e.g. higher credit
        score -> lower interest rate). Uses Spearman correlation. WARN, not
        FAIL, because real data can legitimately surprise you."""
        rho, p = stats.spearmanr(x, y, nan_policy="omit")
        ok = (rho >= min_abs) if direction == "positive" else (rho <= -min_abs)
        status = PASS if (ok and p < 0.01) else WARN
        self._add("Plausibility", name, status, f"Spearman {rho:+.3f}",
                  detail or f"expected a {direction} relationship.", verbose)
        return float(rho)

    # ============================================================ REPORTING
    def trust_score(self) -> float:
        """0 to 100. PASS = full marks, WARN = half, FAIL = zero. Signal checks
        count double, because labels are what a model learns from."""
        if not self.findings:
            return float("nan")
        w = np.array([2.0 if f.group == "Signal" and f.check.startswith("Label signal") else 1.0
                      for f in self.findings])
        pts = np.array([_POINTS[f.status] for f in self.findings])
        return float(100 * (w * pts).sum() / w.sum())

    def report(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(f) for f in self.findings])
        if df.empty:
            return df
        df.insert(2, "result", df["status"].map(lambda s: f"{_ICON[s]} {s}"))
        return df.drop(columns="status")

    def summary(self) -> str:
        counts = pd.Series([f.status for f in self.findings]).value_counts()
        score = self.trust_score()
        verdict = ("Trustworthy for ML" if score >= 80 else
                   "Use with caution" if score >= 50 else
                   "Not suitable for ML as-is")
        return (f"{self.name}: Trust Score {score:.0f}/100 ({verdict}) | "
                f"{counts.get(PASS, 0)} pass, {counts.get(WARN, 0)} warn, {counts.get(FAIL, 0)} fail")
