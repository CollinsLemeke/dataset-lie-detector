"""Builds notebook.ipynb from cells below. Run: python build_notebook.py"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "lie_detector.py").read_text()
REPO = "https://github.com/CollinsLemeke/dataset-lie-detector"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(f"""
# 🕵️ Dataset Lie Detector: Is Your Data Lying to You?

**A reusable toolkit that tells you, in minutes, whether a dataset can be trusted for machine learning. Tested here on 5.9 million rows of banking data.**

Before you spend a weekend tuning XGBoost on a dataset, it is worth asking one simple question:

> *Is there actually anything here for a model to learn?*

Most notebooks skip that question. This one answers it, and gives you a tool to answer it for **any** dataset.

**🔗 GitHub:** [{REPO}]({REPO}) &nbsp;|&nbsp; **Author:** Collins Lemeke

---

### TL;DR

- **All three ML labels are noise.** Fraud, late payment and loan default models score **ROC AUC 0.50 to 0.52**, the same as models trained on randomly shuffled labels. In plain English: coin flips.
- **The detector is proven, not just pessimistic.** It scores **0.99** on a real medical dataset and **0.82** on a realistic fraud pattern planted into the *same* 3 million transactions, and marks both PASS.
- **The tables look clean but do not connect.** Every ID is valid and there are zero missing values, yet **over 99.99%** of cards are linked to an account owned by someone else, **47%** of loan payments happen before the loan starts, and **14%** of customers joined the bank as children.
- **Trust Score: 35/100.** Great for SQL and Power BI practice. Not suitable for training ML models.

---

### What's inside

1. **The toolkit**: one class, `LieDetector`, that you can copy into any project
2. **Check 1, Signal**: do the labels carry a real pattern, or are they coin flips?
3. **Proof the detector works**: it must say PASS on real signal, not just FAIL everything
4. **Check 2, Integrity**: do the tables link together correctly?
5. **Check 3, Time**: do events happen in a possible order?
6. **Check 4, Plausibility**: do real-world rules and relationships hold?
7. **The verdict**: a scorecard and a Trust Score out of 100
8. **Use it on your own data** in 5 lines
""")

md("""
## A note on fairness

This dataset's description says it was built for **SQL, Python and Power BI practice**, and for that purpose it is clean, large and well organised (zero missing values across 5.9M rows). Nothing here is a criticism of the creator.

The point is narrower: **if you plan to train ML models on it, you should know what the labels can and cannot support.** The same checks apply to any dataset you find online, including real ones.
""")

md("""
## 1. The toolkit

The cell below writes `lie_detector.py` to disk and imports it. It is self-contained (pandas, numpy, scipy, scikit-learn, all preinstalled on Kaggle).

**How the key test works, in plain English:**

- Train a model to predict the label and measure its ROC AUC (0.5 = coin flip, 1.0 = perfect).
- Then **shuffle the labels randomly** and train the same model again, several times. This tells us what "pure luck" looks like.
- If the real model is not clearly better than the shuffled ones, **the labels contain no learnable pattern.**

This is called a *permutation test*, and it is one of the most honest checks in machine learning.
""")

code("%%writefile lie_detector.py\n" + SRC)

code("""
import glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from lie_detector import LieDetector, PASS, WARN, FAIL

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 120)

# Chart style: quiet axes, one accent colour, status colours reserved for verdicts
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
STATUS = {PASS: "#0ca30c", WARN: "#fab219", FAIL: "#d03b3b"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
""")

md("""
## 2. Load the data

Ten tables, linked by ID columns, like a real bank database. The loader finds the files wherever Kaggle mounts them.
""")

code("""
hits = glob.glob("/kaggle/input/**/card_transactions.csv", recursive=True)
DATA_DIR = os.path.dirname(hits[0]) if hits else os.environ.get("BANK_DATA", "./data")
print("Reading from:", DATA_DIR if hits else "local ./data folder")

dates = {
    "customers": ["date_of_birth", "join_date"], "accounts": ["open_date"],
    "cards": ["issue_date", "expiry_date"], "card_transactions": ["txn_date"],
    "transactions": ["txn_date"], "loans": ["start_date"], "loan_payments": ["payment_date"],
    "support_tickets": ["date_opened", "date_resolved"], "branches": ["opened_date"],
    "employees": ["hire_date"],
}
T = {name: pd.read_csv(f"{DATA_DIR}/{name}.csv", parse_dates=cols) for name, cols in dates.items()}

overview = pd.DataFrame({
    "rows": {k: len(v) for k, v in T.items()},
    "columns": {k: v.shape[1] for k, v in T.items()},
    "missing values": {k: int(v.isna().sum().sum()) for k, v in T.items()},
}).sort_values("rows", ascending=False)
display(overview)
print(f"Total rows: {overview['rows'].sum():,}")
""")

code("""
ld = LieDetector(name="Banking Transactions Dataset")
cu, ac, ca, ct, tx = T["customers"], T["accounts"], T["cards"], T["card_transactions"], T["transactions"]
lo, lp, st, br = T["loans"], T["loan_payments"], T["support_tickets"], T["branches"]
""")

md("""
## 3. Check 1: Do the labels carry any signal?

The dataset has three natural ML targets:

| Target | Question a model would answer |
|---|---|
| `is_fraud` | Is this card transaction fraudulent? |
| `late_payment_flag` | Will this loan payment be late? |
| `status = Defaulted / Written Off` | Will this loan go bad? |

For each, I build the features a data scientist would reasonably use, joining tables together, then run the permutation test.
""")

md("### 3a. Card fraud")

code("""
fraud = (ct.merge(ca[["card_id", "customer_id", "card_type", "credit_limit", "issue_date"]], on="card_id")
           .merge(cu[["customer_id", "annual_income", "credit_score", "occupation", "date_of_birth"]], on="customer_id")
           .sort_values(["card_id", "txn_date"]))

# Classic fraud features: how unusual is this spend for this card, and how fast are transactions coming?
fraud["amount_vs_card_avg"] = fraud["amount"] / fraud.groupby("card_id")["amount"].transform("mean")
fraud["days_since_last_txn"] = fraud.groupby("card_id")["txn_date"].diff().dt.days
fraud["card_age_days"] = (fraud["txn_date"] - fraud["issue_date"]).dt.days
fraud["customer_age"] = (fraud["txn_date"] - fraud["date_of_birth"]).dt.days / 365.25
fraud["day_of_week"] = fraud["txn_date"].dt.dayofweek
fraud["month"] = fraud["txn_date"].dt.month

FRAUD_FEATURES = ["amount", "merchant_category", "card_type", "credit_limit", "amount_vs_card_avg",
                  "days_since_last_txn", "card_age_days", "annual_income", "credit_score",
                  "occupation", "customer_age", "day_of_week", "month"]
r_fraud = ld.label_signal(fraud, "is_fraud", FRAUD_FEATURES, name="Card fraud")
""")

md("### 3b. Late loan payments and loan defaults")

code("""
loans = lo.merge(cu[["customer_id", "annual_income", "credit_score", "occupation"]], on="customer_id")
pay = lp.merge(loans[["loan_id", "loan_type", "loan_amount", "interest_rate", "term_months",
                      "annual_income", "credit_score", "occupation"]], on="loan_id")

LATE_FEATURES = ["amount_paid", "loan_type", "loan_amount", "interest_rate", "term_months",
                 "annual_income", "credit_score", "occupation"]
r_late = ld.label_signal(pay, "late_payment_flag", LATE_FEATURES, name="Late payment")

# Loan level: did the loan go bad? Includes the loan's own late-payment history, a strong real-world predictor
loans["late_rate"] = loans["loan_id"].map(lp.groupby("loan_id")["late_payment_flag"].mean())
loans["bad_loan"] = loans["status"].isin(["Defaulted", "Written Off"]).astype(int)
DEFAULT_FEATURES = ["loan_type", "loan_amount", "interest_rate", "term_months",
                    "annual_income", "credit_score", "occupation", "late_rate"]
r_default = ld.label_signal(loans, "bad_loan", DEFAULT_FEATURES, name="Loan default")
""")

md("""
## 4. Proof the detector works (positive controls)

A detector that says FAIL to everything is useless. So before trusting the result above, I test it on two cases where I **know** a real pattern exists:

1. **Breast cancer diagnosis** (scikit-learn's built-in, real clinical dataset).
2. **A planted pattern**: I take the *same* bank card transactions and create a new fraud label that follows realistic rules (unusually large spend for that card, travel and shopping merchants, bigger amounts), with randomness added. Same rows, same features, only the label changes.

If the detector is honest, both should PASS. These go into a separate detector so they don't affect the bank's score.
""")

code("""
from sklearn.datasets import load_breast_cancer

controls = LieDetector(name="Positive controls")
r_cancer = controls.label_signal(load_breast_cancer(as_frame=True).frame, "target", name="Breast cancer (real)")

rng = np.random.RandomState(0)
logit = (-6.0
         + 2.0 * (fraud["amount_vs_card_avg"] > 2.5)
         + 1.5 * fraud["merchant_category"].isin(["Travel", "Shopping"])
         + 0.0002 * fraud["amount"])
fraud["planted_fraud"] = (rng.rand(len(fraud)) < 1 / (1 + np.exp(-logit))).astype(int)
print(f"Planted fraud rate: {fraud['planted_fraud'].mean():.2%}")
r_planted = controls.label_signal(fraud, "planted_fraud", FRAUD_FEATURES, name="Planted fraud pattern")
""")

md("""
### The picture: real model vs pure luck

Each row is one label. The **grey band** is the range a model reaches with shuffled (meaningless) labels. The **dot** is the real model. A trustworthy label sits far to the right of its grey band.
""")

code("""
def plot_signal(results):
    results = results[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(results) + 1.4))
    for i, r in enumerate(results):
        lo_, hi_ = r["null_mean"] - 3 * r["null_std"], r["null_mean"] + 3 * r["null_std"]
        ax.barh(i, hi_ - lo_, left=lo_, height=0.42, color="#d6d5cf", zorder=1)
        ax.plot([r["null_mean"], r["auc"]], [i, i], color=MUTED, lw=1, zorder=2)
        ax.scatter(r["auc"], i, s=110, color=STATUS[r["status"]], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{r['auc']:.3f}  {r['status']}", (r["auc"], i), xytext=(12, -4),
                    textcoords="offset points", fontsize=10, color=INK)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_ylim(-0.9, len(results) - 0.5)
    ax.text(0.505, -0.75, "coin flip (0.5)", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(results)), [r["name"] for r in results])
    ax.set_xlim(0.4, 1.08)
    ax.set_xlabel("ROC AUC (cross-validated)")
    ax.set_title("Can a model learn the label? Real model (dot) vs shuffled labels (grey band)")
    ax.grid(axis="y", visible=False)
    plt.tight_layout(); plt.show()

plot_signal(controls.signal_results + ld.signal_results)
""")

md("""
### Look closer: noise vs signal, side by side

Here is the same comparison without any model. On the left, the dataset's fraud rate by merchant category. On the right, the planted (realistic) fraud label on the **exact same transactions**.

Real fraud is never spread perfectly evenly. When every category sits on the same line, the label was almost certainly assigned at random.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, col, title, colour in [(axes[0], "is_fraud", "Dataset label is_fraud: flat", BLUE),
                               (axes[1], "planted_fraud", "Planted realistic pattern: varies", ORANGE)]:
    rates = fraud.groupby("merchant_category")[col].mean().sort_values()
    ax.barh(rates.index, rates.values * 100, color=colour, height=0.7)
    ax.axvline(fraud[col].mean() * 100, color=INK, ls="--", lw=1)
    ax.set_title(title); ax.set_xlabel("Fraud rate (%)")
    ax.grid(axis="y", visible=False)
plt.suptitle("Same 3 million transactions, two different labels", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.show()

# Statistical version of the same idea (chi-square test + effect size)
for col in ["merchant_category", "card_type"]:
    ld.label_spread(fraud, "is_fraud", col)
_ = controls.label_spread(fraud, "planted_fraud", "merchant_category")
""")

code("""
# Late payments: in real lending, lower credit scores pay late far more often
pay["credit_score_band"] = pd.cut(pay["credit_score"], [299, 450, 550, 650, 750, 900],
                                  labels=["300-450", "451-550", "551-650", "651-750", "751-900"])
band = pay.groupby("credit_score_band", observed=True)["late_payment_flag"].mean() * 100

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(band.index.astype(str), band.values, marker="o", color=BLUE, lw=2, ms=8)
for x, y in zip(band.index.astype(str), band.values):
    ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10)
ax.set_ylim(0, 25); ax.set_ylabel("Late payment rate (%)"); ax.set_xlabel("Credit score band")
ax.set_title("Late payment rate is the same for every credit score band")
plt.tight_layout(); plt.show()

ld.label_spread(pay, "late_payment_flag", "credit_score_band")
_ = ld.label_spread(pay, "late_payment_flag", "loan_type")
""")

md("""
## 5. Check 2: Do the tables link together correctly?

In a relational database, every ID should point to a record that exists, and linked records should agree with each other. For example, a card belongs to a customer **and** to an account, so the account should belong to that same customer.
""")

code("""
ld.foreign_key(ac, "customer_id", cu, name="Accounts -> customers exist")
ld.foreign_key(ca, "account_id", ac, name="Cards -> accounts exist")
ld.foreign_key(ct, "card_id", ca, name="Card transactions -> cards exist")
ld.foreign_key(tx, "account_id", ac, name="Transactions -> accounts exist")
ld.foreign_key(lp, "loan_id", lo, name="Loan payments -> loans exist")

card_acct = ca.merge(ac[["account_id", "customer_id"]], on="account_id", suffixes=("", "_of_account"))
ld.agreement("Card owner = owner of its linked account",
             card_acct["customer_id"], card_acct["customer_id_of_account"],
             "a card is linked to an account that belongs to someone else.")
_ = ld.duplicates(ct.drop(columns="card_txn_id"), name="Duplicate card transactions")
""")

md("""
## 6. Check 3: Does time make sense?

Money cannot move through an account before it is opened, and a loan cannot be repaid before it starts.
""")

code("""
t = tx.merge(ac[["account_id", "open_date"]], on="account_id")
ld.date_order("Transaction after account opened", t["open_date"], t["txn_date"],
              "money moves through accounts that do not exist yet.")

c = ct.merge(ca[["card_id", "issue_date", "expiry_date"]], on="card_id")
ld.date_order("Card used after it was issued", c["issue_date"], c["txn_date"],
              "cards are used before they were issued.")
ld.date_order("Card used before it expired", c["txn_date"], c["expiry_date"],
              "cards are used after they expired.")

p = lp.merge(lo[["loan_id", "start_date"]], on="loan_id")
ld.date_order("Loan payment after loan started", p["start_date"], p["payment_date"],
              "loans are repaid before they begin.")

a = ac.merge(br[["branch_id", "opened_date"]], on="branch_id")
ld.date_order("Account opened after its branch opened", a["opened_date"], a["open_date"],
              "accounts open at branches that do not exist yet.")

_ = ld.date_order("Support ticket resolved after it was opened", st["date_opened"], st["date_resolved"])
""")

md("""
## 7. Check 4: Do real-world rules and relationships hold?

**Rules** are things that should never happen. **Relationships** are patterns you would expect in real banking data (they get a softer WARN, because real data can surprise you).
""")

code("""
age_at_join = (cu["join_date"] - cu["date_of_birth"]).dt.days / 365.25
ld.rule("Customers are adults when they join", age_at_join < 18, "customers joined the bank as children.")

is_debit = ca["card_type"].eq("Debit")
ld.rule("Debit cards have no credit limit", is_debit & (ca["credit_limit"] > 0),
        "debit cards carry a credit limit (only credit cards should).")
ld.rule("Credit cards have a credit limit", ~is_debit & (ca["credit_limit"] <= 0),
        "credit cards with a limit of zero.")

pay_count = lp.groupby("loan_id").size().reindex(lo["loan_id"]).fillna(0).values
ld.rule("Loan has no more payments than its term", pay_count > lo["term_months"].values,
        "loans with more monthly payments than months in the term.")

days_to_resolve = (st["date_resolved"] - st["date_opened"]).dt.days
lc = lo.merge(cu, on="customer_id")
ld.expected_relationship("Higher credit score -> lower interest rate",
                         lc["credit_score"], lc["interest_rate"], direction="negative")
ld.expected_relationship("Higher income -> higher credit score", cu["annual_income"], cu["credit_score"])
ld.expected_relationship("Higher income -> bigger loans", lc["annual_income"], lc["loan_amount"])
_ = ld.expected_relationship("Slower ticket resolution -> lower satisfaction",
                             days_to_resolve, st["satisfaction_score"], direction="negative")
""")

md("## 8. The verdict")

code("""
print(ld.summary())
print(controls.summary())
display(ld.report())
""")

code("""
rep = pd.DataFrame([vars(f) for f in ld.findings])
counts = rep.groupby(["group", "status"]).size().unstack(fill_value=0).reindex(columns=[PASS, WARN, FAIL], fill_value=0)
counts = counts.reindex(["Signal", "Integrity", "Time", "Plausibility"])

fig, ax = plt.subplots(figsize=(10, 3.6))
left = np.zeros(len(counts))
for status in [PASS, WARN, FAIL]:
    ax.barh(counts.index, counts[status], left=left, color=STATUS[status], height=0.6,
            label=status, edgecolor=SURFACE, linewidth=2)
    for i, (l, v) in enumerate(zip(left, counts[status])):
        if v:
            ax.text(l + v / 2, i, str(v), ha="center", va="center", fontweight="bold",
                    color=INK if status == WARN else "white")
    left += counts[status].values
ax.invert_yaxis(); ax.set_xlabel("Number of checks"); ax.grid(axis="y", visible=False)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
ax.set_title(f"Trust Score: {ld.trust_score():.0f}/100. Clean tables, but the labels and links do not hold up")
plt.tight_layout(); plt.show()
""")

md("""
## 9. What this means

**What the dataset is good for ✅**
- Practising SQL joins, aggregations and window functions on a realistic 10-table schema
- Building Power BI or Tableau dashboards at scale (5.9M rows, zero missing values)
- Practising data engineering: loading, indexing, partitioning

**What it cannot support ❌**
- Fraud detection, late payment or default prediction models. The labels behave like coin flips, so any model will score around AUC 0.5.
- Any "insight" that connects tables (for example, *customers with low credit scores default more*). The relationships were not generated, so such findings would be artefacts.

**A warning sign to watch for:** if you see a notebook on a dataset like this reporting **99% accuracy on fraud**, check the class balance. With 0.5% fraud, a model that always says *"not fraud"* scores 99.5% accuracy while learning nothing. Use ROC AUC, PR AUC or recall instead.

**What a fix would look like:** generate the data so that labels depend on behaviour (spending spikes, repeat transactions, credit score) and linked tables share the same owners and dates. The *planted pattern* in Section 4 is a tiny example of exactly that.
""")

md("""
## 10. Use it on your own data

Copy `lie_detector.py` (it is written to this notebook's output, and on GitHub), then:

```python
from lie_detector import LieDetector

ld = LieDetector("My dataset")
ld.label_signal(df, target="churn")                           # 1. Is there signal?
ld.foreign_key(orders, "customer_id", customers)              # 2. Do tables link?
ld.date_order("Shipped after ordered", df.order_date, df.ship_date)  # 3. Does time make sense?
ld.rule("No negative prices", df.price < 0)                    # 4. Do rules hold?
print(ld.summary()); ld.report()
```

**Fork this notebook**, swap in your own dataset, and run it before you build your next model.

---

If this saved you a few hours, an **upvote** helps other people find it before they spend a weekend modelling noise. ⭐ the repo on [GitHub](""" + REPO + """) if you want to follow updates. Questions and ideas for new checks are welcome in the comments.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, HERE / "notebook.ipynb")
print("wrote notebook.ipynb with", len(cells), "cells")
