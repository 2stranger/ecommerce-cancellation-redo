"""Order cancellation model, solo redo of a class project.

Run:
    python cancellation_model.py "path/to/Amazon Sale Report.csv"

Writes tables and metrics to outputs/ and charts to charts/.
Every number in README.md comes from these files.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy import sparse
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 28
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CHARTS = ROOT / "charts"
OUT.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

# Locked in PLAN.md before modeling.
TRAIN_END = "2022-05-15"
VAL_START, VAL_END = "2022-05-16", "2022-06-05"
TEST_START = "2022-06-06"
OPEN_STATUSES = ["Pending", "Pending - Waiting for Pick Up", "Shipping"]
COST_FN = 3.0  # assumption: a missed cancellation costs 3 false alarms
COST_FP = 1.0
CLASS_AMOUNT_FILL = 645.754204  # value the group's cleaning put in missing or zero Amount

# Class version results, copied from the rendered class report (XGBoost_Anneti.html).
# These were never validated; they are listed only for comparison.
CLASS_REPORTED = {
    "roc_auc_best": 0.9463969,
    "roc_auc_final": 0.9462954,
    "sensitivity_final": 0.8392608,
    "accuracy_final": 0.8757562,
    "sensitivity_logreg": 0.3518131,
    "roc_auc_logreg": 0.8983188,
}

RESULTS = {}
DEV = "--dev" in sys.argv  # debug runs never touch the test set


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%m-%d-%y")
    RESULTS["data"] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "distinct_orders": int(df["Order ID"].nunique()),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "status_cancelled": int((df["Status"] == "Cancelled").sum()),
        "class_definition_positives": int(class_target(df).sum()),
        "open_rows_dropped": int(df["Status"].isin(OPEN_STATUSES).sum()),
    }
    return df


def class_target(df):
    return (
        (df["Status"] == "Cancelled")
        | df["Courier Status"].isin(["Cancelled", "Unshipped"])
    ).astype(int)


def split_of(dates):
    return np.select(
        [dates <= TRAIN_END, dates <= VAL_END], ["train", "val"], default="test"
    )


# ---------------------------------------------------------------- leak audit
def leak_audit(df):
    """Compare raw fields for cancelled vs other resolved orders."""
    d = df[~df["Status"].isin(OPEN_STATUSES)]
    c = d["Status"] == "Cancelled"
    checks = {
        "Qty is 0": d["Qty"] == 0,
        "Amount missing": d["Amount"].isna(),
        "Amount is 0": d["Amount"] == 0,
        "currency missing": d["currency"].isna(),
        "promotion-ids missing": d["promotion-ids"].isna(),
        "promotion-ids missing (Merchant orders)": d["promotion-ids"].isna().where(
            d["Fulfilment"] == "Merchant"
        ),
        "promotion-ids missing (Amazon orders)": d["promotion-ids"].isna().where(
            d["Fulfilment"] == "Amazon"
        ),
        "Courier Status Unshipped": d["Courier Status"] == "Unshipped",
        "Courier Status missing": d["Courier Status"].isna(),
        "fulfilled-by missing": d["fulfilled-by"].isna(),
        "Fulfilment is Merchant": d["Fulfilment"] == "Merchant",
        "B2B": d["B2B"].astype(bool),
        "ship-state missing": d["ship-state"].isna(),
    }
    verdict = {
        "Qty is 0": "Leak. Drop Qty.",
        "Amount missing": "Leak. Do not use the row's own Amount.",
        "Amount is 0": "Small. Covered by not using the row's own Amount.",
        "currency missing": "Leak. Same rows as missing Amount.",
        "promotion-ids missing": "Leak. Drop promotion features.",
        "promotion-ids missing (Merchant orders)": "Leak. Promotions look written at invoicing.",
        "promotion-ids missing (Amazon orders)": "Leak.",
        "Courier Status Unshipped": "Recorded after the outcome. Not a feature.",
        "Courier Status missing": "Recorded after the outcome. Not a feature.",
        "fulfilled-by missing": "Small gap, tracks Fulfilment. Keep.",
        "Fulfilment is Merchant": "Known at order time. Keep.",
        "B2B": "Known at order time. Keep.",
        "ship-state missing": "No gap. Keep.",
    }
    rows = []
    for name, flag in checks.items():
        f = flag.astype(float)
        rows.append(
            {
                "check": name,
                "cancelled_pct": round(100 * f[c].mean(), 1),
                "not_cancelled_pct": round(100 * f[~c].mean(), 1),
                "verdict": verdict[name],
            }
        )
    amt_c = d.loc[c, "Amount"].dropna()
    amt_n = d.loc[~c, "Amount"].dropna()
    amt_n = amt_n[amt_n > 0]
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "leak_audit.csv", index=False)
    RESULTS["leak_audit"] = {
        "rows": rows,
        "median_amount_cancelled_when_present": round(float(amt_c.median()), 1),
        "median_amount_not_cancelled_when_present": round(float(amt_n.median()), 1),
    }

    # Unshipped check: when does it show up, and on which Status?
    uns = df["Courier Status"] == "Unshipped"
    RESULTS["unshipped"] = {
        "total": int(uns.sum()),
        "with_status_cancelled": int((uns & (df["Status"] == "Cancelled")).sum()),
        "with_status_pending": int(
            (uns & df["Status"].str.startswith("Pending")).sum()
        ),
        "with_other_status": int(
            (uns & ~df["Status"].str.startswith("Pending") & (df["Status"] != "Cancelled")).sum()
        ),
    }
    wk = df.groupby(df["Date"].dt.to_period("W-SUN")).agg(
        rows=("Qty", "size"),
        unshipped_pct=("Courier Status", lambda s: 100 * (s == "Unshipped").mean()),
        cancelled_pct=("Status", lambda s: 100 * (s == "Cancelled").mean()),
        open_rows=("Status", lambda s: int(s.isin(OPEN_STATUSES).sum())),
    )
    wk.index = [str(p.start_time.date()) for p in wk.index]
    wk.round(1).to_csv(OUT / "weekly_status.csv", index_label="week_start")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wk.index, wk["cancelled_pct"], marker="o", label="Status Cancelled")
    ax.plot(wk.index, wk["unshipped_pct"], marker="o", label="Courier Status Unshipped")
    ax.set_ylabel("% of order lines")
    ax.set_title("Unshipped jumps in the last (partial) week; cancellations do not")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "weekly_unshipped_vs_cancelled.png", dpi=150)
    plt.close(fig)
    return tab


# ---------------------------------------------------------------- features
CAT_FINAL = [
    "dow",
    "Fulfilment",
    "Sales Channel",
    "ship-service-level",
    "Category",
    "Size",
    "ship_state",
    "fulfilled_by",
    "B2B",
]
NUM_FINAL = ["te_sku", "te_style", "log_price", "sku_seen"]


def clean_state(s):
    return s.fillna("unknown").str.strip().str.lower()


def target_encode(train_keys, train_y, groups, apply_keys, m=20, n_splits=5):
    """Smoothed mean encoding. Out-of-fold on train, full-train map elsewhere."""
    prior = train_y.mean()

    def fit_map(k, y):
        g = pd.DataFrame({"k": k.values, "y": y.values}).groupby("k")["y"].agg(["sum", "count"])
        return (g["sum"] + m * prior) / (g["count"] + m)

    oof = pd.Series(prior, index=train_keys.index, dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(train_keys, groups=groups):
        mp = fit_map(train_keys.iloc[tr], train_y.iloc[tr])
        oof.iloc[te] = train_keys.iloc[te].map(mp).fillna(prior).values
    full = fit_map(train_keys, train_y)
    return oof, [k.map(full).fillna(prior) for k in apply_keys]


def build_final_features(df):
    d = df[~df["Status"].isin(OPEN_STATUSES)].copy()
    d["y"] = (d["Status"] == "Cancelled").astype(int)
    d["split"] = split_of(d["Date"])
    d["dow"] = d["Date"].dt.day_name()
    d["ship_state"] = clean_state(d["ship-state"])
    d["fulfilled_by"] = d["fulfilled-by"].fillna("none")
    d["B2B"] = d["B2B"].astype(str)

    tr = d["split"] == "train"
    # rare states grouped using training counts only
    counts = d.loc[tr, "ship_state"].value_counts()
    keep = set(counts[counts >= 100].index)
    d["ship_state"] = d["ship_state"].where(d["ship_state"].isin(keep), "other")

    # price proxy: median unit price of the SKU in training rows with a real Amount
    unit = (d["Amount"] / d["Qty"]).where((d["Qty"] > 0) & (d["Amount"] > 0))
    t = d[tr]
    sku_p = unit[tr].groupby(t["SKU"]).median()
    sty_p = unit[tr].groupby(t["Style"]).median()
    cat_p = unit[tr].groupby(t["Category"]).median()
    price = d["SKU"].map(sku_p)
    price = price.fillna(d["Style"].map(sty_p)).fillna(d["Category"].map(cat_p))
    price = price.fillna(unit[tr].median())
    d["log_price"] = np.log(price)
    d["sku_seen"] = d["SKU"].isin(set(t["SKU"])).astype(int)

    # target encoding of SKU and Style, training data only
    va, te = d["split"] == "val", d["split"] == "test"
    for col, name in [("SKU", "te_sku"), ("Style", "te_style")]:
        oof, (v, s) = target_encode(
            d.loc[tr, col], d.loc[tr, "y"], d.loc[tr, "Order ID"], [d.loc[va, col], d.loc[te, col]]
        )
        d[name] = np.nan
        d.loc[tr, name] = oof
        d.loc[va, name] = v.values
        d.loc[te, name] = s.values

    sizes = d.groupby("split").agg(
        rows=("y", "size"),
        positives=("y", "sum"),
        rate=("y", "mean"),
        first=("Date", "min"),
        last=("Date", "max"),
    )
    RESULTS["splits"] = {
        k: {
            "rows": int(r.rows),
            "positives": int(r.positives),
            "cancel_rate": round(float(r.rate), 4),
            "first_date": str(r["first"].date()),
            "last_date": str(r["last"].date()),
        }
        for k, r in sizes.iterrows()
    }
    return d


def make_preprocessor(cats, nums):
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cats),
            ("num", StandardScaler(), nums),
        ],
        sparse_threshold=1.0,
    )


# ---------------------------------------------------------------- metrics
def threshold_for_cost(y, p, cost_fn=COST_FN, cost_fp=COST_FP):
    # last candidate flags nothing, so "do not act" is always an option
    grid = np.append(np.unique(np.quantile(p, np.linspace(0.001, 0.999, 999))), np.inf)
    best = None
    for t in grid:
        pred = p >= t
        fn = int(((~pred) & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if best is None or cost < best[1]:
            best = (float(t), cost)
    return best[0]


def at_threshold(y, p, t):
    y = np.asarray(y)
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    return {
        "threshold": "flag none" if np.isinf(t) else round(float(t), 4),
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
        "flagged_pct": round(100 * pred.mean(), 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def scores(y, p):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "base_rate": round(float(np.mean(y)), 4),
    }


# ---------------------------------------------------------------- class replica
CLASS_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    eta=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=10,
    reg_lambda=10,
    seed=SEED,
)
CLASS_CATS = ["Fulfilment", "Sales Channel", "ship-service-level", "Category", "Size", "ship_state", "fulfilled_by", "B2B"]


def class_frame(df, target="class", amount="class_fill", qty=True, promo=True, drop_open=False):
    """Rebuild the class cleaning, with switches to remove one leak at a time."""
    d = df.copy()
    if drop_open:
        d = d[~d["Status"].isin(OPEN_STATUSES)].copy()
    d["y"] = class_target(d) if target == "class" else (d["Status"] == "Cancelled").astype(int)
    d = d[d["ship-state"].notna()].copy()  # class na.omit dropped these
    d["ship_state"] = clean_state(d["ship-state"])
    d["fulfilled_by"] = d["fulfilled-by"].fillna("none")
    d["B2B"] = d["B2B"].astype(str)
    d["split"] = split_of(d["Date"])
    nums = []
    if qty:
        d["qty_clean"] = d["Qty"].replace(0, 1)
        nums.append("qty_clean")
    if amount == "class_fill":
        d["amount_x"] = d["Amount"].where(d["Amount"].notna() & (d["Amount"] != 0), CLASS_AMOUNT_FILL)
        nums.append("amount_x")
    elif amount == "raw_nan":
        d["amount_x"] = d["Amount"]
        nums.append("amount_x")
    elif amount == "sku_price":
        tr = d["split"] == "train"
        unit = (d["Amount"] / d["Qty"]).where((d["Qty"] > 0) & (d["Amount"] > 0))
        p = d["SKU"].map(unit[tr].groupby(d.loc[tr, "SKU"]).median())
        p = p.fillna(d["Category"].map(unit[tr].groupby(d.loc[tr, "Category"]).median()))
        d["amount_x"] = p.fillna(unit[tr].median())
        nums.append("amount_x")
    if promo:
        d["promotion_flag"] = d["promotion-ids"].notna().astype(int)
        nums.append("promotion_flag")
    return d, nums


def class_matrix(train, others, nums):
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(train[CLASS_CATS].astype(str))

    def m(x):
        cat = enc.transform(x[CLASS_CATS].astype(str))
        num = sparse.csr_matrix(x[nums].to_numpy(dtype=float)) if nums else None
        return sparse.hstack([cat, num]).tocsr() if nums else cat.tocsr()

    return m(train), [m(o) for o in others]


def fit_class_xgb(Xtr, ytr, Xstop, ystop, weighted=True):
    params = dict(CLASS_PARAMS)
    if weighted:
        params["scale_pos_weight"] = float((ytr == 0).sum() / (ytr == 1).sum())
    dtr = xgb.DMatrix(Xtr, label=ytr, missing=np.nan)
    dst = xgb.DMatrix(Xstop, label=ystop, missing=np.nan)
    return xgb.train(
        params, dtr, num_boost_round=500, evals=[(dtr, "train"), (dst, "stop")],
        early_stopping_rounds=20, verbose_eval=False,
    )


def class_replica(df):
    """Class pipeline as run: random 70/30, early stopping on the test set, 0.5 threshold."""
    d, nums = class_frame(df)
    tr, te = train_test_split(d, test_size=0.3, random_state=SEED, stratify=d["y"])
    Xtr, (Xte,) = class_matrix(tr, [te], nums)
    bst = fit_class_xgb(Xtr, tr["y"].values, Xte, te["y"].values)
    p = bst.predict(xgb.DMatrix(Xte, missing=np.nan), iteration_range=(0, bst.best_iteration + 1))
    out = scores(te["y"], p) | at_threshold(te["y"].values, p, 0.5)
    out["accuracy"] = round(float(((p >= 0.5) == te["y"].values).mean()), 4)
    out["n_test"] = int(len(te))
    out["fill_value_rows"] = int((d["amount_x"] == CLASS_AMOUNT_FILL).sum())
    filled = d["amount_x"] == CLASS_AMOUNT_FILL
    out["fill_value_rows_cancel_rate"] = round(float(d.loc[filled, "y"].mean()), 4)
    out["fill_value_share_of_positives"] = round(float(d.loc[filled, "y"].sum() / d["y"].sum()), 4)
    RESULTS["class_replica"] = out
    log(f"class replica: {out}")
    return out


def leak_ladder(df):
    """Remove one leak at a time. Time split, early stopping on validation, scored on validation."""
    steps = [
        ("1. Class pipeline, time split instead of random", dict()),
        ("2. Amount left missing instead of cancel-mean fill", dict(amount="raw_nan")),
        ("3. Amount replaced by SKU median price", dict(amount="sku_price")),
        ("4. Qty dropped", dict(amount="sku_price", qty=False)),
        ("5. Promotion flag dropped", dict(amount="sku_price", qty=False, promo=False)),
        ("6. Target = Status Cancelled, open orders dropped", dict(amount="sku_price", qty=False, promo=False, target="status", drop_open=True)),
    ]
    rows = []
    for name, kw in steps:
        d, nums = class_frame(df, **kw)
        tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
        Xtr, (Xva,) = class_matrix(tr, [va], nums)
        bst = fit_class_xgb(Xtr, tr["y"].values, Xva, va["y"].values)
        p = bst.predict(xgb.DMatrix(Xva, missing=np.nan), iteration_range=(0, bst.best_iteration + 1))
        r = {"step": name} | scores(va["y"], p) | {"sensitivity_at_0.5": at_threshold(va["y"].values, p, 0.5)["sensitivity"]}
        rows.append(r)
        log(f"ladder {r}")
    RESULTS["leak_ladder"] = rows
    pd.DataFrame(rows).to_csv(OUT / "leak_ladder_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r["step"].split(". ", 1)[1] for r in rows]
    yy = np.arange(len(rows))[::-1]
    ax.barh(yy + 0.2, [r["roc_auc"] for r in rows], height=0.4, label="ROC-AUC")
    ax.barh(yy - 0.2, [r["pr_auc"] for r in rows], height=0.4, label="PR-AUC")
    ax.set_yticks(yy, labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation score (May 16 to Jun 5)")
    ax.set_title("Removing one leak at a time")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS / "leak_ladder.png", dpi=150)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- final models
def final_models(d):
    tr, va, te = (d[d["split"] == s] for s in ["train", "val", "test"])
    pre = make_preprocessor(CAT_FINAL, NUM_FINAL)
    Xtr = pre.fit_transform(tr).tocsr()
    Xva = pre.transform(va).tocsr()
    Xte = pre.transform(te).tocsr()
    names = list(pre.get_feature_names_out())
    ytr, yva, yte = tr["y"].values, va["y"].values, te["y"].values

    # logistic regression: pick C on validation PR-AUC
    lr_grid = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        m.fit(Xtr, ytr)
        lr_grid.append((C, average_precision_score(yva, m.predict_proba(Xva)[:, 1]), m))
    C_best, _, lr = max(lr_grid, key=lambda x: x[1])
    p_lr_va = lr.predict_proba(Xva)[:, 1]

    # xgboost: early stopping on validation log loss, grid scored on validation PR-AUC
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=names)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=names)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=names)
    xgb_grid = []
    for depth in [3, 5, 7]:
        for mcw in [1, 10]:
            params = dict(
                objective="binary:logistic", eval_metric="logloss", eta=0.05,
                max_depth=depth, min_child_weight=mcw, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=spw, seed=SEED,
            )
            b = xgb.train(params, dtr, 2000, evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=False)
            pv = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
            ap = average_precision_score(yva, pv)
            xgb_grid.append({"max_depth": depth, "min_child_weight": mcw, "best_iteration": int(b.best_iteration), "val_pr_auc": round(float(ap), 4), "_b": b})
    best = max(xgb_grid, key=lambda g: g["val_pr_auc"])
    bst = best["_b"]
    it = (0, bst.best_iteration + 1)
    p_xgb_va = bst.predict(dva, iteration_range=it)

    # thresholds from validation only
    t_lr = threshold_for_cost(yva, p_lr_va)
    t_xgb = threshold_for_cost(yva, p_xgb_va)
    # 3x is the locked assumption; 2x, 5x and 10x show how much the answer depends on it
    ratios = [2.0, 3.0, 5.0, 10.0]
    thr = {
        m: {k: threshold_for_cost(yva, pv, cost_fn=k) for k in ratios}
        for m, pv in [("logreg", p_lr_va), ("xgb", p_xgb_va)]
    }
    sens = {
        m: {f"{k:g}x": at_threshold(yva, pv, thr[m][k]) for k in ratios}
        for m, pv in [("logreg", p_lr_va), ("xgb", p_xgb_va)]
    }

    RESULTS["validation"] = {
        "logreg": {"C": C_best} | scores(yva, p_lr_va) | at_threshold(yva, p_lr_va, t_lr),
        "xgb": {k: v for k, v in best.items() if k != "_b"} | scores(yva, p_xgb_va) | at_threshold(yva, p_xgb_va, t_xgb),
        "xgb_grid": [{k: v for k, v in g.items() if k != "_b"} for g in xgb_grid],
        "by_cost_ratio": sens,
        "scale_pos_weight": round(spw, 3),
    }
    log(f"validation: {json.dumps(RESULTS['validation'], indent=1)}")

    if DEV:
        log("--dev: stopping before the test set")
        return tr

    # ---------------- test set: scored once, here
    p_lr_te = lr.predict_proba(Xte)[:, 1]
    p_xgb_te = bst.predict(dte, iteration_range=it)
    RESULTS["test"] = {
        "logreg": scores(yte, p_lr_te) | at_threshold(yte, p_lr_te, t_lr),
        "xgb": scores(yte, p_xgb_te) | at_threshold(yte, p_xgb_te, t_xgb),
        "n": int(len(yte)),
        "by_cost_ratio": {
            m: {f"{k:g}x": at_threshold(yte, pt, thr[m][k]) for k in ratios}
            for m, pt in [("logreg", p_lr_te), ("xgb", p_xgb_te)]
        },
    }
    log(f"TEST: {json.dumps(RESULTS['test'], indent=1)}")

    plot_curves(yte, {"Logistic regression": p_lr_te, "XGBoost": p_xgb_te})
    plot_calibration(yte, {"Logistic regression": p_lr_te, "XGBoost": p_xgb_te})
    shap_summary(bst, Xte, names, te)
    return tr


def plot_curves(y, preds):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, test set")
    axes[1].axhline(np.mean(y), color="k", ls="--", lw=0.8, label=f"Base rate {np.mean(y):.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, test set")
    for a in axes:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "roc_pr_test.png", dpi=150)
    plt.close(fig)


def plot_calibration(y, preds):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in preds.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    ax.set(xlabel="Mean predicted score", ylabel="Observed cancel rate", title="Calibration, test set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "calibration_test.png", dpi=150)
    plt.close(fig)


def shap_summary(bst, Xte, names, te):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(Xte.shape[0], size=min(5000, Xte.shape[0]), replace=False)
    X = pd.DataFrame(Xte[idx].toarray(), columns=names)
    sv = shap.TreeExplainer(bst).shap_values(X)

    # group one-hot columns back to their source feature
    def source(n):
        n = n.split("__", 1)[1]
        for c in CAT_FINAL + NUM_FINAL:
            if n == c or n.startswith(c + "_"):
                return c
        return n

    groups = pd.Series([source(n) for n in names])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=names).groupby(groups.values).sum().sort_values()
    RESULTS["shap_mean_abs_by_feature"] = {k: round(float(v), 4) for k, v in imp[::-1].items()}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values)
    ax.set(xlabel="Mean |SHAP| (log-odds), test sample of 5,000", title="XGBoost: what drives the score")
    fig.tight_layout()
    fig.savefig(CHARTS / "shap_by_feature.png", dpi=150)
    plt.close(fig)

    plt.figure()
    shap.summary_plot(sv, X, max_display=15, show=False)
    plt.title("Top one-hot levels, test sample")
    plt.tight_layout()
    plt.savefig(CHARTS / "shap_beeswarm.png", dpi=150)
    plt.close()


def business_read(tr):
    """Cancel rates for actionable fields, training period only."""
    out = {}
    for col in ["Fulfilment", "ship-service-level", "B2B", "Category", "fulfilled_by"]:
        g = tr.groupby(col)["y"].agg(["mean", "size"])
        g = g[g["size"] >= 200].sort_values("mean", ascending=False)
        out[col] = {str(k): {"cancel_rate": round(float(r["mean"]), 4), "rows": int(r["size"])} for k, r in g.iterrows()}
    ct = tr.groupby(["Fulfilment", "ship-service-level"])["y"].agg(["mean", "size"])
    out["Fulfilment x ship-service-level"] = {f"{a} / {b}": {"cancel_rate": round(float(r["mean"]), 4), "rows": int(r["size"])} for (a, b), r in ct.iterrows()}
    RESULTS["business_read_train"] = out


def comparison_table():
    rep, test = RESULTS["class_replica"], RESULTS["test"]["xgb"]
    ladder = {r["step"][0]: r for r in RESULTS["leak_ladder"]}
    rows = [
        ("ROC-AUC", CLASS_REPORTED["roc_auc_final"], rep["roc_auc"], test["roc_auc"],
         "Amount fill value, Qty and promotion leaks removed; time split"),
        ("PR-AUC", None, rep["pr_auc"], test["pr_auc"], "Not reported in class version"),
        ("Sensitivity", CLASS_REPORTED["sensitivity_final"], rep["sensitivity"], test["sensitivity"],
         "Class used a 0.5 cutoff; new uses a validation cutoff from the 3x cost assumption"),
        ("Precision", None, rep["precision"], test["precision"], "Not reported in class version"),
        ("Threshold", 0.5, 0.5, test["threshold"], "Chosen on validation; at 3x no cutoff beat flagging nothing"),
        ("Sensitivity at 5x cost cutoff", None, None, RESULTS["test"]["by_cost_ratio"]["xgb"]["5x"]["sensitivity"], "Secondary cost assumption, cutoff from validation"),
        ("Precision at 5x cost cutoff", None, None, RESULTS["test"]["by_cost_ratio"]["xgb"]["5x"]["precision"], "Secondary cost assumption, cutoff from validation"),
        ("Positive rate", 0.151, rep["base_rate"], test["base_rate"], "Status Cancelled only, open orders dropped"),
    ]
    tab = pd.DataFrame(rows, columns=["metric", "class_reported", "class_replica_rerun", "new_test", "why_different"])
    tab.to_csv(OUT / "old_vs_new.csv", index=False)
    with open(OUT / "old_vs_new.md", "w") as f:
        f.write("| Metric | Class report | Class pipeline rerun | New, test set | Why |\n|---|---|---|---|---|\n")
        for r in rows:
            fmt = lambda v: "n/a" if v is None else (v if isinstance(v, str) else f"{v:.3f}")
            f.write(f"| {r[0]} | {fmt(r[1])} | {fmt(r[2])} | {fmt(r[3])} | {r[4]} |\n")
    return tab


def main():
    args = [a for a in sys.argv[1:] if a != "--dev"]
    path = args[0] if args else str(ROOT / "data" / "Amazon Sale Report.csv")
    df = load(path)
    log(f"data: {RESULTS['data']}")
    leak_audit(df)
    class_replica(df)
    leak_ladder(df)
    d = build_final_features(df)
    log(f"splits: {RESULTS['splits']}")
    tr = final_models(d)
    business_read(tr)
    if DEV:
        return
    comparison_table()
    RESULTS["class_reported"] = CLASS_REPORTED
    RESULTS["versions"] = {
        "pandas": pd.__version__, "xgboost": xgb.__version__, "shap": shap.__version__,
        "numpy": np.__version__,
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
