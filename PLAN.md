# Plan: order cancellation model, solo redo

Status: LOCKED on 2026-11-17, before any model was fit. Target, split, promotion exclusion, and cost ratio confirmed by Kevin.

## Background
This redoes an XGBoost model from a 4-person class project (ITEC-621, spring 2026). The model was mine; the data cleaning was shared by the group. The class version had several leaks and a random split. This version uses only information known when an order is placed, a time split, and a test set touched once.

## Data
- `Amazon Sale Report.csv` from the Kaggle dataset "E-Commerce Sales Dataset" by The Devastator (https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data), which credits data.world user anilsharma87 as the original source.
- Local copy matches the Kaggle file size (68,923,428 bytes).
- 128,975 rows, 24 columns (the 23 expected plus an empty `Unnamed: 22`). One row per order line, Mar 31 to Jun 29, 2022. 120,378 distinct Order IDs. No order spans more than one date.
- License on Kaggle reads "Other (specified in description)", but the description names no license, only a request to credit the source. The CSV will not be committed. The README links to Kaggle.

## Target
`y = 1` if `Status == "Cancelled"`, else 0. 18,332 positives.

Courier Status "Unshipped" is not counted as a cancellation:
- 5,631 Unshipped rows already have Status Cancelled, so they are counted anyway.
- The other 1,050 Unshipped rows are mostly Status Pending (927). Unshipped is 4 to 5% of rows in most weeks and 33% in the last partial week (Jun 27 to 29). That pattern fits "not shipped yet when the data was pulled", not "cancelled".

Drop rows whose outcome was still open at pull time (Status Pending, Pending - Waiting for Pick Up, Shipping): 947 rows, 887 of them in the last week. Labeling them 0 would assume they all shipped.

Resulting data: 128,028 rows, about 14.3% positive.

## Split
Time split by order date, whole days only, so every line of an order lands in one set.

| Set | Dates (2022) | Rows (after dropping open orders) | Cancel rate |
|---|---|---|---|
| Train | Mar 31 to May 15 | 71,499 | 14.5% |
| Validation | May 16 to Jun 5 | 26,955 | 13.5% |
| Test | Jun 6 to Jun 29 | 29,574 | 14.6% |

Note: the last days may still be slightly undercounted for cancellations if some open orders later cancelled. Dropping open orders reduces this but cannot fix it.

## Features (all known at order time)
| Feature | Why known at order time | Treatment |
|---|---|---|
| Day of week (from Date) | Order timestamp | Categorical |
| Fulfilment (Amazon / Merchant) | Chosen when listing is set up | Categorical |
| Sales Channel | Where the order was placed | Categorical (almost all Amazon.in) |
| ship-service-level | Chosen by buyer at checkout | Categorical |
| Category, Size | Product attributes | Categorical |
| Style, SKU | Product attributes | Target-encoded with out-of-fold means on train only |
| ship-state | Buyer address at checkout | Categorical, rare states grouped |
| B2B | Buyer account type | Binary |
| fulfilled-by | Shipping program for merchant orders | Categorical, missing as its own level |
| Price | See leak audit | SKU median Amount from training rows, not the row's own Amount |

## Not features
- Status, Courier Status: recorded after the outcome.
- Qty: 69% of cancelled lines have Qty 0 vs 0.1% of others. Qty looks reset on cancellation. Treated as a sign of the outcome.
- Amount and currency missingness: Amount is missing in 41% of cancelled lines vs 0.2% of others, and currency is missing on the same rows. The row's own Amount is not used. A price proxy (SKU median Amount on training data) is used instead.
- **promotion-ids:** missing in 98.4% of cancelled lines vs 28.1% of others. For Merchant orders it is missing in 100% of cancelled lines vs 0.2% of others, even when Amount is present. This looks like promotions are written at invoicing or shipping, not at order time. Excluded as a leak. The class model named promotion_flag its top predictor, which this would explain.
- Order ID, index, ASIN (duplicates SKU), ship-city and ship-postal-code (very high cardinality), ship-country (constant).

## Models
1. Logistic regression baseline (one-hot categoricals, target-encoded SKU/Style, standardized price).
2. XGBoost with `scale_pos_weight` from training class balance. Early stopping on validation log loss. A small hyperparameter grid scored on validation PR-AUC.

## Metrics
- ROC-AUC and PR-AUC (base rate about 14%, so PR-AUC is the main one).
- Sensitivity and precision at the chosen threshold.
- Calibration plot on test.
- SHAP summary for XGBoost.

## Threshold and cost assumption
Assumption: a missed cancellation costs 3 times a false alarm. Threshold picked on validation to minimize `3 * FN + 1 * FP`. This is an assumption, not a measured cost. The README also reports results at 2x and 5x on validation to show sensitivity to it.

## Test set rule
The test set is scored once, after the model, features, and threshold are fixed on validation. No changes after that run. If a bug forces a rerun, the README says so.

## Leak checks to measure
- Rebuild the class pipeline as closely as possible (cancel-mean Amount fill, Qty 0 to 1, promotion flag, random 70/30) and score it, then remove one leak at a time on the time split, to show how much each one inflated AUC and sensitivity.
