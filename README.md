# Random Forest Network Intrusion Detection

**Course Assignment - Gradescope Submission**

---

## Overview

In this assignment you will train **two `RandomForestClassifier` models** on the UNSW-NB15 network traffic dataset:

1. **`binaryModel`**: classifies each connection as **attack (1)** or **normal (0)**, using the `label` column as the target.
2. **`multiclassModel`**: classifies each connection into one of **10 attack categories** (`Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`), using the `attack_cat` column as the target.

You will train these on a provided training set, and submit a single file, `randomForestModel.joblib`, which the autograder will re-evaluate against a held-out test set you do not have access to.

This is the direct sequel to the rule-based intrusion detector you built last time. Same dataset, same underlying problem, different tool. Keep your old metrics nearby, because the comparison is the point.

---

## What determines your ceiling here

A random forest will comfortably beat your hand-written rules on this dataset, but it does not escape the underlying problem you already ran into: attack and normal traffic genuinely overlap on the features you can measure. A forest handles that overlap differently than a rule set does: instead of one hard boundary per rule, it has hundreds of trees voting, so a connection sitting in genuinely ambiguous territory gets a probability rather than a single hard verdict. That reduces the damage overlap does, but it does not remove it. Two connections identical on every feature and different in ground truth are still identical on every feature. No amount of averaging over trees invents information that was never in the columns.

Where your ceiling *does* move is capacity: a forest can carve much finer-grained regions of feature space than four hand-written `if` statements, and it can do so on all 42 features simultaneously instead of the one or two you could juggle by hand. That is real signal recovery, not a trick. The part of the ceiling that comes from features that simply do not separate the classes will still be there. The part that came from *you* running out of hands to write more rules will mostly disappear.

The `max_depth` cap called for in this assignment is not there to protect the ceiling; it is a size and generalisation control. An unbounded tree can carve out a leaf for effectively every training row, including its noise, which is the tree-ensemble version of the "flag almost nothing with one extremely narrow rule" failure mode from the rule-based assignment: perfect performance on data it has already seen, and nothing left over that generalises to the held-out set.

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Chapter 15: Random Forests. Springer. [Free PDF](https://hastie.su.domains/ElemStatLearn/): the standard graduate-level treatment of why bagging reduces variance and where random forests still have irreducible error.

---

## The Dataset: UNSW-NB15

Each row is one network connection with 42 feature columns plus two label columns. This is the same dataset, and largely the same features, as the rule-based assignment: the difference is what you are permitted to do with them.

**Official dataset page:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

**Citation:**
> Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *2015 Military Communications and Information Systems Conference (MilCIS)*, IEEE. [DOI](https://doi.org/10.1109/MilCIS.2015.7348942)

- **Feature columns**: every column *except* `label` and `attack_cat`. Three of them are categorical strings: `proto`, `service`, `state`. The rest are numeric.
- **`label`**: the binary target (0 = normal, 1 = attack). Only used for training/evaluating `binaryModel`.
- **`attack_cat`**: the multi-class target (one of the 10 categories above). Only used for training/evaluating `multiclassModel`.

Never use `label` or `attack_cat` as input features: they are the answer keys, exactly as they were banned from your rule conditions in the previous assignment. The autograder checks for this class of leakage the same way it did there.

### Attack Categories in the Dataset

| Category | Description |
|---|---|
| Normal | Benign traffic, no attack behaviour present |
| Fuzzers | Sending random or semi-random data to find crashes or unexpected behaviour |
| Analysis | Port scans, spam, and HTML file infiltration |
| Backdoor | Secret bypass of normal authentication |
| DoS | Denial-of-service attacks aimed at exhausting target resources |
| Exploits | Attacks leveraging known software vulnerabilities |
| Generic | Attacks targeting block-cipher-based encryption |
| Reconnaissance | Information gathering, probing and scanning |
| Shellcode | Injecting and executing arbitrary shellcode |
| Worms | Self-replicating malware spreading across networks |

---

## What You Must Submit

A single file named **`randomForestModel.joblib`**, created with `joblib.dump(...)`, containing a Python `dict` with exactly these two keys:

```python
{
    'binaryModel': binaryPipeline,
    'multiclassModel': multiclassPipeline,
}
```

**Both models must satisfy all of the following:**

1. Each is an **`sklearn.pipeline.Pipeline`** whose **final step is a `RandomForestClassifier`**. A bare `RandomForestClassifier` (not wrapped in a `Pipeline`) will fail grading unless it can directly accept the raw feature columns; in practice, you need a `Pipeline`.
2. The pipeline's earlier steps must handle **encoding the categorical columns** (`proto`, `service`, `state`) internally, for example a `ColumnTransformer` with a `OneHotEncoder` for those three columns and `passthrough` for the rest. The autograder calls `.predict()` directly on a raw `pandas.DataFrame` of the 42 feature columns: it does **not** do any encoding for you.
3. `binaryModel.predict(X)` must return `0`/`1`. `multiclassModel.predict(X)` must return one of the 10 `attack_cat` strings.

### A note on model size

Leaving `max_depth` unbounded on this dataset (especially combined with one-hot encoding `proto`, which has 100+ distinct values) produces enormous trees: a `randomForestModel.joblib` file can easily balloon to several hundred MB, which is slow to upload and may exceed the submission size limit. Cap `max_depth` (e.g. 10–20) and keep `n_estimators` reasonable (e.g. 100–200). This barely affects accuracy on this dataset and keeps the file a manageable size (tens of MB).

---

## Rules and Constraints

These apply to your submitted `randomForestModel.joblib` and are enforced automatically. Violations result in a score of **0**.

| Required | Not allowed |
|---|---|
| Final pipeline step is `RandomForestClassifier` | Any other estimator as the final step (`GradientBoostingClassifier`, `LogisticRegression`, etc.) |
| Categorical encoding (`proto`, `service`, `state`) handled inside the pipeline | Relying on the autograder to pre-encode features (it will not) |
| `dict` with exactly the keys `binaryModel` and `multiclassModel` | Extra, missing, or misspelled keys |
| A single `.joblib` file produced by `joblib.dump(...)` | A `.py` file, a zip, or a folder |
| `binaryModel` trained on `label` only, `multiclassModel` trained on `attack_cat` only | Referencing `label` inside features used for `multiclassModel`, or `attack_cat` inside features used for `binaryModel` |
| Reasonable `max_depth` / `n_estimators` for file size | Leaving `max_depth` unbounded |

## How Grading Works

### Step 1: Submission checks (pass/fail)

Before anything is scored, the autograder runs two checks. **Failing either scores 0** with an explanation printed:

- **Artifact check**: `randomForestModel.joblib` must load and be a `dict` with `binaryModel` and `multiclassModel` keys, each exposing a callable `.predict()`.
- **Model type check**: the final estimator of each model (the last step of the `Pipeline`) must be a `RandomForestClassifier`. Submitting a different model type (e.g. `GradientBoostingClassifier`, `LogisticRegression`) will fail this check: this assignment specifically requires a random forest.

### Step 2: Metrics

If both checks pass, your models are run against the held-out test set and scored on **8 metrics**:

| Task | Metrics |
|---|---|
| Binary (`label`) | Accuracy, Precision, Recall, F1 |
| Multi-class (`attack_cat`) | Accuracy, macro-averaged Precision, Recall, F1 |

There is **no baseline** for this assignment: your raw score on each metric directly determines how many points you earn for it:

```
pointsEarned(metric) = (yourMetricValue / 100) × maxPointsForMetric
```

| Metric | Max Points |
|---|---|
| Binary Accuracy | 15 |
| Binary Precision | 10 |
| Binary Recall | 10 |
| Binary F1 | 15 |
| Multi-class Accuracy | 15 |
| Multi-class Precision (macro) | 10 |
| Multi-class Recall (macro) | 10 |
| Multi-class F1 (macro) | 15 |
| **Base Grade Total** | **100** |

A perfect score on every metric (100% accuracy/precision/recall/F1 on both tasks) earns the full 100 points. There's no minimum threshold to "pass" a given metric: every percentage point you earn on each metric counts proportionally.

Unlike the rule-based assignment, there is no regression penalty here: you are scored on raw metric values, not on improvement over a baseline. There is also no free lunch equivalent to "flag everything" or "flag nothing": a degenerate model (predicting the majority class for everything, say) will simply score low on precision/recall/F1 for the classes it never predicts, the same way it would in any standard classification setting.

### Step 3: Early Submission Bonus

Submitting ahead of the deadline earns **extra credit**, on top of your base grade (uncapped by the 100-point ceiling; a perfect submission 6 days early can score up to 103):

| Days Early | Bonus |
|---|---|
| 6+ | +3.0 |
| 5 | +2.5 |
| 4 | +2.0 |
| 3 | +1.5 |
| 2 | +1.0 |
| 1 | +0.5 |
| 0 (on the due date or late) | +0.0 |

---

## Common Mistakes That Score 0

- Submitting a `.py` file, a zip, or a folder instead of a single `.joblib` file.
- Submitting a bare `RandomForestClassifier` (or any model) that can't `.predict()` directly on the raw feature `DataFrame`: categorical columns will crash it.
- Submitting a `dict` with the wrong key names, or missing one of `binaryModel`/`multiclassModel`.
- Using any classifier other than `RandomForestClassifier` as the pipeline's final step.
- A runtime error during prediction (e.g. a feature-mismatch, a stale encoder fit on different categories): this doesn't zero your score outright, but any metric your models fail to produce will count as 0 for that metric.

---

## Testing Locally

```bash
python trainRandomForest.py
```

Hold out a local validation split (e.g. `train_test_split`, stratified on `attack_cat`) and print all 8 metrics before you submit. Your local numbers are only an estimate: the autograder evaluates against a different, held-out set drawn from the same source, so rules and splits that overfit your local file will not necessarily transfer. Test as often as you like; only your Gradescope submission counts for marks.

**Explore before you train.** You are allowed to inspect the training CSV however you want: pandas, a notebook, plotting feature distributions by `label` or `attack_cat`, checking class balance, whatever helps you pick sane hyperparameters. The constraints in the "Rules and Constraints" table apply to what you submit, not to how you get there.

---

## Suggested Workflow

1. Load the provided training CSV with `pandas`.
2. Build a `ColumnTransformer` (`OneHotEncoder(handle_unknown='ignore')` on `proto`/`service`/`state`, `passthrough` for the rest) feeding into a `RandomForestClassifier`, wrapped in a `Pipeline`.
3. Fit one pipeline on `label`, a second on `attack_cat`.
4. Hold out a local validation split (e.g. `train_test_split`, stratified on `attack_cat`) to sanity-check all 8 metrics before submitting; your local numbers are only an estimate, since the autograder evaluates against a different, held-out set.
5. `joblib.dump({'binaryModel': ..., 'multiclassModel': ...}, 'randomForestModel.joblib')` and upload that file to Gradescope.

---

## Submission

Upload to the Gradescope assignment:

1. **`randomForestModel.joblib`**, produced by `joblib.dump(...)`. Upload the file itself, not a zip, not a notebook, and not the training script.

Autograder results are returned within Gradescope. You may resubmit as many times as the assignment allows (read the output carefully to find this); each resubmission before the deadline also banks you a fresh early-submission bonus tier, so submitting a working baseline early and iterating is strictly better than saving one attempt for the last minute.
