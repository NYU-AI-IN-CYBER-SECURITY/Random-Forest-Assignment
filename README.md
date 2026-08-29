# Random Forest Network Intrusion Detection

**Course Assignment - Gradescope Submission**

---

## Overview

In this assignment you will train **two `RandomForestClassifier` models** on the UNSW-NB15 network traffic dataset:

1. **`binaryModel`**: classifies each connection as **attack (1)** or **normal (0)**, using the `label` column as the target.
2. **`multiclassModel`**: classifies each connection into one of **10 categories** (`Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`), using the `attack_cat` column as the target.

You train on the same CSV as last time and submit a single file, `randomForestModel.joblib`, which the autograder re-evaluates against a held-out test set you do not get to see.

This is the direct sequel to the rule-based intrusion detector. Same dataset, same problem, different tool. **Keep your old metrics nearby, because the comparison is the point.**

### What this assignment is actually about

Read this part carefully, because it changes how you should spend your time.

This is **not** a machine learning course, and this is not an assignment about producing a hyper-optimised model. Your work is measured, and you will get a real score back. But the goal is not to squeeze out the last two points of F1, and if you find yourself building an eight-hour grid search, you have misread the assignment.

Note also that submissions are made through the course portal and are **limited in number**. The portal shows you how many attempts you have and how many you have used, so check it before you plan your week. You have enough to work with, but the autograder is not your development loop. Do your iterating locally, where attempts are free, and spend your submissions on configurations you already have reason to believe in. See "Testing Locally" for how to build a validation setup worth trusting.

What you are being asked to understand is **where a model like this one fits in**. You are working through a progression: hand-written rules last week, a learned model this week, and AI systems later in the course. Each layer is a different tool for the same underlying job, but perhaps different aspects of it (network detection), and each has a shape of problem it suits and a shape it does not. So as you work, keep three questions open:

- What does the model do well that a rule set cannot, and why.
- What does it do no better than a rule set, and why that limit is not a tuning problem.
- What will it get wrong quietly, in a way that still looks like success on the report you hand your manager.

That last one is the heart of it. A security engineer who cannot tell a good model from a good-looking model is a liability, and by the end of this assignment you will have produced a good-looking bad model yourself, on purpose, and then fixed it.

That said, **you are expected to tune**. A model left entirely at its defaults will show you nothing interesting and will not score well. The point is deliberate, informed adjustment where you can say what each change did and why, not exhaustive search. Parameters worth experimenting with: `max_depth`, `n_estimators`, `max_features`, `max_leaf_nodes`, and `min_samples_leaf`. Change one, re-measure, form an opinion. That is the loop. We also impose a size limit, so you will have to make some choices!

**A high score is very achievable.** You are scored on your raw metric values, similar to the previous assignment. Benchmark thoroughly on your own machine, then spend a submission when you have something worth testing.

**TIP: Read everything before getting started!**

---

## Before You Start

**Install from the `requirements.txt` in this repository**, not the latest of everything. The autograder runs pinned versions, and a scikit-learn version mismatch is one of the few failures you cannot diagnose from your own machine. More on this under "Environment versions".

**One naming quirk, to save you a confusing five minutes:** the package installs as `scikit-learn` and imports as `sklearn`.

**If you have never used scikit-learn**, budget an hour before you write anything. Read [Getting Started](https://scikit-learn.org/stable/getting_started.html) and then [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html). The full set of resources, including video, is in "Tools, and where to learn them" below.

**There is no starter code, and that is deliberate.** The single most common way to score 0 on this assignment is to do your encoding outside the model you submit. A skeleton with the pipeline already wired would make that mistake impossible, and it would also delete the thing you are here to learn: that a trained model is the fitted trees *plus* the transformations that produced its inputs. Assembling that yourself is the exercise. The structure you are aiming for is described plainly in "Your Task", and the linked worked example builds the same shape on a different dataset.

---

## Why This Is a Very Real Exercise

You have not switched paradigms as much as it looks.

A decision tree is a rule-based system. It is a nested chain of threshold comparisons on individual features, structurally identical to the `if` statements you wrote last time. The difference is that the thresholds are fit from data rather than chosen by you, and there are thousands of them rather than a handful. Print a fitted tree and you will see conditions like `sttl <= 63.5` and `ct_state_ttl <= 1.5`. Those are your rules. Nobody typed them.

Random forests are also not a teaching toy. They were the workhorse of applied security ML for well over a decade and remain in production in malware family classification, phishing detection, spam filtering, and network flow triage. They train fast, tolerate mixed data types, and need relatively little tuning to be useful.

For those curious about where these concepts come from, beyond what a dedicated Machine Learning course would cover:

- Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81-106. [DOI](https://doi.org/10.1007/BF00116251)
  How a tree chooses its splits. The formal version of what you were doing by eye last week.
- Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123-140. [DOI](https://doi.org/10.1007/BF00058655)
  Single trees are unstable. Train many on resampled data, average them, and the variance drops.
- Ho, T. K. (1998). The Random Subspace Method for Constructing Decision Forests. *IEEE TPAMI*, 20(8), 832-844. [DOI](https://doi.org/10.1109/34.709601)
  Give each tree only a random subset of features so the trees disagree in useful ways.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32. [DOI](https://doi.org/10.1023/A:1010933404324) | [Free PDF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  Both ideas combined. `RandomForestClassifier` is close to a direct transcription of this paper, and it reads well.
- Hastie, Tibshirani, & Friedman (2009). *The Elements of Statistical Learning*, Ch. 15. [Free PDF](https://hastie.su.domains/ElemStatLearn/)

And two on reading security ML claims specifically, if that side interests you:

- Sommer, R., & Paxson, V. (2010). Outside the Closed World: On Using Machine Learning for Network Intrusion Detection. *IEEE S&P*. [Free PDF](https://www.icir.org/robin/papers/oakland10-ml.pdf)
  Cited last week too. It lands differently now that you are the one training the model.
- Arp, D., et al. (2022). Dos and Don'ts of Machine Learning in Computer Security. *USENIX Security '22*. [Free PDF](https://www.usenix.org/system/files/sec22-arp.pdf)
  A catalogue of how security ML papers fool themselves. This assignment walks you past several of the traps deliberately.

---

## What Determines Your Ceiling Here

### The part that lifts

Think about what your rule set could physically express. If you finished the last assignment with, say, four or five `if` statements, each testing one or two features, then your detector drew four or five hard boundaries through a 42-dimensional space. That is not a criticism of your rules. It is a limit on how much any human can hold in their head at once, and it is probably why you stopped adding rules when you did (assuming a human created the rules).

A forest draws tens of thousands of boundaries, across all 42 features simultaneously, and classifies each connection by a vote across all of its trees rather than by whichever rule happened to fire first. The part of last week's ceiling that came from **you running out of hands** mostly disappears. Expect your binary metrics to jump noticeably.

Voting also changes the shape of the errors, which matters more than it sounds. A rule set returns a hard verdict. A forest returns a vote share, so a genuinely ambiguous connection comes out at 0.55 (think of that in this situation as 'barely sure') rather than a confident (but perhaps missleading) `True`. Call `predict_proba` once on a few rows (always one at a time) and look at what comes back. In production, that number is what lets an analyst set an alerting threshold instead of accepting whatever the rules decided.

### The part that does not lift

Take the four connections from the rule-based assignment:

| Connection | What it actually was | Label |
|---|---|---|
| 2 packets out, 0 back | Port scan probing a closed port | Attack |
| 2 packets out, 0 back | A laptop retrying a DNS server that was down | Normal |
| 1 packet out, 0 back | Shellcode delivery the host dropped | Attack |
| 1 packet out, 0 back | Monitoring agent health check against an offline service | Normal |

Every tree that reaches these rows lands in a leaf containing both classes, because on the features available the rows are identical. The forest returns roughly 0.5 and gets two of the four wrong, exactly as your rule did. It fails more gracefully and more informatively, but it fails.

Important point! Averaging over trees cannot invent information that was never in the columns. If you want that ceiling to move you need better features, which means going back to the packet capture, not turning `n_estimators` up. `n_estimators` defines (in this case) the number of base models, such as decision trees, created within our ensemble method (random forest). A high value means a really WIDE ensemble! Higher values stabilize predictions and reduce variance, but yield diminishing returns past a certain threshold. It also increases the space your model takes up!

Note that `max_depth` is not what protects this ceiling. It is a size and generalisation control. An unbounded tree keeps splitting until every leaf is pure, which on 30,000 rows means carving out a leaf for effectively every training row including its noise. That is the ensemble version of last week's "one extremely narrow rule" failure: perfect on data it has already seen, less (perfect) to data it has not. Or potentially, really poorly!

### The new ceiling: rare classes

**Always look at your training data before you train on it.** You cannot reason about a model's failures if you do not know the shape of the data it learned from, and every hour students lose on this assignment is lost by tuning hyperparameters blindly instead of spending time looking first. So start here:

```python
df['attack_cat'].value_counts()
```

What you should be asking as you read the output: are the ten categories evenly represented, or does a handful of them dominate? How many examples does the *smallest* category actually have? Is that enough for anything to learn from? Maybe a balanced set isn't always the best option?

You will find `Normal` and `Generic` plentiful, and `Analysis`, `Backdoor`, `Shellcode` and especially `Worms` scarce. A forest trained on that distribution does something entirely rational and entirely useless: it learns that predicting `Worms` is almost never worth the risk, because every `Worms` row sits in a leaf outvoted by `Exploits` or `Generic` neighbours. The class is simply never predicted at all.

Now connect that to how you are scored. Your multi-class metrics are **macro-averaged**, meaning the metric is computed separately for each of the ten classes and then averaged without weighting. A class you never predict scores 0, and it pulls the average down by a full tenth regardless of how rare it was. Perfect scores on six classes and silence on four gives you:

```
macro F1 = (1.0 x 6 + 0.0 x 4) / 10 = 0.60
```

Roughly 14 points gone, while your multi-class **accuracy** still reads in the high 90s, because those four categories are almost no rows.

This is the point of the exercise, not a scoring quirk invented to trouble you. Rarity and importance are unrelated: `Worms` is rare on the wire, and it is also self-replicating malware spreading through your network. Macro-averaging is the metric choice that makes structural blindness visible instead of letting a large `Normal` class hide it. Whenever you read "99% accurate" in a security product datasheet, this is the first question to ask!!

Closing the gap is yours to solve, but three places to look: `class_weight` on the classifier, `min_samples_leaf` (its default of 1 interacts badly with a capped `max_depth` here), and resampling your training data. Whatever you try, print a `classification_report` on a validation split and read the **per-class rows**, not the summary line. It tells you precisely which classes are dead and how dead, which is a much faster loop than watching one aggregate number wobble.

---

## The Dataset: UNSW-NB15

Created by the Australian Centre for Cyber Security at UNSW, mixing real packet captures with synthetic attack traffic from the IXIA PerfectStorm tool.

**Official dataset page:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

**Citation:**
> Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *2015 MilCIS*, IEEE. [DOI](https://doi.org/10.1109/MilCIS.2015.7348942)

### Your Training File

**`UNSW_NB15_balanced_30k.csv`**, the same 30,000-record subset you used for the rule-based assignment:

| Split | Count |
|-------|-------|
| Normal (`label = 0`) | 15,000 |
| Attack (`label = 1`) | 15,000 |
| **Total** | **30,000** |

This file is balanced on `label` and badly imbalanced on `attack_cat`. **That is deliberate.** It is a realistic sample of network traffic and a poor multi-class training set at the same time, and noticing that, then deciding what to do about it, is part of the assignment rather than a defect to work around. Run `value_counts()` and see how bad it is for yourself before you decide anything. Nothing stops you from building a better training set out of the full public dataset, and you are encouraged to.

> **On the test set:** the autograder evaluates against a separate, unseen set that is held back from what is publicly downloadable. Building your own evaluation sets from the full UNSW-NB15 will not overlap with it. It may not be balanced on either column, and you do not get to look at it. All we promise is that it is the same *type* of network activity. That is the real-world condition and it is worth sitting with: you prepare with what you have, for what you expect, with no guarantee the two match. A model tuned to the exact composition of your local file is a model betting that they do match, but we all know that this is almost never the case!

### Columns

Every column except `label`, `attack_cat`, and `id` is a feature.

`id` is the record number from the original UNSW-NB15 dataset. It is carried in this CSV purely so you can trace these 30,000 rows back to where they were pulled from. It says nothing about the traffic, it is not a feature, and it must not be fed to your model. The autograder passes your pipeline the 42 features only, so keep your training inputs shaped the same way.

Three of the 42 features hold text rather than numbers:

| Column | Notes |
|---|---|
| `proto` | Transport protocol. 100+ distinct values, most of them very rare |
| `service` | Application service (`http`, `dns`, `ftp`, `-` for none) |
| `state` | Connection state (`FIN`, `CON`, `INT`, `RST`, etc.) |

These columns were always in the CSV. Last week you could ignore them, because a hand-written comparison like `row['sttl'] > 200` does not care what type sits in the columns next to it. This week you cannot ignore them, for good reason!

A decision tree splits by asking "is this value less than that threshold?" That question is meaningless for the string `"tcp"`. There is no numeric ordering of protocol names, so before any tree can use these columns, each text value has to be turned into numbers. The standard approach is **one-hot encoding**: replace one `proto` column with one new column per distinct protocol, each holding 0 or 1. The tree can then ask "is the `proto_tcp` column equal to 1?", which is a threshold question it can answer.

You do not have to implement that yourself. You do have to make sure it happens, and, as the next section explains, that it happens *inside the model you submit*.

The remaining 39 features are the numeric ones from the rule-based assignment (`dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `dwin`, `smean`, `dmean`, `trans_depth`, `response_body_len`, the ten `ct_*` counters, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `is_sm_ips_ports`).

**Never use `label` or `attack_cat` as an input feature.** They are the answer keys, banned exactly as they were from your rule conditions, and the autograder checks for this the same way it did there. That includes the cross-case: no `label` in the multi-class features, no `attack_cat` in the binary features. Build your feature set programmatically (everything except those three columns, `id` included) rather than typing out a list, so it cannot drift out of sync with what the autograder passes you.

### Attack Categories

| Category | Description |
|---|---|
| Normal | Benign traffic, no attack behaviour present |
| Fuzzers | Random or semi-random data sent to find crashes |
| Analysis | Port scans, spam, HTML file infiltration |
| Backdoor | Secret bypass of normal authentication |
| DoS | Denial of service, exhausting target resources |
| Exploits | Leveraging known software vulnerabilities |
| Generic | Attacks targeting block-cipher-based encryption |
| Reconnaissance | Information gathering, probing, scanning |
| Shellcode | Injecting and executing arbitrary shellcode |
| Worms | Self-replicating malware spreading across networks |

These overlap more than the descriptions suggest. `Exploits`, `Backdoor` and `Shellcode` are frequently three stages of the same intrusion and produce similar flows. `Analysis` and `Reconnaissance` are both scanning. When you look at your confusion matrix you will find the model making the same confusions a human analyst would make, which is a good sign rather than a bad one.

---

## Your Task

You write your own training code, in whichever form you prefer: a **Jupyter notebook** or a plain **Python script**. There is no starter file this week, for the reason given under "Before You Start".

That file is a **required deliverable**: you will submit it to Gradescope alongside your token, named `<netid>_rf.ipynb` or `<netid>_rf.py`. It is **not graded**, since your score comes entirely from the `.joblib` artifact the portal evaluates, but a Gradescope submission without a correctly named training file is rejected. See [Submission](#submission) for the naming rule and upload steps.

### The requirement that trips everyone up

The autograder calls `.predict()` on a raw `pandas.DataFrame` containing the 42 feature columns, text values and all. **It does no encoding for you.**

That means the encoding has to travel *inside* the object you submit. If you one-hot encode in your training script and then save only the classifier, the autograder will hand your model the string `"tcp"` where it expects a number, and it will crash. This is the single most common way to score 0 on this assignment.

Understand why rather than just complying, because the principle generalises well beyond this course. A trained model is not only the fitted trees. It is the trees **plus every transformation applied to the data before fitting**, and those transformations carry state: which categories existed, in what order, mapped to which column positions. A model shipped without that state is not reproducible by anyone but you, and it will silently produce garbage the moment someone encodes their data in a slightly different order. Bundling the preprocessing and the classifier into one object makes the artifact self-contained. That is standard production practice, and it is what is being enforced here.

**The structure you need**, stated plainly: a single fitted object that holds both the encoder and the classifier, so that calling `.predict()` on raw text-and-numbers data runs the encoding and the forest in one step. In scikit-learn, the pieces that do this are `OneHotEncoder` (turns the three text columns into numbers), `ColumnTransformer` (applies that encoding to those three columns and passes the 39 numeric ones through untouched), and `Pipeline` (chains the transformer and the `RandomForestClassifier` into one saveable object). Build one such pipeline for `label` and a second for `attack_cat`. How you wire them together is yours to work out, and the [mixed types worked example](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) linked below builds this exact shape on a different dataset.

### Tools, and where to learn them

**scikit-learn** (usually written `sklearn`) is one of the standard open-source machine learning libraries for Python. It provides the classifier, the preprocessing steps, the pipeline container, and the metric functions this assignment refers to. If you have not taken a prior machine learning course, you may never have used it, you will need to take some time to get familiar. Everything you need is in the first two links below.

**Start here:**

- [scikit-learn: Getting Started](https://scikit-learn.org/stable/getting_started.html). Fifteen minutes. Covers fitting an estimator and introduces pipelines.
- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html). The user guide for `Pipeline` and `ColumnTransformer`. This is the one to read properly.
- [Column Transformer with Mixed Types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html). A worked example of exactly the structure this assignment needs, on a different dataset.

**Reference pages for the pieces:**

- [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). Pay attention to `handle_unknown`.
- [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and the [forest user guide](https://scikit-learn.org/stable/modules/ensemble.html#forest). Read the parameters listed in the Overview: `max_depth`, `n_estimators`, `max_features`, `max_leaf_nodes`, `min_samples_leaf`.
- [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Note the `stratify` argument, which matters a great deal here.
- [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) and [`classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html). Understand what `average='macro'` does before you tune anything.
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html) and [joblib](https://joblib.readthedocs.io/en/stable/persistence.html).
- New to pandas? [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html).
- Depending on your dataset, the SMOTE imbalance-learn library may be useful (as well as others), uses scikit-learn! [Imbalanced Learn](https://imbalanced-learn.org/stable/index.html)

**If you prefer video.** Watch enough to understand how a forest works and how to drive one. Do not disappear into a machine learning curriculum; that is not what this assignment is testing, don't overwhelm yourself!

- [What is Random Forest?](https://www.youtube.com/watch?v=gkXX4h3qYm4) (IBM). Short, conceptual, a good first watch.
- [Random Forest Algorithm Explained with Python and scikit-learn](https://www.youtube.com/watch?v=_QuGM_FW9eo). Practical, code-level, closer to what you are doing here.
- [Random forest walkthrough](https://www.youtube.com/watch?v=LrCylIe0RJM). Another applied treatment.
- [StatQuest](https://www.youtube.com/@statquest) has really clear visual explanations of decision trees and random forests, if you want the theory behind the mechanics.

**Other libraries.** These resources talk about scikit-learn, but you are welcome to use other tools in your own workflow for exploration, plotting, notebooks, or learning the concepts. The one hard constraint is the artifact itself: because the autograder loads and calls your saved object directly, the submitted pipeline must be a scikit-learn `Pipeline` ending in a `RandomForestClassifier`. How you get there is up to you. Does make sure to upload what we ask for!

### Environment versions

The autograder runs the versions of Python, scikit-learn, pandas, and everything else pinned in the **`requirements.txt` in this repository**. Install from that file rather than pulling the latest of everything.

Models saved by one version of scikit-learn are not guaranteed to load correctly in another. A version mismatch can produce a warning, a hard failure to load, or, worst of all, an artifact that loads and predicts incorrectly. If your submission fails a check that your local verification passed, a version mismatch is the first thing to check, again look at `requirements.txt`.

### Saving your submission

Once both pipelines are fitted:

```python
import joblib

joblib.dump({
    'binaryModel': binaryPipeline,
    'multiclassModel': multiclassPipeline,
}, 'randomForestModel.joblib')
```

Exactly those two keys, both camelCase, nothing else in the dictionary. Don't be fancy!

### Verify before you upload

Run this in a **fresh** interpreter, not the one you trained in. It'll help you catch score-zero failures:

```python
import joblib, pandas as pd

models = joblib.load('randomForestModel.joblib')
print(models.keys())
print(type(models['binaryModel'].steps[-1][1]))
print(type(models['multiclassModel'].steps[-1][1]))

df = pd.read_csv('UNSW_NB15_balanced_30k.csv')
raw = df.drop(columns=['id', 'label', 'attack_cat']).head(5)   # raw, unencoded, 42 features

print(models['binaryModel'].predict(raw))       # expect 0s and 1s
print(models['multiclassModel'].predict(raw))   # expect category strings
```

If `predict(raw)` raises an exception, your encoding is not inside your pipeline and you would have scored 0. Fixing it now costs you nothing...

### Refit on everything before you submit

Your validation split exists so you can measure honestly. Once you have settled on hyperparameters, refit both pipelines on your data. Those held-back rows are free signal, and they matter most for the rare classes, where you may only have a few dozen examples to begin with.

---

## Rules and Constraints

Enforced automatically. Violations score **0**.

| Required | Not allowed |
|---|---|
| Final pipeline step is `RandomForestClassifier` | Any other final estimator (`GradientBoostingClassifier`, `LogisticRegression`, `XGBClassifier`) |
| Categorical encoding handled inside the pipeline | Relying on the autograder to pre-encode (it will not) |
| `dict` with exactly `binaryModel` and `multiclassModel` | Extra, missing, or misspelled keys |
| A single `.joblib` from `joblib.dump(...)` uploaded to the **portal** | A `.py` file, a notebook, a zip, or a folder in place of the `.joblib` |
| `binaryModel` on `label`, `multiclassModel` on `attack_cat` | `label` in the multi-class features, or `attack_cat` in the binary features |
| Final file under 30 MB | Leaving `max_depth` unbounded |
| `token.txt` **and** `<netid>_rf.ipynb` *or* `<netid>_rf.py` uploaded to **Gradescope** | A missing, misnamed, or empty training file |

Everything about *how you get there* is otherwise unconstrained. Explore, plot, run a search if you want to. The constraints apply to what you submit, not to how you arrive at it, with the one exception that the work has to live in a notebook or script you hand in.

### Model size

Your finished `randomForestModel.joblib` is expected to come in **under 30 MB**, and the autograder checks this.

The reason a file can blow past that is worth understanding. One-hot encoding `proto` turns one column into 100+, and an unbounded tree keeps splitting until every leaf is pure, so each of your trees can end up with tens of thousands of nodes. Multiply by `n_estimators` and by two models in one file, and several hundred MB is easy to reach by accident.

How you get under the limit is your decision, which part of the exercise. `max_depth` is the most direct lever, but `n_estimators`, `max_leaf_nodes`, `min_samples_leaf`, and `max_features` all change the size of the resulting artifact, and they change your metrics differently from one another. There are others to look at too! Experiment, watch both the file size and the eight metrics, and form a view about the trade-off. On this dataset, sensible caps cost very little accuracy. You won't get brownie points for getting a smaller file size, just get under the file size cap we list, and then simply work on getting the highest performance possible.

---

## How Grading Works

### Step 1: Submission checks (pass/fail)

**Failing any of these scores a 0**, with an explanation printed:

- **Size check**: the file must be under 30 MB.
- **Artifact check**: it must load and be a `dict` with `binaryModel` and `multiclassModel`, each exposing a callable `.predict()`.
- **Model type check**: the final estimator of each pipeline must be a `RandomForestClassifier`. A better-performing gradient booster still fails this check, because this assignment is specifically about random forests.

On the Gradescope side, one more check runs before your score is fetched:

- **Training file check**: your submission must contain a file named exactly `<netid>_rf.ipynb` or `<netid>_rf.py`, and it must have real content (a notebook with at least one cell, or a script with at least one line of code). Its *contents* are never inspected or graded. Check the filename against your netid before you upload. If you get it wrong, the fix is a rename and a re-upload of the same token, never a retrain, and it does not cost you a portal attempt.

### Step 2: Metrics

| Task | Metrics |
|---|---|
| Binary (`label`) | Accuracy, Precision, Recall, F1 |
| Multi-class (`attack_cat`) | Accuracy, macro Precision, macro Recall, macro F1 |

Unlike the rule-based assignment, your score is not measured as improvement over a starting rule set. Your raw score on each metric determines your points directly:

```
pointsEarned(metric) = ([yourMetricValue / 100] + score adjuster) x maxPointsForMetric
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

No minimum threshold to pass a metric. The score adjuster is something we have set to ensure you are able to still hit 100 on the assignment with good performance. We know a perfect score is highly unlikely, so there is an adjustment in place for each score. Once you run it on gradescope, you'll see how it maps. Please note, it is a relatively insignificant score adjustment, as Random Forest (ML) can perform **very well** on this type of data for MOST of the attack types.

**Rough orientation, so you know whether something is broken or merely unpolished.** Binary metrics should land high; if any of them is below the mid-80s, suspect a structural problem with your pipeline rather than your hyperparameters. Multi-class accuracy will be noticeably lower, which is expected: ten-way discrimination is a harder problem than two-way. Multi-class macro F1 will be lower still, and closing that gap is the interesting part of the assignment. If your macro F1 sits far below your multi-class accuracy, go count how many classes have a poor recall or even 0.00.

### Degenerate strategies

There is no regression penalty this week, but there is also no free lunch. **Predicting the majority class for everything** yields roughly 50% binary accuracy on balanced data and a multi-class macro F1 near 0.02. **Overfitting hard** to your local file produces numbers that look excellent and do not survive the hidden set, which is precisely what makes it dangerous: you will not see the damage in your own output, but you will on submission.

### Step 3: Early Submission Bonus

Extra credit on top of your base grade, uncapped by the 100-point ceiling. A perfect submission 6 days early scores 103.

| Days Early | Bonus |
|---|---|
| 6+ | +3.0 |
| 5 | +2.5 |
| 4 | +2.0 |
| 3 | +1.5 |
| 2 | +1.0 |
| 1 | +0.5 |
| 0 (on the due date) | +0.0 |

Each submission before the deadline banks a fresh tier, so getting a working artifact in early beats saving your attempts for the last minute.

---

## Common Mistakes

**These score 0:**

- Uploading a `.py`, a notebook, a zip, or a folder to the portal instead of the single `.joblib`.
- Submitting a bare `RandomForestClassifier` instead of a pipeline, or one-hot encoding outside the pipeline. Same underlying error, and the most common failure on this assignment.
- Wrong or misspelled dictionary keys.
- Any estimator other than `RandomForestClassifier` as the final step.
- Using `label` or `attack_cat` as an input feature.
- A file over 30 MB.

**This one just costs you a re-upload:**

- Forgetting your `<netid>_rf.ipynb` / `<netid>_rf.py` on Gradescope, or naming it something else (`Untitled.ipynb`, `assignment2.py`, `rf.ipynb`, `<netid>rf.py`). Gradescope will reject the submission, but your token is still valid and your portal attempt is not consumed. Rename and upload again.

**These quietly cost you points instead:**

- **Leaving `handle_unknown` at its default.** The hidden set contains `proto` and `service` values yours does not. Prediction raises an exception, and any metric your model fails to produce counts as 0.
- **Not stratifying your validation split.** `Worms` can land entirely on one side of the split, and your local multi-class numbers become fiction.
- **Reading accuracy and ignoring macro F1.** Accuracy is the metric least sensitive to the thing you are graded on hardest.
- **Not refitting on the full training set** after tuning.
- **Tuning against the same split you keep re-measuring on.** After twenty rounds of adjustment it is no longer held out in any meaningful sense, and your local numbers will read higher than the autograder's. This is the mild form of a mistake that has wrecked published security ML results.
- **Ignoring `requirements.txt`.** Version mismatches between your environment and the autograder's cause failures that are genuinely painful to diagnose.

---

## Testing Locally

Because your submissions are limited, **your local benchmarking is the real feedback loop**. Build it properly and you will know roughly what the autograder is going to tell you before you spend an attempt. Skip it and you are guessing with a resource you cannot get back.

At minimum: hold out a validation split (stratified on `attack_cat`), and print all 8 metrics plus a `classification_report` for every configuration you try. Keep a record. A short table of "what I changed, what happened to each metric" is worth more than a folder of forgotten experiments, and it is what lets you say *why* your final model is the one you submitted.

**Use more data than we gave you.** The full UNSW-NB15 dataset is public, and the [project page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) is linked above. Nothing stops you from downloading it and building your own held-out evaluation sets from records outside your **balanced** 30,000-row training file. This is the closest you can get to the autograder's conditions: a differently composed sample of the same kind of traffic, which your model has never seen. If you want to know whether your configuration generalises or merely memorises, that is how you find out.

A few things worth testing while you are there, since the hidden set is not guaranteed to look like your training file:

- How do your metrics move on a sample that is **not** balanced 50/50 on `label`?
- How do they move on a sample with a different mix of attack categories?
- Do the rare classes your model learned to predict still get predicted when they appear at a different frequency?

**Explore before you train.** Two commands, and they will save you quite some time:

```python
df['attack_cat'].value_counts()               # how imbalanced is the multi-class target?
df[['proto','service','state']].nunique()     # how many columns will one-hot encoding create?
```

Two more once you have a fitted model. Neither is graded, both will teach you something:

- **`confusion_matrix(...)`**. Read across the rows and you can see which categories the model is folding into which. Ask yourself whether a human analyst would make the same mistake.
- **`feature_importances_`** on the fitted forest. Compare the top features against the ones you chose by hand last week. Some of the overlap will surprise you. Treat the ranking as a hint rather than a fact: impurity-based importance is biased toward high-cardinality features, which after one-hot encoding `proto` means biased in a specific and predictable direction here. See Strobl et al. (2007), [DOI](https://doi.org/10.1186/1471-2105-8-25), if you want the details.

See here: [`feature_importances_`](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

---

## Suggested Workflow

1. Load your CSV. Run `value_counts()` on `attack_cat` and look hard at what you are up against.
2. Build the encoder, the classifier, and the pipeline that holds them. Fit both models with modest starting parameters.
3. Split, evaluate, and print all 8 metrics plus the `classification_report`. Write the numbers down. Make any obvious fixes you spot.
4. Save, verify with the reload snippet, and **submit once early**, even if the model is barely tuned. This is not a hedge, it is the point of having attempts. Your local checks tell you the model works in *your* environment; only a real submission tells you it survives one you do not control. It also banks you a bonus tier.
5. Do the real work locally. Attack the multi-class metrics, find the classes with zero (or poor) recall, and change one parameter at a time so you can attribute each result. Test against data from outside your training file.
6. Refit your best configuration on your whole data set (if you can), save, verify, and submit again. Keep an eye on how many attempts you have left.
7. Rename your working notebook or script to `<netid>_rf.ipynb` / `<netid>_rf.py` and upload it to Gradescope with your `token.txt`.

---

## Submission

Submitting takes **two steps, in two different places**. Do both: step 1 produces your score, step 2 is what delivers it to Gradescope.

**The short version:**

- **Portal** gets `randomForestModel.joblib` and nothing else. This is where your attempts are counted.
- **Gradescope** gets `token.txt` plus `<netid>_rf.ipynb` or `<netid>_rf.py`.
- The training file is never read or graded, but it is always checked.

### How attempts are counted

Each portal submission mints one token, and the portal enforces your limit by capping how many tokens it will issue. Your remaining attempts are displayed on the portal and on Gradescope.

This means **a rejected Gradescope upload costs you nothing**. If you forget the training file or misname it, fix it and upload again with the same token. You have not lost an attempt. The scarce resource is running your model against real data on the portal, not the paperwork afterwards.

### Step 1: Upload your model to the course portal

Upload **`randomForestModel.joblib`**, produced by `joblib.dump(...)`. The file itself: not a zip, not a notebook, not your training script.

The portal runs the checks and computes your metrics, then **displays a token** on your submission's results page, tied to that specific submission. There is no file to download. Copy the token value and paste it into a plain text file named **`token.txt`**, containing the token and nothing else.

### Step 2: Upload to Gradescope

Upload **both** of these files together:

| File | What it is | Graded? |
|---|---|---|
| `token.txt` | The token from your portal results page. This is what carries your score across. | This *is* your score |
| `<netid>_rf.ipynb` **or** `<netid>_rf.py` | The notebook or script you trained your random forest in. | **No**, but required |

Miss either one and the submission is rejected with an explanation. Fix it and upload again with the same token.

### Naming your training file

Your training file must be named **`<netid>_rf.ipynb`** or **`<netid>_rf.py`**, where `<netid>` is your own NYU netid, the part of your NYU email address *before* the `@`.

Both formats are accepted and neither is preferred. Submit a notebook if you worked in Jupyter, a script if you worked in a plain `.py` file. **One is enough**; you do not need both.

| If your NYU email is | Your file must be named |
|---|---|
| `abc1234@nyu.edu` | `abc1234_rf.ipynb` or `abc1234_rf.py` |
| `jd42@nyu.edu` | `jd42_rf.ipynb` or `jd42_rf.py` |
| `xy9876@nyu.edu` | `xy9876_rf.ipynb` or `xy9876_rf.py` |

The autograder works out the expected filename from **your own Gradescope account**, so the name is unique to you and there is no list to look up. Just use your netid. A classmate's file will not pass under their name, and yours will not pass under theirs.

What is enforced, precisely:

- The filename must match `<netid>_rf.ipynb` or `<netid>_rf.py`. Capitalisation is forgiven (`ABC1234_RF.ipynb` is accepted), nothing else is. No spaces, no `-rf`, no `_RF_final(2)`.
- The file must have real content. A notebook needs valid `.ipynb` JSON with at least one cell; a script needs at least one line that is not blank and not a comment. An empty placeholder will not pass.
- It may sit inside a folder in your upload; the autograder will find it.
- If you happen to upload both a `.ipynb` and a `.py`, that is fine. The notebook is the one that gets checked.

What is **not** enforced: anything about the contents. Nobody's score depends on what is in the file, and it is never imported, run, re-executed, or checked against your artifact. Submit the notebook or script you actually worked in, exploration, dead ends, plots and all. It is there so your process is on record, not to be marked.

---

## Where This Goes Next

You now have two sets of metrics on the same problem: rules you wrote, and a model that wrote its own. Both halves of the comparison matter. The gain is real, and it came from capacity rather than magic. The residual error is also real, and it sits in the same overlapping region of feature space where your rules failed, for the same reason.

What is new this assignment: a model can be accurate and useless at once, and which metric gets reported decides whether anyone notices. The gap between multi-class accuracy and macro F1 is the substance of a great many unreliable security ML claims. You have now produced that gap yourself, deliberately, and had to close it. Keep that in mind!
