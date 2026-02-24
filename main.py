
# ------------------------------------------------------------
# Subpopulation Fairness + Invariance Training on Concrete Data
# AIF360 pipeline: Reweighing + DIR -> Adversarial Debiasing -> Equalized Odds
# ------------------------------------------------------------

import os, random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# ---------------------------
# 0) Reproducibility
# ---------------------------
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.set_random_seed(SEED)

# ---------------------------
# 1) Load dataset
# ---------------------------
FILE_PATH = "Concrete_Data.xls"  # keep next to this script
assert os.path.exists(FILE_PATH), "Concrete_Data.xls not found."

df = pd.read_excel(FILE_PATH, header=0)
df.columns = [
    'cement','slag','fly_ash','water','superplasticizer',
    'coarse_aggregate','fine_aggregate','age','strength'
]

# Binary label: high strength (by median)
median_strength = df['strength'].median()
df['high_strength'] = (df['strength'] > median_strength).astype(int)

# Group (engineering subpopulation): high vs low cement (by median)
median_cement = df['cement'].median()
df['high_cement'] = (df['cement'] > median_cement).astype(int)

# ---------------------------
# 2) Wrap into AIF360 dataset
# ---------------------------
dataset = StandardDataset(
    df=df,
    label_name='high_strength',
    favorable_classes=[1],
    protected_attribute_names=['high_cement'],
    privileged_classes=[[1]],   # privileged = high cement
    categorical_features=[],
    features_to_keep=[
        'cement','slag','fly_ash','water','superplasticizer',
        'coarse_aggregate','fine_aggregate','age','high_cement'
    ]
)

# Split
train, test = dataset.split([0.7], shuffle=True, seed=SEED)

# ---------------------------
# 3) Pre-processing: Reweighing + Disparate Impact Remover
# ---------------------------
priv = [{'high_cement': 1}]
unpriv = [{'high_cement': 0}]

rw = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
train_rw = rw.fit_transform(train)

dir_ = DisparateImpactRemover(repair_level=1.0)
train_dir = dir_.fit_transform(train_rw)

# ---------------------------
# 4) Scaling (keep protected column intact!)
# ---------------------------
feat_names = train_dir.feature_names
z_name = 'high_cement'
z_idx = feat_names.index(z_name)

def split_XZ(dset, z_col_idx):
    X = np.delete(dset.features, z_col_idx, axis=1)
    Z = dset.features[:, [z_col_idx]]
    return X, Z

X_tr, Z_tr = split_XZ(train_dir, z_idx)
X_te, Z_te = split_XZ(test, z_idx)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# put Z back in original column slot
train_dir.features = np.insert(X_tr, z_idx, Z_tr.ravel(), axis=1)
test.features      = np.insert(X_te, z_idx, Z_te.ravel(), axis=1)

# ---------------------------
# 5) Invariance training via augmentation (FIXED: keeps protected_attributes aligned)
# ---------------------------
def augment_with_noise(aif_dataset, z_col_idx, noise_std=0.05, copies=2, rng=None):
    """
    Duplicate dataset with Gaussian noise on non-protected features.
    Keeps labels, protected attrs, and weights aligned in size.
    """
    if rng is None:
        rng = np.random.RandomState(SEED)

    base = aif_dataset
    X_all = base.features.copy()                    # [n, d]
    y_all = base.labels.copy()                      # [n, 1]
    w_all = base.instance_weights.copy()            # [n,]
    # Protected attr column as [n, 1]
    Z_all = X_all[:, [z_col_idx]].copy()

    # Remove Z from X for perturbation
    X_woZ = np.delete(X_all, z_col_idx, axis=1)     # [n, d-1]

    X_woZ_list = [X_woZ]
    Z_list     = [Z_all]
    y_list     = [y_all]
    w_list     = [w_all]

    for _ in range(copies):
        noise = rng.normal(loc=0.0, scale=noise_std, size=X_woZ.shape)
        X_noisy = X_woZ + noise
        X_woZ_list.append(X_noisy)
        Z_list.append(Z_all.copy())
        y_list.append(y_all.copy())
        w_list.append(w_all.copy())

    X_woZ_aug = np.vstack(X_woZ_list)               # [n*(copies+1), d-1]
    Z_aug     = np.vstack(Z_list)                   # [n*(copies+1), 1]
    y_aug     = np.vstack(y_list)                   # [n*(copies+1), 1]
    w_aug     = np.hstack(w_list)                   # [n*(copies+1),]

    # Reinsert Z at its original column index
    X_aug = np.insert(X_woZ_aug, z_col_idx, Z_aug.ravel(), axis=1)

    # Build a consistent StandardDataset clone
    aug_ds = base.copy(deepcopy=True)
    aug_ds.features = X_aug
    aug_ds.labels = y_aug
    aug_ds.instance_weights = w_aug
    aug_ds.protected_attributes = Z_aug.astype(float)  # keep aligned
    return aug_ds

train_aug = augment_with_noise(train_dir, z_col_idx=z_idx, noise_std=0.05, copies=2)

# ---------------------------
# 6) In-processing: Adversarial Debiasing (TF v1)
# ---------------------------
tf.reset_default_graph()
sess = tf.Session()

ad = AdversarialDebiasing(
    privileged_groups=priv,
    unprivileged_groups=unpriv,
    scope_name='adv_debiasing',
    debias=True,
    sess=sess,
    # Optional knobs to tune:
    # adversary_loss_weight=0.5,
    # num_epochs=50,
    # batch_size=128,
)
ad.fit(train_aug)

test_pred = ad.predict(test)  # labels (+ scores if available)

# ---------------------------
# 7) Metrics after in-processing
# ---------------------------
debias_metric = BinaryLabelDatasetMetric(
    test_pred, unprivileged_groups=unpriv, privileged_groups=priv
)
cls_metric = ClassificationMetric(
    test, test_pred, unprivileged_groups=unpriv, privileged_groups=priv
)

print("== After In-processing ==")
print("Disparate Impact:", debias_metric.disparate_impact())
print("Statistical Parity Difference:", debias_metric.statistical_parity_difference())
print("Equal Opportunity Difference:", cls_metric.equal_opportunity_difference())
print("Accuracy:", accuracy_score(test.labels, test_pred.labels.ravel()))

# ---------------------------
# 8) Post-processing: Equalized Odds
# ---------------------------
eq_odds = EqOddsPostprocessing(
    unprivileged_groups=unpriv,
    privileged_groups=priv
).fit(test, test_pred)

test_eq = eq_odds.predict(test_pred)

debias_metric_post = BinaryLabelDatasetMetric(
    test_eq, unprivileged_groups=unpriv, privileged_groups=priv
)
cls_metric_post = ClassificationMetric(
    test, test_eq, unprivileged_groups=unpriv, privileged_groups=priv
)

print("== After Equalized Odds ==")
print("Disparate Impact:", debias_metric_post.disparate_impact())
print("Statistical Parity Difference:", debias_metric_post.statistical_parity_difference())
print("Equal Opportunity Difference:", cls_metric_post.equal_opportunity_difference())
print("Accuracy:", accuracy_score(test.labels, test_eq.labels.ravel()))

sess.close()
