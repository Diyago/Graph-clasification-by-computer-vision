import importlib
import ntpath
import os
import os.path
import random
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def qwk_metric(y_pred, y, detach=True):
    y_pred = torch.round(y_pred).to("cpu").numpy().argmax(1)
    y = y.to("cpu").numpy()
    # k = torch.tensor(cohen_kappa_score(torch.round(y_pred), y, weights='quadratic'), device='cuda:0')
    # k[k != k] = 0
    # k[torch.isinf(k)] = 0
    # return k
    return cohen_kappa_score(y_pred, y)


def create_folds(configs):
    folds = pd.read_csv(configs["train_csv"], sep=';')
    train_labels = folds[configs["target_col"]].values
    kf = KFold(n_splits=configs["nfolds"])

    for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
        folds.loc[val_index, "fold"] = int(fold)
    folds["fold"] = folds["fold"].astype(int)
    folds.to_csv(configs["folds_path"], index=None)
    return folds


class OptimizedRounder(object):
    """
    Usage:

    opt = OptimizedRounder()
    preds,y = learn.get_preds(DatasetType.Test)
    tst_pred = opt.predict(preds, coefficients)
    test_df.diagnosis = tst_pred.astype(int)
    test_df.to_csv('submission.csv',index=False)
    print ('done')


    """

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif coef[0] <= pred < coef[1]:
                X_p[i] = 1
            elif coef[1] <= pred < coef[2]:
                X_p[i] = 2
            elif coef[2] <= pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )
        print(-loss_partial(self.coef_["x"]))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_["x"]


def convert_model(model, full_checkpoint_path, output_path):
    checkpoint = torch.load(
        full_checkpoint_path, map_location=lambda storage, loc: storage
    )

    sanitized_dict = {}
    for k, v in checkpoint["state_dict"].items():
        sanitized_dict[k.replace("model.model", "model")] = v
        # sanitized_dict[k.replace("model.", "")] = v

    sample = torch.rand(1, 3, 256, 256, dtype=torch.float32)
    model.load_state_dict(sanitized_dict)
    scripted_model = torch.jit.trace(model, sample)
    filename = ntpath.basename(full_checkpoint_path).replace("=", "")
    os.makedirs(output_path, exist_ok=True)
    scripted_model.save(f"{output_path}/{filename}.pth")


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])
