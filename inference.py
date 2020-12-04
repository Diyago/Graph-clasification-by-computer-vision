import codecs
import os
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from poyo import parse_string
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from common_blocks.datasets import TestDataset
from common_blocks.transforms import get_transforms
from common_blocks.utils import create_folds
from common_blocks.utils import plot_prec_recall_vs_tresh
from models.lightningclassifier import LightningClassifier

with codecs.open("config/config_classification.yml", encoding="utf-8") as ymlfile:
    config_yaml = ymlfile.read()
    config = parse_string(config_yaml)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_tta_preds(net, images, augment=["null"]):
    with torch.no_grad():
        net.eval()
        if 1:  # null
            logit = net(images)
            probability = torch.sigmoid(logit)
        if "flip_lr" in augment:
            logit = net(torch.flip(images, dims=[3]))
            probability += torch.sigmoid(logit)
        if "flip_ud" in augment:
            logit = net(torch.flip(images, dims=[2]))
            probability += torch.sigmoid(logit)
        probability = probability / len(augment)
    return probability.data.cpu().numpy()


def get_all_models(path):
    all_models = []

    for model_path in os.listdir(path):
        model = LightningClassifier(config)
        checkpoint = torch.load(
            os.path.join(path, model_path), map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.freeze()
        all_models.append(model)
    return all_models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    folds = create_folds(config["test_inference"])

    dataset = TestDataset(
        folds,
        config["test_inference"]["Dataset"],
        transform=get_transforms(data="valid", width=config["test_inference"]["Dataset"]["target_width"],
                                 height=config["test_inference"]["Dataset"]["target_height"]))
    loader = DataLoader(dataset, **config["test_inference"]["loader"])
    all_models = get_all_models(config["test_inference"]["models_path"])
    model_results = {"preds": [], "image_names": [], "image_label": {}}
    for fnames, images in tqdm(loader):
        images = images.to(device)
        batch_preds = None
        for model in all_models:
            if batch_preds is None:
                batch_preds = get_tta_preds(model, images, augment=config["test_inference"]["TTA"])
            else:
                batch_preds += get_tta_preds(model, images, augment=config["test_inference"]["TTA"])
        model_results["image_names"].extend(list(fnames))
        model_results["preds"].append(batch_preds)

    model_results['preds'] = np.concatenate(model_results["preds"]).ravel() / len(all_models)
    model_results["image_label"] = list((model_results["preds"] > config["test_inference"]["threshold"]
                                         ).astype(int))
    model_results = pd.DataFrame(model_results)
    model_results['gt_label'] = folds.label.reset_index(drop=True)

    model_results.to_excel('./lightning_logs/model_preds_train.xlsx', index=True)
    print('ROC AUC', round(metrics.roc_auc_score(model_results['gt_label'], model_results['preds']), 3))
    print('Precision', round(model_results[model_results['image_label'] == 1].gt_label.mean(), 3))
    print('Recall', round(metrics.recall_score(model_results['gt_label'], model_results['image_label']), 3))
    print('F1_score', round(metrics.f1_score(model_results['gt_label'], model_results['image_label']), 3))
    print('MAP', round(metrics.average_precision_score(model_results['gt_label'], model_results['preds']), 3))

    prec, rec, tre = metrics.precision_recall_curve(model_results['gt_label'], model_results['preds'])

    plot_prec_recall_vs_tresh(prec, rec, tre)
    plt.show()
