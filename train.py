import codecs
import os
import warnings

import numpy as np
import pytorch_lightning as pl
from poyo import parse_string
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateLogger
from common_blocks.datasets import TrainDataset
from common_blocks.logger import init_logger
from common_blocks.transforms import get_transforms
from common_blocks.utils import seed_torch, create_folds
from models.lightningclassifier import LightningClassifier

with codecs.open("config/config_classification.yml", encoding="utf-8") as ymlfile:
    config_yaml = ymlfile.read()
    config = parse_string(config_yaml)

LOGGER = init_logger(config['logger_path']['main_logger'])
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == '__main__':
    seed_torch(seed=config['total_seed'])
    folds = create_folds(config['validation'])
    fold_best_metrics = []
    for fold in range(config['validation']['nfolds']):
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        #todo move dataset, loader to func
        train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True),
                                     config['Train']['Dataset'],
                                     transform=get_transforms(data='train',
                                                              width=config["Train"]["Dataset"]["target_width"],
                                                              height=config["Train"]["Dataset"]["target_height"]))
        valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True),
                                     config['Val']['Dataset'],
                                     transform=get_transforms(data='valid',
                                                              width=config["Val"]["Dataset"]["target_width"],
                                                              height=config["Val"]["Dataset"]["target_height"]))

        train_loader = DataLoader(train_dataset, **config['Train']['loader'])
        valid_loader = DataLoader(valid_dataset, **config['Val']['loader'])
        #import ipdb;        ipdb.set_trace()
        tb_logger = TensorBoardLogger(save_dir=config['logger_path']['lightning_logger'],
                                      name=config['model_params']['model_name'], version=f'fold_{fold + 1}')
        os.makedirs('{}/{}'.format(config['logger_path']['lightning_logger'], config['model_params']['model_name']),
                    exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            filepath=tb_logger.log_dir + config['training']['ModelCheckpoint']['path'],
            **config['training']['ModelCheckpoint']['kwargs'])
        early_stop_callback = EarlyStopping(**config['training']['early_stop_callback'])

        model = LightningClassifier(config)
        lr_logger = pl.callbacks.LearningRateLogger()
        trainer = pl.Trainer(logger=tb_logger,
                             callbacks=[lr_logger],
                             early_stop_callback=early_stop_callback,
                             checkpoint_callback=checkpoint_callback,
                             **config['training']['Trainer'])
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
        fold_best_metrics.append(np.max(model.val_metrics))
        
    print('MEAN METRIC:', round(np.mean(fold_best_metrics), 3), 'std', round(np.std(fold_best_metrics), 3))
    print('ALL METRICS:', [round(x, 3) for x in fold_best_metrics])
