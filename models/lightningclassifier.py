import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.core.memory import ModelSummary
from sklearn.metrics import average_precision_score
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from common_blocks.losses import LabelSmoothingCrossEntropyBCE, FocalLoss
from models.pretrained_models import get_model_output


class LightningClassifier(pl.LightningModule):
    def __init__(self, config):
        super(LightningClassifier, self).__init__()
        self.hparams = config
        self.model = get_model_output(**config['model_params'])  # CustomSEResNeXt(config['model_params'])
        self.val_metrics = []

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        return self.model.forward(x)

    def get_loss(self, y_preds, labels):
        if self.hparams['training']['loss'] == 'FocalLoss':
            loss_func = FocalLoss()
            return loss_func(y_preds, labels.float())
        elif self.hparams['training']['loss'] == 'LabelSmoothingCrossEntropyBCE':
            loss_func = LabelSmoothingCrossEntropyBCE()
            return loss_func(y_preds, labels.float())
        elif self.hparams['training']['loss'] == 'BCELoss':
            loss_func = nn.BCELoss()
            return loss_func(y_preds, labels.float())
        else:
            raise NotImplementedError("This loss {} isn't implemented".format(self.hparams['training']['loss']))

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_preds = self.forward(x)
        loss = self.get_loss(torch.flatten(torch.sigmoid(y_preds)), y)
        with torch.no_grad():
            preds = torch.flatten((y_preds > 0.5).int().cpu())
            metric = (preds == y.int().cpu()).sum().item() / len(y)

        logs = {'train_loss': loss, 'train_metric': metric}
        progress_bar = {'train_metric': metric}
        return {'loss': loss, 'metric': metric, 'log': logs, "progress_bar": progress_bar}

    def training_step_mixup(self, train_batch, batch_idx):
        x, y = train_batch
        x, targets_a, targets_b, lam = mixup_data(x, y, alpha=1)
        x, targets_a, targets_b = map(Variable, (x, targets_a, targets_b))
        y_preds = torch.flatten(torch.sigmoid(self.model(x)))

        loss = self.mixup_criterion(y_preds, targets_a.float(), targets_b.float(), lam)

        total = x.size(0)
        correct = (lam * y_preds.eq(targets_a.data).cpu().sum().float()
                   + (1 - lam) * y_preds.eq(targets_b.data).cpu().sum().float())
        logs = {'train_loss': loss, 'train_metric': correct / total}
        progress_bar = {'train_metric': correct / total}
        return {'loss': loss, 'metric': correct / total, 'log': logs, "progress_bar": progress_bar}

    def training_epoch_end(self, outputs):

        avg_loss_train = torch.stack([x['loss'] for x in outputs]).mean()
        avg_metric_train = np.stack([x['metric'] for x in outputs]).mean()

        tensorboard_logs = {'avg_train_loss': avg_loss_train, 'acc_train_metric': avg_metric_train}
        print('\ntrain', 'avg_train_metric', avg_metric_train)
        return {'avg_train_loss': avg_loss_train, 'avg_train_metric': avg_metric_train, 'log': tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        with torch.no_grad():
            y_preds = self.forward(x)

        loss = self.get_loss(torch.flatten(torch.sigmoid(y_preds)), y)
        preds = torch.flatten((y_preds > 0.5).int().cpu())
        metric = (preds == y.int().cpu()).sum().item() / len(y)
        return {'val_loss': loss,
                'pred_label': (y_preds).cpu(),
                'val_metric': metric,
                'label': y.int().cpu()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_metric = np.stack([x['val_metric'] for x in outputs]).mean()

        print('\nval', 'avg_val_metric', avg_metric)
        all_pred_label = torch.cat([x['pred_label'] for x in outputs])
        all_label = torch.cat([x['label'] for x in outputs])
        try:
            avg_metric = average_precision_score(y_score=all_pred_label, y_true=all_label)
        except ValueError:
            avg_metric = 0.5
        print('validation_epoch_end', avg_metric)
        self.val_metrics.append(avg_metric)
        tensorboard_logs = {'val_loss': avg_loss, 'acc_metric': avg_metric,
                            'average_precision_score': avg_metric}
        return {'avg_val_loss': avg_loss, 'avg_val_metric': avg_metric, 'log': tensorboard_logs,
                "progress_bar": tensorboard_logs}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure):
        if self.trainer.global_step < self.hparams['training']['warmup_steps']:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams['training']['warmup_steps'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams['training']['optimizer']['kwargs']['lr']
        optimizer.step()
        optimizer.zero_grad()

    def prepare_data(self):
        pass

    def summarize(self, mode: str) -> None:
        if self.hparams['model_params']['show_model_summary']:
            model_summary = ModelSummary(self, mode=mode)
            log.info('\n' + model_summary.__str__())

    def configure_optimizers(self):
        if self.hparams['training']['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.hparams['training']['optimizer']['kwargs'])
        else:
            NotImplementedError(
                "This optimizer {} isn't implemented".format(self.hparams['training']['optimizer']['name']))

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **self.hparams['training']['scheduler']['ReduceLROnPlateau']),
            **self.hparams['training']['scheduler']['kwargs']
        }
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_preds = self.forward(x)
            y_preds = torch.flatten(torch.sigmoid(y_preds))

        return {'y_preds': (y_preds).cpu(),
                'y_labels': torch.flatten((y_preds > 0.5).int().cpu())
                }

    def test_epoch_end(self, outputs):
        all_pred_label = torch.cat([x['y_labels'] for x in outputs])
        all_preds = torch.cat([x['y_preds'] for x in outputs])
        return {'all_pred_label': all_pred_label,
                'all_preds': all_preds}

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.get_loss(pred, y_a) + (1 - lam) * self.get_loss(pred, y_b)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
