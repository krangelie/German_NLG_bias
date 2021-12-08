import os
from datetime import datetime

import hydra
import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from src.classifier.dataset import get_dataloader
from src.classifier.torch_helpers.eval_torch import evaluate
from src.classifier.utils import build_experiment_name
from src.visualize import aggregate_metrics


class TorchFitter:
    def __init__(self, cfg, X_train, Y_train, X_val=None, Y_val=None,
                 X_test=None, Y_test=None,
                 texts_test=None, scoring=None, trial=None, path=None):
        self.cfg = cfg
        self.hyperparameters = cfg.classifier.majority
        self.gpu_params = cfg.run_mode.gpu
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val
        self.X_test, self.Y_test, self.texts_test = X_test, Y_test, texts_test
        self.classes = set(self.Y_train)
        self.is_tuning = cfg.classifier_mode.name == "tune"
        self.scoring = scoring
        self.trial = trial
        self.path = path
        self.callbacks = self.get_callbacks()
        self.trainer = self.get_trainer()

    def get_callbacks(self):
        early_stopping = EarlyStopping(
            "val_loss", patience=self.hyperparameters.patience
        )  # change to val_loss
        callbacks = [early_stopping]

        if self.cfg.run_mode.store_after_training:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=os.path.join(
                    self.cfg.classifier.model_path,
                    build_experiment_name(self.cfg, f_ending=None),
                    datetime.now().strftime("%b-%d-%Y-%H-%M-%S"),
                ),
                filename="{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}",
                save_top_k=2,
                mode="min",
            )
            callbacks += [checkpoint_callback]

        elif self.is_tuning:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=self.path,
                filename="{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}",
                save_top_k=0,
                mode="min",
            )
            pruning_callback = PyTorchLightningPruningCallback(self.trial, monitor=self.scoring)
            callbacks += [checkpoint_callback, pruning_callback]

        return callbacks

    def get_trainer(self):
        if self.gpu_params.use_amp:
            trainer = pl.Trainer(
                precision=self.gpu_params.precision,
                amp_level=self.gpu_params.amp_level,
                amp_backend=self.gpu_params.amp_backend,
                gpus=self.gpu_params.n_gpus,
                max_epochs=self.hyperparameters.n_epochs,
                progress_bar_refresh_rate=20,
                callbacks=self.callbacks,
            )
        else:
            trainer = pl.Trainer(
                gpus=self.gpu_params.n_gpus,
                max_epochs=self.hyperparameters.n_epochs,
                progress_bar_refresh_rate=20,
                callbacks=self.callbacks,
            )
        return trainer

    def cv_loop(self, model):
        skf = StratifiedKFold(n_splits=self.cfg.classifier_mode.cv_folds)
        if self.is_tuning:
            avg_score = self.cv_loop_tune_mode(model, skf)
        else:
            avg_score = self.cv_loop_train_mode(model, skf)
        return avg_score


    def cv_loop_train_mode(self, model, skf):
        accs, result_dicts, confs = [], [], []
        test_loader = get_dataloader(self.X_test, self.Y_test,
                                     self.hyperparameters.batch_size,
                                     shuffle=False)
        for train_index, val_index in skf.split(self.X_train, self.Y_train):
            print(f"Num train {len(train_index)}, num val {len(val_index)}")

            if self.path:
                output_path = self.path
            else:
                output_path = hydra.utils.to_absolute_path(self.cfg.run_mode.plot_path)
                output_path = os.path.join(output_path, self.cfg.classifier.name)

            model = self.fit(model, train_index, val_index)

            mean_acc, results_dict, conf_matrix_npy = evaluate(
                model, test_loader, self.texts_test, self.classes,
                f"{self.cfg.embedding.name}_{self.cfg.classifier.name}", output_path
            )
            accs.append(mean_acc)
            result_dicts.append(results_dict)
            confs.append(conf_matrix_npy)

        aggregate_metrics(result_dicts, confs, output_path)
        print(
            f"--- Avg. accuracy across {self.cfg.classifier_mode.cv_folds} folds (cv-score) is: "
            f"{np.mean(accs)}, SD={np.std(accs)}---"
        )
        if self.cfg.classifier_mode.name == "incremental_train":
            return accs
        else:
            return np.mean(accs)

    def cv_loop_tune_mode(self, model, skf):
        scores = []
        for train_index, val_index in skf.split(self.X_train, self.Y_train):
            model = self.fit(model, train_index, val_index)
            scores.append(self.trainer.callback_metrics[self.scoring].item())
        return np.mean(scores)

    def fit(self, model, train_index, val_index):
        x_train, x_val = self.X_train[train_index], self.X_train[val_index]
        y_train, y_val = (
            self.Y_train[train_index],
            self.Y_train[val_index],
        )
        train_loader = get_dataloader(x_train, y_train, self.hyperparameters.batch_size)
        val_loader = get_dataloader(x_val, y_val, self.hyperparameters.batch_size, shuffle=False)

        mlflow.pytorch.autolog()
        with mlflow.start_run():
            #self.trainer.logger.log_hyperparams(self.hyperparameters)
            self.trainer.fit(model, train_loader, val_loader)
        print(self.trainer.current_epoch)
        return model








