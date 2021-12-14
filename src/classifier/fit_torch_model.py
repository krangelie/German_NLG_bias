import os
from abc import ABC, abstractmethod
from datetime import datetime

import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback


from src.classifier.dataset import get_dataloader, RegardBertDataset
from src.classifier.eval_model import evaluate, get_metrics, \
    gather_preds_and_labels
from src.classifier.utils import build_experiment_name
from src.visualize import aggregate_metrics


class TorchFitter(ABC):
    def __init__(self, cfg, X_train, Y_train, X_test=None, Y_test=None,
                 texts_test=None, scoring=None, trial=None, path=None):
        self.cfg = cfg
        self.hyperparameters = cfg.classifier.majority
        self.gpu_params = cfg.run_mode.gpu
        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test, self.texts_test = X_test, Y_test, texts_test
        self.classes = set(self.Y_train)
        self.is_tuning = cfg.classifier_mode.name == "tune"
        self.scoring = scoring
        self.trial = trial
        if path is None:
            path = cfg.run_mode.plot_path
            path = os.path.join(path, cfg.classifier.name)
        self.path = path
        self.callbacks = self.get_callbacks()

    @abstractmethod
    def get_callbacks(self):
        pass

    @abstractmethod
    def get_trainer(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def fit_and_eval(self, model, x_train, x_val, y_train, y_val):
        return None, None, None


    def cv_loop_train_mode(self, model, skf):
        accs, result_dicts, confs = [], [], []
        for train_index, val_index in skf.split(self.X_train, self.Y_train):
            print(f"Num train {len(train_index)}, num val {len(val_index)}")
            x_train, x_val = self.X_train[train_index], self.X_train[val_index]
            y_train, y_val = (
                self.Y_train[train_index],
                self.Y_train[val_index],
            )
            mean_acc, results_dict, conf_matrix_npy = self.fit_and_eval(model, x_train, x_val, y_train, y_val)
            accs.append(mean_acc)
            result_dicts.append(results_dict)
            confs.append(conf_matrix_npy)

        aggregate_metrics(result_dicts, confs, self.path)
        print(
            f"--- Avg. accuracy across {self.cfg.classifier_mode.cv_folds} folds (cv-score) is: "
            f"{np.mean(accs)}, SD={np.std(accs)}---"
        )
        if self.cfg.classifier_mode.name == "incremental_train":
            return accs
        else:
            return np.mean(accs)


class PLFitter(TorchFitter):
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

    def fit(self, model, x_train, x_val, y_train, y_val):
        self.trainer = self.get_trainer()
        train_loader = get_dataloader(x_train, y_train, self.hyperparameters.batch_size)
        val_loader = get_dataloader(x_val, y_val, self.hyperparameters.batch_size, shuffle=False)

        mlflow.pytorch.autolog()
        with mlflow.start_run():
            #self.trainer.logger.log_hyperparams(self.hyperparameters)
            self.trainer.fit(model, train_loader, val_loader)
        print(self.trainer.current_epoch)
        return model

    def evaluate(self, model):
        test_loader = get_dataloader(self.X_test, self.Y_test,
                                     self.hyperparameters.batch_size,
                                     shuffle=False)
        preds, labels = gather_preds_and_labels(model, test_loader)
        mean_acc, results_dict, conf_matrix_npy = evaluate(preds, labels, self.texts_test,
                                                           self.classes,
                                                           f"{self.cfg.embedding.name}_{self.cfg.classifier.name}",
                                                           self.path)
        return mean_acc, results_dict, conf_matrix_npy

    def fit_and_eval(self, model, x_train, x_val, y_train, y_val):
        model = self.fit(model, x_train, x_val, y_train, y_val)
        mean_acc, results_dict, conf_matrix_npy = self.evaluate(model)
        return mean_acc, results_dict, conf_matrix_npy

    def cv_loop_and_eval(self, model):
        skf = StratifiedKFold(n_splits=self.cfg.classifier_mode.cv_folds)
        if self.is_tuning:
            avg_score = self.cv_loop_tune_mode(model, skf)
        else:
            avg_score = self.cv_loop_train_mode(model, skf)
        return avg_score

    def cv_loop_tune_mode(self, model, skf):
        scores = []
        for train_index, val_index in skf.split(self.X_train, self.Y_train):
            x_train, x_val = self.X_train[train_index], self.X_train[val_index]
            y_train, y_val = (
                self.Y_train[train_index],
                self.Y_train[val_index],
            )
            model = self.fit(model, x_train, x_val, y_train, y_val)
        scores.append(self.trainer.callback_metrics[self.scoring].item())
        return np.mean(scores)


class HFFitter(TorchFitter):
    def get_callbacks(self):
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.hyperparameters.patience
        )  # change to val_loss
        callbacks = [early_stopping]
        return callbacks

    def get_trainer(self, model, train_dataset, val_dataset):
        args = TrainingArguments(
            output_dir=self.path,
            evaluation_strategy="steps",
            eval_steps=500,
            per_device_train_batch_size=self.hyperparameters.batch_size,
            per_device_eval_batch_size=self.hyperparameters.batch_size,
            num_train_epochs=self.hyperparameters.n_epochs,
            seed=0,
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )
        return trainer

    def cv_loop_and_eval(self, model):
        skf = StratifiedKFold(n_splits=self.cfg.classifier_mode.cv_folds)
        avg_score = self.cv_loop_train_mode(model, skf)
        return avg_score

    def fit(self, model, x_train, x_val, y_train, y_val, return_model=False):
        train_dataset = RegardBertDataset(x_train, y_train)
        val_dataset = RegardBertDataset(x_val, y_val)
        trainer = self.get_trainer(model, train_dataset, val_dataset)
        trainer.train()
        torch.cuda.empty_cache()
        if return_model:
            return trainer.model
        else:
            return trainer

    def compute_metrics(self, p):
        logits, labels = p
        preds = np.argmax(logits, axis=1)
        return get_metrics(preds, labels)


    def evaluate(self, trainer):
        test_dataset = RegardBertDataset(self.X_test, self.Y_test)
        logits, labels, results_dict = trainer.predict(test_dataset)
        preds = np.argmax(logits, axis=1)
        mean_acc, results_dict, conf_matrix_npy = evaluate(preds, labels, self.texts_test,
                                                           self.classes,
                                                           f"{self.cfg.embedding.name}_{self.cfg.classifier.name}",
                                                           self.path)
        return mean_acc, results_dict, conf_matrix_npy

    def fit_and_eval(self, model, x_train, x_val, y_train, y_val):
        trainer = self.fit(model, x_train, x_val, y_train, y_val)
        mean_acc, results_dict, conf_matrix_npy = self.evaluate(trainer)
        return mean_acc, results_dict, conf_matrix_npy

