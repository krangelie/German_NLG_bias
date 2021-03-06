from pprint import pformat
import os
import logging
import sys
from datetime import datetime

import hydra.utils
import yaml

import torch
import optuna

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.utils.class_weight import compute_sample_weight

from src.classifier.classifiers import RegardLSTM, RegardBERT
from src.classifier.fit_torch_model import PLFitter
from src.classifier.utils import build_experiment_name
from src.classifier.tune_suggest_functions import suggest_lstm, suggest_xgb, \
    suggest_rf, suggest_sbert

class Tuner:
    def __init__(self, cfg, X, Y, X_val=None, Y_val=None, fold=None):
        # uses TPESampler by default
        self.cfg = cfg
        if cfg.dev_settings.annotation == "unanimous":
            hyperparameters = cfg.classifier.unanimous
        elif cfg.dev_settings.annotation == "majority":
            hyperparameters = cfg.classifier.majority
        self.log = logging.getLogger(__name__)
        optuna.logging.enable_propagation()
        self.X, self.Y = X, Y
        self.X_val, self.Y_val = X_val, Y_val
        self.model_type = cfg.classifier.name
        self.model_path = hydra.utils.to_absolute_path(hyperparameters.model_path)
        self.model_params = cfg.classifier.hyperparameters
        self.tune_params = cfg.classifier_mode
        self.n_embed = cfg.embedding.n_embed
        self.n_output = hyperparameters.n_output
        self.unit = (
            hyperparameters.unit if self.model_type == "lstm" else None
        )  # if recurrent net: GRU or LSTM
        self.patience = hyperparameters.patience if self.model_type != "xgb" else None
        self.n_epochs = hyperparameters.n_epochs if self.model_type != "xgb" else None
        self.timestamp = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_params = cfg.run_mode.gpu
        self.study_name = self.tune_params.study_name
        self.experiment_name = build_experiment_name(cfg, fold=fold, f_ending=None)
        self.outer_fold = fold
        self.yaml_path = hydra.utils.to_absolute_path(cfg.classifier_mode.yaml_path)
        self.annotation = cfg.dev_settings.annotation

    def find_best_params(self):

        study = self.setup_study()

        self.log.info(
            f"Tuning with total of {len(self.X)} train samples. Results in {len(self.X)-(len(self.X)//self.tune_params.cv_folds)} train samples per fold."
        )

        study.optimize(self.objective, n_trials=self.tune_params.trials)

        self.log.info(f"Finished hyperparameter tuning for {self.model_type}")
        self.log.info("Best params are:\n")
        self.log.info(pformat(study.best_params))
        out_yaml_path = os.path.join(self.yaml_path, f"{self.model_type}")
        os.makedirs(out_yaml_path, exist_ok=True)
        with open(
            os.path.join(out_yaml_path, f"{self.model_type}_{self.outer_fold}.yaml"),
            "a",
        ) as file:
            yaml.dump({self.annotation: study.best_params}, file)

    def setup_study(self):
        store_path = hydra.utils.to_absolute_path(self.tune_params.store_path)
        db_name = hydra.utils.to_absolute_path(self.tune_params.db_name)
        if self.tune_params.resume_study:
            if self.study_name == "" or db_name == "":
                self.log.warning(
                    "\nABORTING load_study because either db_name or study_name haven't been specified."
                )
                sys.exit()
            elif not os.path.exists(os.path.join(store_path, db_name)):
                self.log.warning(
                    f"\nABORTING because {os.path.join(store_path, db_name)} does not exist."
                )
                sys.exit()

            study = optuna.load_study(
                study_name=self.study_name,
                storage="sqlite:///" + store_path + "/" + db_name,
            )
            self.log.info(f"\nResuming hyperparameter tuning for {self.study_name}")

        else:
            if self.study_name == "":
                cls_name = (
                    f"{self.model_type}_"
                    if self.model_type != "lstm"
                    else f"{self.unit}_"
                )
                self.study_name = (
                    f"{cls_name}_{self.experiment_name}"
                    f"_{self.tune_params.cv_folds}folded"
                    f"_{self.tune_params.scoring}"
                )

            os.makedirs(store_path, exist_ok=True)
            study = optuna.create_study(
                direction="maximize",
                study_name=self.study_name + "_" + self.timestamp,
                storage="sqlite:///" + store_path + "/" f"{self.study_name}.db",
            )
            self.log.info(
                f"\nStarting hyperparameter tuning for {self.study_name + '_' + self.timestamp}"
            )

        return study

    def objective(self, trial):
        print(f"Num folds is {self.tune_params.cv_folds}.")
        if self.model_type not in ["lstm", "transformer"]:
            # For Sklearn API
            if self.model_type == "xgb":
                classifier = suggest_xgb(self.model_params, trial)

            elif self.model_type == "rf":
                classifier = suggest_rf(self.model_params, trial)

            scorer = self._get_scorer()
            # TODO: add option for train - val training
            scores = cross_val_score(
                classifier,
                self.X,
                self.Y,
                scoring=scorer,
                n_jobs=self.tune_params.n_jobs,
                cv=self.tune_params.cv_folds,
                fit_params={"sample_weight": self._get_sample_weights()},
            )
            score = scores.mean()

        else:
            # For PyTorch API
            scoring = (
                "val_acc" if self.tune_params.scoring == "accuracy_score" else "val_f1"
            )

            if self.model_type == "lstm":
                hyperparameters, batch_size = suggest_lstm(
                    self.model_params,
                    trial,
                    self.Y,
                    self.device,
                    self.unit,
                    self.n_embed,
                    self.n_output,
                )
                model = RegardLSTM(**hyperparameters)
            elif self.model_type == "transformer":
                hyperparameters, batch_size = suggest_sbert(
                    self.model_params,
                    trial,
                    self.Y,
                    self.device,
                    self.n_embed,
                    self.n_output,
                )
                model = RegardBERT(**hyperparameters)

            dest_path = os.path.join(
                self.model_path,
                self.study_name,
                self.timestamp,
                f"trial_{trial.number}",
            )

            fitter = PLFitter(self.cfg, self.X, self.Y,
                                 scoring=scoring, trial=trial, path=dest_path)
            score = fitter.cv_loop_and_eval(model)
        return score

    def _get_sample_weights(self):
        weight_vector = compute_sample_weight("balanced", self.Y)
        return weight_vector

    def _get_scorer(self):
        if self.tune_params.scoring == "roc_auc_score":
            scorer = make_scorer(
                eval(self.tune_params.scoring),
                average=self.tune_params.average,
                multi_class=self.tune_params.multi_class,
            )
        elif self.tune_params.scoring == "accuracy_score":
            scorer = make_scorer(eval(self.tune_params.scoring))
        else:
            scorer = make_scorer(
                eval(self.tune_params.scoring),
                average=self.tune_params.average,
            )
        return scorer
