import os
from datetime import datetime

import hydra.utils
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import BertForSequenceClassification

from src.classifier.eval_model import evaluate_sklearn
from src.classifier.fit_torch_model import HFFitter, PLFitter
from src.classifier.get_classifier_or_embedding import compute_weight_vector, get_classifier, \
    save_pretrained_sklearn
from src.classifier.utils import build_experiment_name


def train_classifier(cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test, texts_test, logger,
                     seed=42):
    classes = set(Y_train)

    if cfg.classifier.name.startswith("lstm") or "sentence" in cfg.embedding.path:
        score = train_torch_model(cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test, texts_test,
                                  classes, seed=seed)
    elif cfg.embedding.path.startswith("bert"):
        score = train_torch_model(cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test, texts_test,
                                  classes, is_hf=True, seed=seed)
    else:
        score = train_sklearn(
            cfg, X_train, X_test, Y_train, Y_test, logger, texts_test
        )

    return score


def train_torch_model(cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test, texts_test,
                      is_hf=False, seed=42):
    print("Train/dev set size", len(X_train))
    if X_val is not None:
        print("Val set size", len(X_val))

    hyperparameters = cfg.classifier.majority
    if is_hf is True:
        model = BertForSequenceClassification.from_pretrained(cfg.embedding.path, num_labels=3)
        fitter = HFFitter(cfg, X_train, Y_train, X_test=X_test, Y_test=Y_test,
                          texts_test=texts_test)
    else:
        weight_vector = compute_weight_vector(Y_train, use_torch=True)
        model = get_classifier(hyperparameters, cfg.classifier.name, cfg.embedding.n_embed, weight_vector
        )
        fitter = PLFitter(cfg, X_train, Y_train, X_test=X_test, Y_test=Y_test,
                          texts_test=texts_test)

    if cfg.classifier_mode.cv_folds:
        avg_score = fitter.cv_loop_and_eval(model)
    else:
        if X_val is None:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train,
                Y_train,
                test_size=cfg.run_mode.val_split,
                shuffle=True,
                stratify=Y_train,
                random_state=seed,
            )
        avg_score, _, _ = fitter.fit_and_eval(model, X_train, X_val, Y_train, Y_val)
    return avg_score


def train_sklearn(
    cfg, X_dev_emb, X_test_emb, Y_dev, Y_test, logger, texts_test, seed=None
):
    if cfg.dev_settings.annotation == "unanimous":
        hyperparameters = cfg.classifier.unanimous
    else:
        hyperparameters = cfg.classifier.majority
    model = get_classifier(cfg.embedding.path, hyperparameters, cfg.classifier.name,
                           cfg.embedding.n_embed)
    if cfg.classifier_mode.cv_folds:
        skf = StratifiedKFold(n_splits=cfg.classifier_mode.cv_folds)
        scores = []
        for train_index, val_index in skf.split(X_dev_emb, Y_dev):
            X_train = X_dev_emb[train_index]
            Y_train = Y_dev.to_numpy()[train_index]

            model.fit(X_train, Y_train)
            scores.append(
                evaluate_sklearn(
                    cfg.embedding.name,
                    cfg.classifier.name,
                    model,
                    X_test_emb,
                    Y_test,
                    texts_test,
                    cfg.run_mode.plot_path,
                )
            )
        score = np.mean(scores)
        print(
            f"--- Avg. accuracy across {cfg.classifier_mode.cv_folds} folds (cv-score) is: "
            f"{score}, SD={np.std(scores)}---"
        )
        if cfg.classifier_mode:
            timestamp = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
            out_path = hydra.utils.to_absolute_path(cfg.classifier_mode.out_path)
            output_path = os.path.join(
                out_path,
                cfg.classifier.name,
                build_experiment_name(cfg, f_ending=""),
                timestamp,
            )
            save_pretrained_sklearn(output_path, model, logger)
    else:
        model.fit(X_dev_emb, Y_dev)
        score = evaluate_sklearn(
            cfg.embedding.name,
            cfg.classifier.name,
            model,
            X_test_emb,
            Y_test,
            texts_test,
            cfg.run_mode.plot_path,
        )
        if cfg.classifier_mode:
            timestamp = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
            out_path = hydra.utils.to_absolute_path(cfg.classifier_mode.out_path)
            output_path = os.path.join(
                out_path,
                cfg.classifier.name,
                build_experiment_name(cfg, f_ending=""),
                timestamp,
            )
            save_pretrained_sklearn(output_path, model, logger)
    return score