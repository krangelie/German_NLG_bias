import os
from datetime import datetime

import hydra.utils
import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
import torch

from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import BertForSequenceClassification

from src.classifier.torch_helpers.eval_torch import evaluate
from src.classifier.visualizers.plots import aggregate_metrics
from src.classifier.torch_helpers.torch_data import RegardBertDataset
from src.classifier.utils import build_experiment_name

# Train via huggingface API
def train_hf_model(cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test, texts_test, classes,
                  seed=42):
    print("Train/dev set size", len(X_train))
    if X_val is not None:
        print("Val set size", len(X_val))
    output_path = hydra.utils.to_absolute_path(cfg.run_mode.plot_path)
    output_path = os.path.join(output_path, cfg.classifier.name)
    if cfg.dev_settings.annotation == "unanimous":
        hyperparameters = cfg.classifier.unanimous
    else:
        hyperparameters = cfg.classifier.majority

    model = BertForSequenceClassification.from_pretrained(cfg.embedding.path, num_labels=3)
    test_dataset = RegardBertDataset(X_test, Y_test)
    callbacks = get_callbacks(cfg, hyperparameters)

    if cfg.classifier_mode.cv_folds:
        skf = StratifiedKFold(n_splits=cfg.classifier_mode.cv_folds)
        accs, result_dicts, confs = [], [], []
        for train_index, val_index in skf.split(X_train, Y_train):
            print(f"Num train {len(train_index)}, num val {len(val_index)}")

            X_train, X_val = X_train[train_index], X_train[val_index]
            Y_train, Y_val = (
                Y_train.to_numpy()[train_index],
                Y_train.to_numpy()[val_index],
            )
            train_dataset = RegardBertDataset(X_train, Y_train)
            val_dataset = RegardBertDataset(X_val, Y_val)
            trainer = get_trainer(hyperparameters, model, train_dataset, val_dataset, callbacks, output_path)
            # Train the model
            trainer.train()
            torch.cuda.empty_cache()
            logits, labels, results_dict = trainer.predict(test_dataset)
            conf_matrix_npy, acc_per_class = get_conf_matrix(classes, logits, labels)
            results_dict["acc_per_class"] = acc_per_class
            accs.append(results_dict["accuracy"])
            result_dicts.append(results_dict)
            confs.append(conf_matrix_npy)
        aggregate_metrics(result_dicts, confs, output_path)
        if cfg.classifier_mode.cv_folds == "incremental_train":
            return accs
        else:
            return np.mean(accs)
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

        train_dataset = RegardBertDataset(X_train, Y_train)
        val_dataset = RegardBertDataset(X_val, Y_val)
        trainer = get_trainer(hyperparameters, model, train_dataset, val_dataset, callbacks,
                              output_path)
        # Train the model
        trainer.train()
        torch.cuda.empty_cache()
        logits, labels, results_dict = trainer.predict(test_dataset)
        _, _ = get_conf_matrix(classes, logits, labels)
        return results_dict["accuracy"]


def get_callbacks(cfg, hyperparameters):
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=hyperparameters.patience
    )  # change to val_loss
    callbacks = [early_stopping]
    # if cfg.run_mode.store_after_training:
    #     checkpoint_callback = ModelCheckpoint(
    #         monitor="val_loss",
    #         dirpath=os.path.join(
    #             cfg.classifier.model_path,
    #             build_experiment_name(cfg, f_ending=None),
    #             datetime.now().strftime("%b-%d-%Y-%H-%M-%S"),
    #         ),
    #         filename="{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}",
    #         save_top_k=2,
    #         mode="min",
    #     )
    #     callbacks += [checkpoint_callback]
    return callbacks


def get_trainer(hyperparameters, model, train_dataset, val_dataset, callbacks, output_path):

    args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=hyperparameters.batch_size,
        per_device_eval_batch_size=hyperparameters.batch_size,
        num_train_epochs=hyperparameters.n_epochs,
        seed=0,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    return trainer


def compute_metrics(p):
    logits, labels = p
    pred = np.argmax(logits, axis=1)
    average = "macro"
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average=average)
    precision = precision_score(y_true=labels, y_pred=pred, average=average)
    f1 = f1_score(y_true=labels, y_pred=pred, average=average)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def get_conf_matrix(classes, logits, labels):
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes))
    num_per_class = {c: 0 for c in classes}
    preds = np.argmax(logits, axis=1)
    for t, p in zip(labels, preds):
        num_per_class[int(p.item())] += 1
        confusion_matrix[t, p] += 1
    print(f"Confusion matrix: {confusion_matrix}")
    acc_list = confusion_matrix.diag() / confusion_matrix.sum(1)
    acc_per_class = dict(zip(classes, acc_list))
    print(f"Accuracy per class:", acc_per_class)
    return confusion_matrix, acc_per_class