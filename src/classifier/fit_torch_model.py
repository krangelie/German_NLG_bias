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


def cv_loop(cfg, model, X_train, Y_train, X_test, Y_test, texts_test, scoring=None, trial=None,
            path=None):
    hyperparameters = cfg.classifier.majority
    batch_size = hyperparameters.batch_size
    gpu_params = cfg.run_mode.gpu
    test_loader = get_dataloader(X_test, Y_test, batch_size, shuffle=False)
    skf = StratifiedKFold(n_splits=cfg.classifier_mode.cv_folds)
    accs, result_dicts, confs = [], [], []
    for train_index, val_index in skf.split(X_train, Y_train):
        print(f"Num train {len(train_index)}, num val {len(val_index)}")

        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = (
            Y_train[train_index],
            Y_train[val_index],
        )
        train_loader = get_dataloader(x_train, y_train, batch_size)
        val_loader = get_dataloader(x_val, y_val, batch_size, shuffle=False)
        model = fit(cfg, model, train_loader, val_loader, gpu_params, hyperparameters, scoring, trial, path)
        if path:
            output_path = path
        else:
            output_path = hydra.utils.to_absolute_path(cfg.run_mode.plot_path)
            output_path = os.path.join(output_path, cfg.classifier.name)
        mean_acc, results_dict, conf_matrix_npy = evaluate(
                model, test_loader, texts_test, set(Y_train),
            f"{cfg.embedding.name}_{cfg.classifier.name}", output_path
            )
        accs.append(mean_acc)
        result_dicts.append(results_dict)
        confs.append(conf_matrix_npy)
    aggregate_metrics(result_dicts, confs, output_path)
    print(
        f"--- Avg. accuracy across {cfg.classifier_mode.cv_folds} folds (cv-score) is: "
        f"{np.mean(accs)}, SD={np.std(accs)}---"
    )
    if cfg.classifier_mode.cv_folds == "incremental_train":
        return accs
    else:
        return np.mean(accs)


def fit(
    cfg,
    model,
    train_loader,
    val_loader,
    gpu_params,
    hyperparameters,
    scoring, trial, path
):

    callbacks = get_callbacks(cfg, hyperparameters, scoring, trial, path)
    trainer = get_trainer(callbacks, gpu_params, hyperparameters)

    mlflow.pytorch.autolog()
    # Train the model
    with mlflow.start_run():
        trainer.fit(model, train_loader, val_loader)

    print(trainer.current_epoch)

    return model


def get_trainer(callbacks, gpu_params, hyperparameters):
    if gpu_params.use_amp:
        trainer = pl.Trainer(
            precision=gpu_params.precision,
            amp_level=gpu_params.amp_level,
            amp_backend=gpu_params.amp_backend,
            gpus=gpu_params.n_gpus,
            max_epochs=hyperparameters.n_epochs,
            progress_bar_refresh_rate=20,
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpu_params.n_gpus,
            max_epochs=hyperparameters.n_epochs,
            progress_bar_refresh_rate=20,
            callbacks=callbacks,
        )
    return trainer


def get_callbacks(cfg, hyperparameters, scoring, trial, path):
    early_stopping = EarlyStopping(
        "val_loss", patience=hyperparameters.patience
    )  # change to val_loss

    callbacks = [early_stopping]

    if cfg.run_mode.store_after_training:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(
                cfg.classifier.model_path,
                build_experiment_name(cfg, f_ending=None),
                datetime.now().strftime("%b-%d-%Y-%H-%M-%S"),
            ),
            filename="{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=2,
            mode="min",
        )
        callbacks += [checkpoint_callback]
    elif cfg.classifier_mode.name == "tune":
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=path,
            filename="{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=0,
            mode="min",
        )
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=scoring)
        callbacks += [checkpoint_callback, pruning_callback]
    return callbacks

