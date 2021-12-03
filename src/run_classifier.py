import sys

import numpy as np
import mlflow
import hydra

from src.classifier.classifier_training.training import train_classifier
from src.classifier.classifier_evaluation.eval_on_testset import evaluate_on_test_set
from src.regard_prediction.inference import predict
from src.classifier.utils import get_data
from src.classifier.classifier_tuning.tuning import Tuner
from src.classifier.classifier_training.incremental_training import train_on_increments


def run(cfg, rootLogger):
    mlruns_folder = hydra.utils.to_absolute_path("mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_folder}")
    mode = cfg.classifier_mode.name
    print("Redirecting stdout to 'outputs' folder.")
    print("When training with k-fold cv: trained models will be stored in 'outputs', "
          "in the mlflow artifacts folders.")
    orig_stdout = sys.stdout
    f = open(f"{mode}_stdout.txt", "a")
    sys.stdout = f
    print("Classifier mode", mode)
    if mode == "predict":
        predict(cfg, logger=rootLogger)

    elif mode == "eval":
        splits_dict = get_data(cfg)
        evaluate_on_test_set(
            cfg,
            splits_dict["X_test"],
            splits_dict["Y_test"],
            splits_dict["texts_test"],
        )
    else:
        splits_dict = get_data(cfg)
        if cfg.k_fold:
            if cfg.specific_fold != -1:
                run_on_split_set(
                    cfg,
                    mode,
                    rootLogger,
                    splits_dict[f"fold_{cfg.specific_fold}"],
                    cfg.specific_fold,
                )
            else:
                scores = []
                for fold in range(cfg.k_fold):
                    print("\nRUN FOLD", fold, "\n")
                    scores.append(
                        run_on_split_set(
                            cfg, mode, rootLogger, splits_dict[f"fold_{fold}"], fold
                        )
                    )
                print("\nSCORE ACROSS FOLDS", np.mean(scores), "\n")
        else:
            run_on_split_set(cfg, mode, rootLogger, splits_dict)

    sys.stdout = orig_stdout
    f.close()


def run_on_split_set(cfg, mode, rootLogger, splits_dict, fold=None):
    if cfg.language == "GER":
        # X_train is a complete dev set for applying cross-val
        X_train, Y_train, X_test, Y_test, texts_test = (
            splits_dict["X_dev"],
            splits_dict["Y_dev"],
            splits_dict["X_test"],
            splits_dict["Y_test"],
            splits_dict["texts_test"],
        )
        X_val, Y_val, texts_val = None, None, None
    else:
        X_train, Y_train, X_val, Y_val, texts_val, X_test, Y_test, texts_test = (
            splits_dict["X_train"],
            splits_dict["Y_train"],
            splits_dict["X_val"],
            splits_dict["Y_val"],
            splits_dict["texts_val"],
            splits_dict["X_test"],
            splits_dict["Y_test"],
            splits_dict["texts_test"],
        )
    print(splits_dict["texts_test"])

    if mode == "tune":
        score = None
        tuner = Tuner(cfg, X_train, Y_train, fold=fold)
        tuner.find_best_params()

    elif mode == "train":
        score = train_classifier(
            cfg, X_train, Y_train, X_val, Y_val, texts_val, X_test, Y_test, texts_test, rootLogger
        )

    elif mode == "incremental_train":
        score = None
        # Analyze effect of dev-set size on training
        train_on_increments(cfg, X_train, Y_train, X_test, Y_test, texts_test, rootLogger)
    return score
