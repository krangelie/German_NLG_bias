import numpy as np

from src.classifier.torch_helpers.pl_training import train_pl_model
from src.classifier.torch_helpers.hf_training import train_hf_model
from src.classifier.non_torch.non_torch_training import train_sklearn


def train_classifier(
    cfg, X_train, Y_train, X_val, Y_val, texts_val, X_test, Y_test, texts_test, logger, seed=42
):
    classes = set(Y_train)

    if cfg.classifier.name.startswith("lstm") or "sentence" in cfg.embedding.path:
        score = train_pl_model(cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test, texts_test,
                                  classes, seed)
    elif cfg.embedding.path.startswith("bert"):
        score = train_hf_model(cfg, X_train, Y_train, X_val, Y_val, X_test, Y_test, texts_test,
                                  classes, seed)
    else:
        score = train_sklearn(
            cfg, X_train, X_test, Y_train, Y_test, logger, texts_test
        )

    return score
