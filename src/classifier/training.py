from src.classifier.torch_helpers.torch_training import train_torch_model
from src.classifier.non_torch.non_torch_training import train_sklearn


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