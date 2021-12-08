from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification

from src.classifier.fit_torch_model import PLFitter, HFFitter
from src.classifier.get_classifier_or_embedding import get_classifier, compute_weight_vector

# Custom PyTorch Lightning training
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

