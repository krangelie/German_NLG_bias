import xgboost
from sklearn.ensemble import RandomForestClassifier

from src.classifier.get_classifier_or_embedding import compute_weight_vector


def suggest_lstm(
    model_params,
    trial,
    Y,
    device,
    unit,
    n_embed,
    n_output,
):
    n_hidden = trial.suggest_categorical(
        model_params.n_hidden.name, model_params.n_hidden.choices
    )
    n_hidden_lin = trial.suggest_categorical(
        model_params.n_hidden_lin.name, model_params.n_hidden_lin.choices
    )
    n_layers = trial.suggest_int(
        model_params.n_layers.name,
        model_params.n_layers.lower,
        model_params.n_layers.upper,
    )
    batch_size = trial.suggest_categorical(
        model_params.batch_size.name, model_params.batch_size.choices
    )
    lr = trial.suggest_float(
        model_params.learning_rate.name,
        model_params.learning_rate.lower,
        model_params.learning_rate.upper,
        log=True,
    )
    bidir = trial.suggest_categorical(
        model_params.bidir.name, model_params.bidir.choices
    )
    if n_hidden_lin > 0:
        dropout = trial.suggest_float(
            model_params.dropout.name,
            model_params.dropout.lower,
            model_params.dropout.upper,
            step=model_params.dropout.step,
        )
    else:
        dropout = 0

    dropout_gru = trial.suggest_float(
        model_params.dropout_gru.name,
        model_params.dropout_gru.lower,
        model_params.dropout_gru.upper,
        step=model_params.dropout_gru.step,
    )

    weight_vector = compute_weight_vector(Y, device)

    hyperparameters = dict(
        n_embed=n_embed,
        n_hidden=n_hidden,
        n_hidden_lin=n_hidden_lin,
        n_output=n_output,
        n_layers=n_layers,
        lr=lr,
        weight_vector=weight_vector,
        bidirectional=bidir,
        gru=unit,
        drop_p=dropout,
        drop_p_gru=dropout_gru,
    )
    return hyperparameters, batch_size


def suggest_xgb(model_params, trial, xgb=None):
    n_estimators = trial.suggest_int(
        model_params.n_estimators.name,
        model_params.n_estimators.lower,
        model_params.n_estimators.upper,
        model_params.n_estimators.step,
    )
    lr = trial.suggest_float(
        model_params.learning_rate.name,
        model_params.learning_rate.lower,
        model_params.learning_rate.upper,
        log=True,
    )
    max_depth = trial.suggest_int(
        model_params.max_depth.name,
        model_params.max_depth.lower,
        model_params.max_depth.upper,
        model_params.max_depth.step,
    )

    classifier = xgboost.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        random_state=42,
        use_label_encoder=False,
        tree_method="gpu_hist",
        gpu_id=0,
    )
    return classifier


def suggest_rf(model_params, trial):
    n_estimators = trial.suggest_int(
        model_params.n_estimators.name,
        model_params.n_estimators.lower,
        model_params.n_estimators.upper,
        model_params.n_estimators.step,
    )
    max_depth = trial.suggest_int(
        model_params.max_depth.name,
        model_params.max_depth.lower,
        model_params.max_depth.upper,
        model_params.max_depth.step,
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    return classifier


def suggest_sbert(model_params, trial, Y, device, n_embed, n_output):
    n_hidden_lin = trial.suggest_categorical(
        model_params.n_hidden_lin.name, model_params.n_hidden_lin.choices
    )

    n_hidden_lin_2 = trial.suggest_categorical(
        model_params.n_hidden_lin_2.name, model_params.n_hidden_lin_2.choices
    )

    batch_size = trial.suggest_categorical(
        model_params.batch_size.name, model_params.batch_size.choices
    )
    lr = trial.suggest_float(
        model_params.learning_rate.name,
        model_params.learning_rate.lower,
        model_params.learning_rate.upper,
        log=True,
    )
    dropout = trial.suggest_float(
        model_params.dropout.name,
        model_params.dropout.lower,
        model_params.dropout.upper,
        step=model_params.dropout.step,
    )

    weight_vector = get_weight_vector(Y, device)

    hyperparameters = dict(
        n_embed=n_embed,
        n_hidden_lin=n_hidden_lin,
        n_hidden_lin_2=n_hidden_lin_2,
        n_output=n_output,
        lr=lr,
        weight_vector=weight_vector,
        drop_p=dropout,
    )

    # return batch size separately as this is given to dataloader not model
    return hyperparameters, batch_size