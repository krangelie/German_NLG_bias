import hydra.utils
import numpy as np
import torch
import xgboost as xgb
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

from src.classifier.classifiers import RegardLSTM, RegardBERT


def get_classifier(model_params, model_type, n_embed, weight_vector=None):
    if model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=model_params.n_estimators,
            max_depth=model_params.max_depth,
            random_state=42,
        )
    elif model_type == "xgb":
        classifier = xgb.XGBClassifier(
            n_estimators=model_params.n_estimators,
            learning_rate=model_params.learning_rate,
            max_depth=model_params.max_depth,
            random_state=42,
        )
    elif model_type == "lstm":
        classifier = RegardLSTM(
            n_embed=n_embed,
            n_hidden=model_params.n_hidden,
            n_hidden_lin=model_params.n_hidden_lin,
            n_output=model_params.n_output,
            n_layers=model_params.n_layers,
            lr=model_params.lr,
            weight_vector=weight_vector,
            bidirectional=model_params.bidir,
            gru=model_params.unit,
            drop_p=model_params.dropout,
            drop_p_gru=model_params.dropout_gru,
        )
    elif model_type == "transformer":
        classifier = RegardBERT(n_embed=n_embed,
                                n_hidden_lin=model_params.n_hidden_lin,
                                n_hidden_lin_2=model_params.n_hidden_lin_2,
                                n_output=model_params.n_output, lr=model_params.lr,
                                weight_vector=weight_vector, drop_p=model_params.dropout)
    else:
        print(
            "Please choose a classifier type that is implemented.\
             So far only rf for RandomForest, xgb for XGBoost, or lstm."
        )

    return classifier


def compute_weight_vector(Y, use_torch=True):
    weight_vector = len(Y) / (len(set(Y)) * np.bincount(Y))
    if use_torch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_vector = torch.FloatTensor(weight_vector).to(device)
    return weight_vector


def get_embedding(cfg):
    if cfg.embedding.name != "transformer":
        emb_path = hydra.utils.to_absolute_path(cfg.embedding.path)
    else:
        emb_path = cfg.embedding.path

    if cfg.embedding.name == "w2v":
        embedding = KeyedVectors.load_word2vec_format(
            emb_path, binary=False, no_header=cfg.embedding.no_header
        )

    elif cfg.embedding.name == "fastt":
        embedding = load_facebook_vectors(emb_path)
    elif cfg.embedding.name == "transformer":
        if "sentence" in emb_path:
            embedding = SentenceTransformer(emb_path)

    else:
        raise SystemExit(f"{cfg.embedding.name} not implemented.")

    return embedding