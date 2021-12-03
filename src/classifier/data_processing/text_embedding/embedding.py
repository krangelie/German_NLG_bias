import os

import hydra.utils
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer, pipeline
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import (
    load_facebook_vectors,
)


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
    #elif cfg.embedding.name == "transformer":
    #    if "sentence-transformer" in emb_path:
    #        embedding = SentenceTransformer(emb_path)

    else:
        raise SystemExit(f"{cfg.embedding.name} not implemented.")

    return embedding
