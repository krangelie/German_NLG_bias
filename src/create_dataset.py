import os
import pickle
import sys

import hydra.utils
import numpy as np
import pandas as pd

from src.classifier.data_processing.splitting.create_splits import get_data_splits
from src.classifier.utils import get_data_dir
from src.classifier.data_processing.data_augmentation.gendered_prompts import (
    replace_with_gendered_pronouns,
)
from src.classifier.visualizers.plots import plot_label_histogram, plt_labels_by_gender
from src.classifier.data_processing.annotate.annotate_sentences import (
    create_combined_df,
    clean_uncertain_labels,
    label_with_aggregate_annotation,
)
from src.classifier.data_processing.text_embedding.simple_tokenizer import (
    SimpleTokenizer,
)
from src.classifier.data_processing.text_embedding.vectorizer import (
    # TfidfWeights,
    MeanEmbeddingVectorizer,
    WordEmbeddingVectorizer,
)
from src.classifier.data_processing.text_embedding.embedding import get_embedding


def _store_data(data, dest_dir, file_name):
    dest_dir = hydra.utils.to_absolute_path(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    pickle.dump(data, open(os.path.join(dest_dir, file_name), "wb"))
    print(f"Saved {file_name} at {dest_dir}.")


def _common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if a_set & b_set:
        return True
    else:
        return False


def main(cfg):
    # get df with all annotations
    if cfg.language == "GER":
        preprocess_german_data(cfg)
    else:
        preprocess_english_data(cfg)

def preprocess_english_data(cfg):
    # refers to the predefined train and test data by Sheng et al.
    dcfg = cfg.run_mode

    train_df = pd.read_csv(hydra.utils.to_absolute_path(dcfg.paths.train_set_path))
    val_df = pd.read_csv(hydra.utils.to_absolute_path(dcfg.paths.val_set_path))
    test_df = pd.read_csv(hydra.utils.to_absolute_path(dcfg.paths.test_set_path))

    input_col = cfg.token_type if cfg.embedding.name != "transformer" else cfg.text_col
    model = get_embedding(cfg)

    if cfg.embedding.name != "transformer":
        sys.exit("Aborting - EN support so far only for transformer architecture implemented")
        # TODO: adjust for EN
        # sgt = SimpleTokenizer(
        #     "english",
        #     dcfg.tokenize.to_lower,
        #     dcfg.tokenize.remove_punctuation,
        # )
        # path_to_tfidf = hydra.utils.to_absolute_path(dcfg.paths.tfidf_weights)
        #
        # tfidf_weights = np.load(
        #     os.path.join(path_to_tfidf, "word2weight_idf.npy"), allow_pickle=True
        # ).item()
        # max_idf = np.load(
        #     os.path.join(path_to_tfidf, "max_idf.npy"),
        #     allow_pickle=True,
        # )
        # assert isinstance(tfidf_weights, dict)
        # if cfg.pre_processing.mean:
        #     vectorizer = MeanEmbeddingVectorizer(
        #         model, tfidf_weights, max_idf=max_idf
        #     )
        # else:
        #     vectorizer = WordEmbeddingVectorizer(
        #         model,
        #         tfidf_weights,
        #         max_idf=max_idf,
        #         seq_length=cfg.pre_processing.seq_length,
        #     )
    else:
        sgt = None
        vectorizer = None

    def _vectorize_split(split_df, split_name):
        print(split_name)
        Y_split = split_df[cfg.label_col]
        if sgt:
            # tokenize
            split_df = sgt.tokenize(split_df, text_col=cfg.text_col)
        if dcfg.augment and split_name == "train_split":
            split_df = replace_with_gendered_pronouns(dcfg.augment, cfg.text_col, split_df, "EN")
            plt_labels_by_gender(
                dcfg.annotation,
                dcfg.paths.plot_path,
                split_df,
                Y_split,
                name=split_name,
            )
        texts = split_df[cfg.text_col]
        if cfg.embedding.name != "transformer":
            X_split = vectorizer.transform(split_df[input_col])
        else:
            X_split = model.encode(split_df[input_col].tolist())

        if -1 in Y_split.tolist():
            Y_split += 1

        print(X_split[:10], Y_split[:10], texts[:10])
        _store_data(
            {"X": X_split, "Y": Y_split, "texts": texts},
            get_data_dir(cfg),
            split_name,
        )
        return split_df

    split_names = ["train_split", "val_split", "test_split"]
    for i, split_df in enumerate([train_df, val_df, test_df]):
        _vectorize_split(split_df, split_names[i])




def preprocess_german_data(cfg):
    dcfg = cfg.run_mode
    df, annotator_names = create_combined_df()
    # init tokenizer
    if cfg.embedding.name != "transformer":
        sgt = SimpleTokenizer(
            "german",
            dcfg.tokenize.to_lower,
            dcfg.tokenize.remove_punctuation,
        )
        # tokenize
        df = sgt.tokenize(df, text_col=cfg.text_col)
    # get stored split indices (independent of pre-processing steps)
    if dcfg.augment:
        df = replace_with_gendered_pronouns(dcfg.augment, cfg.text_col, df, "GER")
    # if k_fold: returns lists of splits
    dev_set, test_set = get_data_splits(dcfg, cfg.label_col, df, annotator_names)
    if not dcfg.k_fold:
        preprocess_and_store_splits(cfg, dev_set, test_set, annotator_names)
    else:
        if cfg.specific_fold == -1:
            for fold in range(dcfg.k_fold):
                preprocess_and_store_splits(
                    cfg, dev_set[fold], test_set[fold], annotator_names, fold=fold
                )
        else:
            fold = cfg.specific_fold
            preprocess_and_store_splits(
                cfg, dev_set[fold], test_set[fold], annotator_names, fold=fold
            )


def preprocess_and_store_splits(cfg, dev_set, test_set, annotator_names, fold=None):
    dcfg = cfg.run_mode
    # make sure dev and test split are mutually exclusive
    assert not _common_member(dev_set.index.tolist(), test_set.index.tolist())
    split_names = ["dev_split", "test_split"]

    # fit TFIDF on dev-set to get IDF-weights
    if cfg.embedding.name != "transformer":
        input_col = cfg.token_type
        path_to_tfidf = hydra.utils.to_absolute_path(dcfg.paths.tfidf_weights)
        tfidf_weights = np.load(
            os.path.join(path_to_tfidf, "word2weight_idf.npy"), allow_pickle=True
        ).item()
        max_idf = np.load(
            os.path.join(path_to_tfidf, "max_idf.npy"),
            allow_pickle=True,
        )
        assert isinstance(tfidf_weights, dict)
    else:
        input_col = cfg.text_col
    # load embedding dictionary or model in case of sentence-transformer
    model = get_embedding(cfg)
    # The following steps are done after splitting to ensure that the same
    # indices are used for dev and test set irrespective of the cleaning
    # procedure
    for i, split in enumerate([dev_set, test_set]):

        # clean out cases where annotators were uncertain
        split = clean_uncertain_labels(
            cfg.pre_processing.remove_uncertain, split, annotator_names
        )

        # annotate
        split = label_with_aggregate_annotation(
            dcfg.annotation,
            cfg.label_col,
            split,
            annotator_names,
        )
        Y_split = split[cfg.label_col].tolist()
        if not dcfg.augment:
            plot_label_histogram(
                Y_split,
                name=f"{split_names[i]} balanced on majority, " f"after cleaning",
            )
        else:
            plt_labels_by_gender(
                dcfg.annotation,
                dcfg.paths.plot_path,
                split,
                Y_split,
                name=split_names[i],
            )
        if -1 in Y_split:
            Y_split += 1  # avoid negative labels
        X_split = split[input_col]
        texts = split[cfg.text_col]

        if cfg.embedding.name != "transformer":
            # vectorize (for transformer, embedding will be applied later)
            if cfg.pre_processing.mean:
                vectorizer = MeanEmbeddingVectorizer(
                    model, tfidf_weights, max_idf=max_idf
                )
            else:
                vectorizer = WordEmbeddingVectorizer(
                    model,
                    tfidf_weights,
                    max_idf=max_idf,
                    seq_length=cfg.pre_processing.seq_length,
                )
            X_split = vectorizer.transform(X_split)
        else:
            # get sentence embeddings
            X_split = model.encode(X_split.tolist())

        # store
        if fold is None:
            _store_data(
                {"X": X_split, "Y": Y_split, "texts": texts},
                get_data_dir(cfg),
                split_names[i],
            )
        else:
            dest = os.path.join(get_data_dir(cfg), f"fold_{fold}")
            os.makedirs(dest, exist_ok=True)
            _store_data(
                {"X": X_split, "Y": Y_split, "texts": texts},
                dest,
                split_names[i],
            )
