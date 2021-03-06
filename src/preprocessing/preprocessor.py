import os
import pickle
from abc import ABC, abstractmethod

import hydra
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.classifier.utils import get_data_dir
from src.preprocessing.split_indexing import get_dev_test_sets
from src.classifier.get_classifier_or_embedding import get_embedding
from src.preprocessing.gendered_prompts import \
    replace_with_gendered_pronouns
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.vectorizer import MeanEmbeddingVectorizer, WordEmbeddingVectorizer


def format_y(y):
    if -1 in set(y):
        y += 1
    y = np.array(y).astype(int)
    return y


class Preprocessor(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def preprocess_and_store(self):
        pass

    def load_dataframe(self):
        df = pd.read_csv(hydra.utils.to_absolute_path(self.cfg.run_mode.paths.raw_data))
        return df

    def get_x_y_texts(self, df):
        x = df[self.cfg.pre_processing.token_type]
        y = format_y(df[self.cfg.label_col])
        texts = df[self.cfg.text_col]
        return x, y, texts

    def store_data(self, data, dest_dir, file_name):
        dest_dir = hydra.utils.to_absolute_path(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        pickle.dump(data, open(os.path.join(dest_dir, file_name), "wb"))
        print(f"Saved {file_name} at {dest_dir}.")


class FastTextPreprocessor(Preprocessor):
    def __init__(self):
        path_to_tfidf = hydra.utils.to_absolute_path(self.cfg.run_mode.paths.tfidf_weights)

        self.tfidf_weights = np.load(
            os.path.join(path_to_tfidf, "word2weight_idf.npy"), allow_pickle=True
        ).item()
        assert isinstance(self.tfidf_weights, dict)

        self.max_idf = np.load(
            os.path.join(path_to_tfidf, "max_idf.npy"),
            allow_pickle=True,
        )

    def preprocess_and_store(self):
        df = self.load_dataframe()
        if self.cfg.run_mode.augment:
            df = replace_with_gendered_pronouns(self.cfg.run_mode.augment, self.cfg.text_col, df)
        df = self.basic_tokenize(df)
        model = get_embedding(self.cfg)
        vectorizer = self.get_vectorizer(model)

        dev_set, test_set = get_dev_test_sets(self.cfg, self.cfg.label_col, df)
        for split_df, split_name in zip([dev_set, test_set], ["dev_split", "test_split"]):
            x, y, texts = self.get_x_y_texts(split_df)
            x = vectorizer.transform(x)
            self.store_data(
                {"X": x, "Y": y, "texts": texts},
                get_data_dir(self.cfg),
                split_name,
            )

    def basic_tokenize(self, df):
        sgt = SimpleTokenizer(
            ("german" if self.cfg.language == "GER" else "english"),
            self.cfg.run_mode.tokenize.to_lower,
            self.cfg.run_mode.tokenize.remove_punctuation,
        )
        # tokenize
        df = sgt.tokenize(df, text_col=self.cfg.text_col)
        return df

    def get_vectorizer(self, model):
        if self.cfg.pre_processing.mean:
            vectorizer = MeanEmbeddingVectorizer(
                model, self.tfidf_weights, max_idf=self.max_idf
            )
        else:
            vectorizer = WordEmbeddingVectorizer(
                model,
                self.tfidf_weights,
                max_idf=self.max_idf,
                seq_length=self.cfg.pre_processing.seq_length,
            )
        return vectorizer


class SBertPreprocessor(Preprocessor):
    def preprocess_and_store(self):
        df = self.load_dataframe()
        if self.cfg.run_mode.augment:
            df = replace_with_gendered_pronouns(self.cfg.run_mode.augment, self.cfg.text_col, df)
        model = get_embedding(self.cfg)

        dev_set, test_set = get_dev_test_sets(self.cfg, self.cfg.label_col, df)
        for split_df, split_name in zip([dev_set, test_set], ["dev_split", "test_split"]):
            x, y, texts = self.get_x_y_texts(split_df)
            x = model.encode(x.tolist())
            self.store_data(
                {"X": x, "Y": y, "texts": texts},
                get_data_dir(self.cfg),
                split_name,
            )


class ShengPreprocessor(Preprocessor):
    def preprocess_and_store(self):
        train_df = pd.read_csv(hydra.utils.to_absolute_path(self.cfg.run_mode.paths.train_set_path))
        val_df = pd.read_csv(hydra.utils.to_absolute_path(self.cfg.run_mode.paths.val_set_path))
        test_df = pd.read_csv(hydra.utils.to_absolute_path(self.cfg.run_mode.paths.test_set_path))
        for split_df, split_name in zip([train_df, val_df, test_df], ["train_split", "val_split",
                                                               "test_split"]):
            self.preprocess_split(split_df, split_name)

    def preprocess_split(self, split_df, split_name):
        if self.cfg.run_mode.augment and split_name == "train_split":
            split_df = replace_with_gendered_pronouns(self.cfg.run_mode.augment, self.cfg.text_col,
                                                      split_df)
        x, y, texts = self.get_x_y_texts(split_df)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.embedding.path)
        x = tokenizer(x.tolist(), padding="max_length", truncation=True)
        print(x[:3], y[:3], texts[:3])
        self.store_data(
            {"X": x, "Y": y, "texts": texts},
            get_data_dir(self.cfg),
            split_name,
        )
