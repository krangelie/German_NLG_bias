import os
import pickle
from abc import ABC, abstractmethod

import hydra
import numpy as np

from src.classifier.utils import get_data_dir
from src.preprocessing import annotate_sentences
from src.preprocessing.annotate_sentences import clean_uncertain_labels, \
    label_with_aggregate_annotation
from src.preprocessing.create_splits import get_dev_test_indices


def format_y(y):
    if -1 in y:
        y += 1
    y = np.array(y)
    return y


class Preprocessor(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def preprocess_and_store(self):
        pass

    def load_dataframe(self):
        df, annotator_names = annotate_sentences.create_combined_df(
            self.cfg.run_mode.paths.raw_data)
        return df, annotator_names

    def annotate(self, df, annotator_names):
        df = clean_uncertain_labels(
            self.cfg.pre_processing.remove_uncertain, df, annotator_names
        )
        df = label_with_aggregate_annotation(
            self.cfg.run_mode.annotation,
            self.cfg.label_col,
            df,
            annotator_names,
        )
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

    def store_by_split(self, indices_dict, x, y, texts):
        for split_name, idx_list in indices_dict.items():
            self.store_data(
                {"X": x[idx_list], "Y": y[idx_list], "texts": [texts[i] for i in idx_list]},
                get_data_dir(self.cfg),
                split_name,
            )