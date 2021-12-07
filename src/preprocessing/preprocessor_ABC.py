import os
import pickle
from abc import ABC, abstractmethod

import hydra
import numpy as np
import pandas as pd

from src.preprocessing.annotate_sentences import clean_uncertain_labels, \
    label_with_aggregate_annotation


def format_y(y):
    if -1 in set(y):
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

