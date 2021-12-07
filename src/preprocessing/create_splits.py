import os
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold

from src.preprocessing.annotate_sentences import (
    label_with_aggregate_annotation,
)


def get_dev_test_indices(cfg, label_col, df):
    test_size = cfg.run_mode.test_split
    dest = os.path.join(cfg.run_mode.paths.dev_test_indcs, f"test_size_{test_size}")
    if not os.path.isdir(dest) or not os.listdir(dest):
        os.makedirs(dest, exist_ok=True)
        dev_indices, test_indices = train_test_split(
            df.index,
            test_size=test_size,
            shuffle=True,
            stratify=df[label_col],
            random_state=42,
        )
        dump_dev_test(dest, dev_indices, test_indices)
    else:
        dev_indices, test_indices = load_dev_test(dest)

    assert not _common_member(dev_indices, test_indices)
    indices_dict = {"dev_split": dev_indices, "test_split": test_indices}
    return indices_dict


def load_dev_test(dest):
    with open(os.path.join(dest, "dev_indices.pkl"), "rb") as d:
        dev_indices = pickle.load(d)
    with open(os.path.join(dest, "test_indices.pkl"), "rb") as t:
        test_indices = pickle.load(t)
    return dev_indices, test_indices


def dump_dev_test(dest, dev_indices, test_indices):
    with open(os.path.join(dest, "dev_indices.pkl"), "wb") as d:
        pickle.dump(dev_indices, d)
    with open(os.path.join(dest, "test_indices.pkl"), "wb") as t:
        pickle.dump(test_indices, t)


def _common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if a_set & b_set:
        return True
    else:
        return False