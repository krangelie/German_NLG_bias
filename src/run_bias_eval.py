import os.path
import sys

import hydra
from omegaconf import OmegaConf

from src.classifier.inference import predict
from src.evaluate_bias_in_nlg.eval_bias_in_labeled_generations import eval_bias
from src.evaluate_bias_in_nlg.qualitative_eval import eval_qual_bias


def run_bias_evaluation(cfg):
    orig_stdout = sys.stdout
    f = open(f"eval_bias_stdout.txt", "a")
    print("Redirecting stdout to 'outputs' folder.")
    sys.stdout = f
    predict(cfg)
    if cfg.run_mode.quant_eval:
        OmegaConf.set_struct(cfg.run_mode, False)  # allows overriding conf
        cfg.run_mode.input_path = os.path.join(cfg.classifier_mode.results_path, "transformer")
        eval_bias(cfg)
    if cfg.run_mode.qual_eval:
        eval_qual_bias(cfg)
    sys.stdout = orig_stdout
    f.close()

