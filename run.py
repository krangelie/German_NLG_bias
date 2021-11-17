import logging

import hydra
from omegaconf import DictConfig

from src.constants_GER import Constants_GER
from src.constants_EN import Constants_EN
from src import constants

rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler()
rootLogger.addHandler(consoleHandler)
# rootLogger = None

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    mode = cfg.run_mode.name
    print("Language = ", cfg.language)
    # set language-specific constants
    if cfg.language == "GER":
        constants.constants = Constants_GER()
    elif cfg.language == "EN":
        constants.constants = Constants_EN()

    print("Run mode", mode)
    if mode == "data":
        from src import create_dataset
        create_dataset.main(cfg)
    elif mode == "classifier":
        from src import run_classifier
        run_classifier.run(cfg, rootLogger)
    elif mode == "generate":
        from src.text_generator.run_text_generation import run_txt_generation
        run_txt_generation(cfg)
    elif mode == "trigger":
        from src.bias_mitigator import run_bias_mitigation
        run_bias_mitigation.run(cfg)
    elif mode == "eval_bias":
        from src.evaluate_bias_in_nlg.run_bias_eval import run_bias_evaluation
        run_bias_evaluation(cfg)
    elif mode == "naive_trigger":
        from src.adjective_based_mitigation.sample_and_eval_adjectives import (
            find_best_adjective,
        )
        find_best_adjective(cfg)
    else:
        print("Run mode not implemented. Typo?")


if __name__ == "__main__":
    run()
