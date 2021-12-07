from src.preprocessing.preprocessor import SBertPreprocessor, FastTextPreprocessor, \
    ShengPreprocessor


def main(cfg):
    # get df with all annotations
    if cfg.language == "GER":
        if cfg.embedding.name == "transformer":
            preprocessor = SBertPreprocessor(cfg)
        else:
            preprocessor = FastTextPreprocessor(cfg)
    else:
        preprocessor = ShengPreprocessor(cfg)

    preprocessor.preprocess_and_store()
