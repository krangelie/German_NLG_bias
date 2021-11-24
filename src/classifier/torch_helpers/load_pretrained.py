import hydra
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from src.classifier.lstm.lstm_classifier import RegardLSTM
from src.classifier.sent_transformer.sbert_classifier import RegardBERT

def load_torch_model(model_path, model_type, logger=None):
    model_path = hydra.utils.to_absolute_path(model_path)
    if logger is not None:
        logger.info(f"Loading pretrained torch model from {model_path}")
    if "bert_regard_v2_large" in model_path:
        print(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        return (model, tokenizer)
    elif model_path.endswith("pth"):
        model = torch.load(model_path, map_location=torch.device('cpu'))
    elif model_path.endswith("ckpt"):
        if model_type == "lstm":
            model = RegardLSTM.load_from_checkpoint(model_path)
        elif model_type == "transformer":
            model = RegardBERT.load_from_checkpoint(model_path)
    return model
