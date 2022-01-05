import os
from pprint import pprint
import json

import hydra.utils
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from germansentiment import SentimentModel


from src.preprocessing.simple_tokenizer import (
    SimpleTokenizer,
)
from src.classifier.get_classifier_or_embedding import get_embedding, get_pretrained_classifier
from src.preprocessing.vectorizer import (
    MeanEmbeddingVectorizer,
    WordEmbeddingVectorizer,
)
from src.dicts_and_contants.constants import constants


def predict(cfg,
            eval_model=None,
            eval_model_type=None,
            embedding_path=None,
            sample_file=None,
            eval_dest=None,
            regard_only=True
            ):
    print(cfg.classifier_mode)
    model_type = eval_model_type if eval_model_type else cfg.classifier.name
    by_class_results = not eval_model and cfg.classifier_mode.split_by_class
    use_sklearn_model = not eval_model and model_type not in [
        "lstm",
        "transformer",
    ]
    text_path = cfg.classifier_mode.inference_texts if not sample_file else sample_file
    results_path = hydra.utils.to_absolute_path(cfg.classifier_mode.results_path)
    output_path = (
        os.path.join(
            results_path,
            model_type,
        )
        if not eval_dest
        else eval_dest
    )
    os.makedirs(output_path, exist_ok=True)

    model, tokenizer = get_pretrained_classifier(cfg, eval_model, model_type)
    print(model, tokenizer)
    
    if any([text_path.endswith(ending) for ending in [".csv", ".txt"]]):
        predictor = Predictor(cfg, model, model_type, use_sklearn_model=use_sklearn_model,
                              by_class_results=by_class_results,
                              input_path=text_path, output_path=output_path,
                              embedding_path=embedding_path,
                              tokenizer=tokenizer)
        predictor.classify_all_sentences(regard_only)
    else:
        text_path = hydra.utils.to_absolute_path(text_path)
        for file in os.listdir(text_path):
            path = os.path.join(text_path, file)
            if not os.path.isdir(path):
                print(f"Processing {path}")
                predictor = Predictor(cfg, model, model_type, use_sklearn_model=use_sklearn_model,
                                      by_class_results=by_class_results,
                                      input_path=path, output_path=output_path,
                                      embedding_path=embedding_path,
                                      tokenizer=tokenizer)
                predictor.classify_all_sentences(regard_only)


def flip_gender(texts, f_to_m):
    female_to_male = constants.F_TO_M_PRONOUNS
    flipped_texts = []
    for txt in texts:
        flipped_txt = []
        dictionary = female_to_male if f_to_m else female_to_male.inverse
        for word in txt.split(" "):
            if word in dictionary.keys():
                word = dictionary[word]
            flipped_txt += [word]
        flipped_texts += [" ".join(flipped_txt)]
    return flipped_texts


def add_demographics(texts, placeholder, demographics_list):
    demo_added = {}

    for demo in demographics_list:
        demo_prefix = constants.VARIABLE_DICT[demo]
        demo_added[demo] = [txt.replace(placeholder, demo_prefix) for txt in texts]
        demo_added[demo] = flip_gender(
            demo_added[demo],
            any(demo_prefix == female for female in constants.FEMALE_PREFIXES),
        )
    print(demo_added)
    return demo_added


def remove_demographic(cfg, text):
    for demo in cfg.classifier_mode.demographics:
        text = text.replace(demo, "").strip()
    return text


class Predictor:
    def __init__(self, cfg, model, model_type, use_sklearn_model, by_class_results, input_path, \
                                       output_path, embedding_path=None, tokenizer=None):
        self.cfg = cfg
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.embedding_path = embedding_path
        self.use_sklearn_model = use_sklearn_model
        self.by_class_results = by_class_results
        self.input_path, self.output_path = input_path, output_path
        self.sent_dict = self.load_and_prepare_inference_data()

    def load_and_prepare_inference_data(self):
        embed = not bool(self.tokenizer)
        print(self.input_path)
        if self.input_path.endswith(".txt"):
            with open(self.input_path) as f:
                lines = [line.rstrip() for line in f]
            sentence_df = pd.DataFrame(lines, columns=[self.cfg.text_col])
        else:
            sentence_df = pd.read_csv(self.input_path)
        sentence_df = sentence_df.dropna(subset=[self.cfg.text_col])
        # Add different demographic mentions to sentences
        if self.cfg.classifier_mode.add_demographic_terms:
            return self.get_by_demo_sentence_dict(embed, sentence_df)
        else:
            # Add just one demographic's mentions to sentences
            return self.get_single_demo_sentence_dict(embed, sentence_df)

    def get_single_demo_sentence_dict(self, embed, sentence_df):
        if self.cfg.classifier_mode.add_first_demo:
            sentence_df[self.cfg.text_col] = sentence_df[self.cfg.text_col].apply(
                lambda txt: constants.VARIABLE_DICT[self.cfg.classifier_mode.demographics[0]]
                            + " "
                            + txt
            )
        if embed:
            sentence_df, sentences_emb = self.embed_texts(sentence_df)
        else:
            sentences_emb = sentence_df[self.cfg.text_col].to_list()
        text_embs = {"text_df": sentence_df, "text_emb": sentences_emb}
        return text_embs

    def get_by_demo_sentence_dict(self, embed, sentence_df):
        print("Add demographics")
        # returns dict {gender : gendered text}
        demographic_texts = add_demographics(
            sentence_df[self.cfg.text_col],
            constants.PERSON,
            self.cfg.classifier_mode.demographics,
        )
        # Embed/tokenize sentences
        gendered_text_embs = {}
        for gen, texts in demographic_texts.items():
            sentence_df = pd.DataFrame(texts, columns=[self.cfg.text_col])
            if embed:
                sentence_df, sentences_emb = self.embed_texts(sentence_df)
            else:
                if not isinstance(texts, list):
                    sentences_emb = texts.to_list()
                else:
                    sentences_emb = texts
            gendered_text_embs[gen] = {
                "text_df": sentence_df,
                "text_emb": sentences_emb,
            }
        return gendered_text_embs

    def embed_texts(self, sentence_df):
        def _vectorize(sentences, tfidf_weights=None):
            model = (
                get_embedding(self.cfg)
                if not self.embedding_path
                else SentenceTransformer(self.embedding_path)
            )

            if self.model_type != "transformer":
                if self.model_type != "lstm":
                    vectorizer = MeanEmbeddingVectorizer(model, tfidf_weights)
                else:
                    vectorizer = WordEmbeddingVectorizer(
                        model,
                        tfidf_weights,
                    )
                return vectorizer.transform(sentences)
            else:
                return model.encode(sentences)

        if self.model_type != "transformer":
            sgt = SimpleTokenizer(
                True,
                True,
                False,
                False,
            )
            sentence_df = sgt.tokenize(sentence_df, text_col=self.cfg.text_col)
            sentences_emb = _vectorize(sentence_df[self.cfg.token_type])
        else:
            sentences_emb = _vectorize(sentence_df[self.cfg.text_col])
        return sentence_df, sentences_emb

    def classify_all_sentences(self, regard_only):
        if self.cfg.classifier_mode.add_demographic_terms:
            for gen, gen_dict in self.sent_dict.items():
                print("Processing texts for ", gen)
                dest = os.path.join(self.output_path, f"{gen}_texts_regard_labeled.csv")
                sentence_df = self.classify_dict_of_sentences(gen_dict, regard_only)
                sentence_df.to_csv(dest)
                torch.cuda.empty_cache()
        else:
            dest = os.path.join(
                self.output_path,
                f"{os.path.basename(self.input_path).split('.')[0]}_regard_labeled.csv",
            )
            sentence_df = self.classify_dict_of_sentences(self.sent_dict, regard_only)
            sentence_df.to_csv(dest)
        print("Predictions stored at", dest)
        # if self.by_class_results:
        #     store_preds_per_class(
        #         cfg.classifier_mode, output_path, sentence_df["regard"], sentence_df, cfg.text_col
        #     )

    def classify_dict_of_sentences(self, input_dict, regard_only):
        sentence_df, sentences_emb = input_dict["text_df"], input_dict["text_emb"]
        sentence_df = self.predict_regard(sentence_df, sentences_emb)
        if not regard_only:
            if self.cfg.classifier_mode.sentiment:
                sentence_df = self.predict_sentiment(sentence_df)
            if self.cfg.classifier_mode.toxicity:
                sentence_df = self.predict_toxicity(sentence_df, sentences_emb)
        return sentence_df

    def predict_regard(self, sentence_df, sentences_emb):
        batch_size, dataloader = self.get_dataloader(sentences_emb)

        all_preds = np.empty(len(sentence_df))
        for i, inputs in enumerate(dataloader):
            if self.use_sklearn_model:
                preds = self.model.predict(inputs)
            else:
                if self.tokenizer:
                    inputs = self.tokenizer(inputs, padding=True, truncation=True,
                                          return_tensors="pt")
                    outputs = self.model(**inputs)
                    probs = outputs.logits.detach().cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                else:
                    outputs = self.model(torch.Tensor(inputs))
                    probs = F.log_softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1).detach().numpy()
            all_preds[i * batch_size: i * batch_size + batch_size] = preds

        # if self.cfg.label_col in sentence_df.columns:
        #     sentence_df[self.cfg.label_col] = sentence_df[self.cfg.label_col].astype(int)
        #     eval_prediction(
        #         self.output_path,
        #         self.input_path,
        #         all_preds,
        #         sentence_df,
        #         self.cfg.label_col,
        #         self.cfg.classifier_mode.store_misclassified,
        #     )
        sentence_df["regard"] = all_preds
        return sentence_df

    def predict_sentiment(self, sentence_df):
        model = SentimentModel()
        batch_size, dataloader = self.get_dataloader(sentence_df[self.cfg.text_col])
        all_preds = np.empty(len(sentence_df))
        for i, texts in enumerate(dataloader):
            preds_txt = model.predict_sentiment(texts)
            preds = [constants.VALENCE_MAP[x] for x in preds_txt]
            all_preds[i * batch_size: i * batch_size + batch_size] = preds
        sentence_df["sentiment"] = all_preds
        return sentence_df

    def predict_toxicity(self, sentence_df, sentences_emb):
        return sentence_df

    def get_dataloader(self, sentences):
        batch_size = 64
        if len(sentences) < batch_size:
            batch_size = len(sentences)
        dataloader = DataLoader(
            sentences,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
        )
        return batch_size, dataloader

    # def store_preds_per_class(self, cfg.classifier_mode, dest, preds, sentence_df, text_col):
    #     classes = set(preds)
    #     class_map = {0: "negative", 1: "neutral", 2: "positive"}
    #     for c in classes:
    #         texts = sentence_df.loc[sentence_df["regard"] == c, text_col]
    #         if cfg.classifier_mode.cda:
    #             dest_curr = os.path.join(dest, class_map[c])
    #             os.makedirs(dest_curr, exist_ok=True)
    #             if list(
    #                     filter(
    #                         any(texts.iloc[0].startswith(f) for f in constants.FEMALE_PREFIXES)
    #                     )
    #             ):
    #                 new_gender = "MALE"
    #                 orig_gender = "FEMALE"
    #                 flipped_texts = flip_gender(texts, True)
    #             elif list(
    #                     filter(
    #                         any(texts.iloc[0].startswith(f) for f in constants.MALE_PREFIXES)
    #                     )
    #             ):
    #                 new_gender = "FEMALE"
    #                 orig_gender = "MALE"
    #                 flipped_texts = flip_gender(texts, False)
    #
    #             text_per_gen = [texts, flipped_texts]
    #             for i, gen in enumerate([orig_gender, new_gender]):
    #                 with open(
    #                         os.path.join(dest_curr, f"{gen}_{class_map[c]}_regard.txt"), "a+"
    #                 ) as f:
    #                     for txt in text_per_gen[i]:
    #                         f.write(f"{remove_demographic(cfg.classifier_mode, txt)}\n")
    #         else:
    #             print("Storing predictions to", dest)
    #             with open(os.path.join(dest, f"{class_map[c]}_regard.txt"), "a+") as f:
    #                 for txt in texts:
    #                     f.write(f"{remove_demographic(cfg.classifier_mode, txt)}\n")


def eval_prediction(dest, path, preds, sentence_df, label_col, store_misclassified):
    if sentence_df.dtypes[label_col] == str:
        sentence_df[label_col] = sentence_df[label_col].map(constants.VALENCE_MAP)
    classes = set(sentence_df[label_col])
    n_classes = len(classes)
    results_dict = classification_report(
        sentence_df[label_col], preds, output_dict=True
    )
    sentence_df[label_col] = sentence_df[label_col].astype(int)
    confusion_matrix = np.zeros((n_classes, n_classes))
    misclassified_idcs = []
    for t_idx, p in zip(sentence_df.index, preds):
        t = sentence_df.loc[t_idx, label_col]
        confusion_matrix[t, p] += 1
        if t != p:
            misclassified_idcs.append(t_idx)

    labels = ["negative", "neutral", "positive"]
    plot = sns.heatmap(
        confusion_matrix,
        cmap="coolwarm",
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        annot_kws={"fontsize": 13},
    )
    plot.set_xlabel("True labels", fontsize=15)
    plot.set_ylabel("Predicted labels", fontsize=15)
    name_str = f"{os.path.basename(path).split('.')[0]}"
    plt.savefig(os.path.join(dest, f"conf_matrix_{name_str}.png"))
    pprint(results_dict)
    with open(os.path.join(dest, f"results_{name_str}.json"), "w") as outfile:
        json.dump(results_dict, outfile)

    if store_misclassified:
        misclassified_df = sentence_df.loc[misclassified_idcs, :]
        misclassified_df["regard"] = preds[misclassified_idcs]
        misclassified_df.to_csv(os.path.join(dest, f"misclassified_{name_str}.csv"))
    print(f"Storing results at {dest}.")














