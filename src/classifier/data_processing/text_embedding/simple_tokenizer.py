import string
from HanTa import HanoverTagger as ht
from pprint import pprint
from pandas import DataFrame
import nltk


class SimpleTokenizer:
    """Simple tokenization for sentences. Used for non-Transformer-based models."""

    def __init__(
        self,
        language,
        to_lower=True,
        use_remove_punctuation=True,
    ):
        self.language = language
        self.to_lower = to_lower
        self.use_remove_punctuation = use_remove_punctuation


    def tokenize(self, df: DataFrame, text_col="text") -> DataFrame:
        """
        Takes as input a DataFrame with sentences in column 'text_col'.
        Appends at least one new column to the DataFrame containing lists of single word tokens.
        """

        texts = df[text_col]
        if self.to_lower:
            texts = texts.apply(lambda text: text.lower())

        if self.use_remove_punctuation:
            texts = texts.apply(lambda text: self.remove_punctuation(text))

        # Split sentences into list of words
        df["tokenized"] = texts.apply(
            lambda text: nltk.tokenize.word_tokenize(text, language=self.language)
        )
        return df

    def remove_punctuation(self, text):
        regular_punct = list(string.punctuation)
        for punc in regular_punct:
            if punc in text:
                text = text.replace(punc, " ")
        return text.strip()
