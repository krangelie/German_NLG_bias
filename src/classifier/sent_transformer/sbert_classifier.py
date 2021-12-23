from torch import tanh, nn as nn

from src.classifier.classifiers import RegardClassifier


class RegardBERT(RegardClassifier):
    def __init__(
        self, n_embed, n_hidden_lin, n_hidden_lin_2=0, n_output=1, lr=1, weight_vector=[],
            drop_p=0.0
    ):
        RegardClassifier.__init__(
            self, n_embed, n_hidden_lin, n_output, lr, weight_vector, drop_p
        )
        self.n_hidden_lin_2 = n_hidden_lin_2
        self.linear = nn.Linear(n_embed, n_hidden_lin)  # dense
        self.dense = nn.Linear(n_hidden_lin, n_hidden_lin_2)
        self.dropout = nn.Dropout(drop_p)
        if n_hidden_lin_2 > 0:
            self.classifier = nn.Linear(n_hidden_lin_2, n_output)  # out_proj
        else:
            self.classifier = nn.Linear(n_hidden_lin, n_output)  # out_proj

    def forward(self, input_sentences):
        # input = sentence embeddings
        x = self.linear(input_sentences)
        if self.n_hidden_lin_2 > 0:
            x = self.dense(x)
            x = tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x