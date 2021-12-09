import torch
from torch import tanh, nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy, f1


class RegardClassifier(pl.LightningModule):
    def __init__(self, n_embed, n_hidden_lin, n_output, lr, weight_vector, drop_p):
        super(RegardClassifier, self).__init__()
        self.save_hyperparameters()
        self.n_embed = n_embed
        self.n_hidden_lin = n_hidden_lin
        self.n_output = n_output
        self.lr = lr
        self.weight_vector = weight_vector

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # training_step defined the train loop. It is independent of forward
        inputs, labels = batch
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        output = self(inputs)
        if isinstance(output, tuple):
            output = output[0]
        loss = F.cross_entropy(output, labels, weight=self.weight_vector)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # training_step defined the train loop. It is independent of forward
        inputs, labels = batch
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        outputs = self(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = F.cross_entropy(outputs, labels)

        probs = F.log_softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = accuracy(preds, labels)
        f1_score = f1(preds, labels, num_classes=self.n_output, average="macro")

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # training_step defined the train loop. It is independent of forward
        inputs, labels = batch
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        outputs = self(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = F.cross_entropy(outputs, labels)

        probs = F.log_softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = accuracy(preds, labels)
        f1_score = f1(preds, labels, num_classes=self.n_output, average="macro")

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1_score, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-8)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer, mode="min", patience=5
        # )
        return {
            "optimizer": optimizer,
            #     "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class RegardLSTM(RegardClassifier):
    def __init__(
        self,
        n_embed,
        n_hidden,
        n_hidden_lin,
        n_output,
        n_layers,
        lr,
        weight_vector,
        bidirectional,
        gru,
        drop_p,
        drop_p_gru,
    ):
        RegardClassifier.__init__(
            self, n_embed, n_hidden_lin, n_output, lr, weight_vector, drop_p
        )
        drop_p_gru = drop_p_gru if drop_p_gru is not None else 0
        drop_p = drop_p if drop_p is not None else 0
        if gru:
            if n_hidden_lin > 0:

                self.lin1 = nn.Linear(n_embed, n_hidden_lin)
                self.dropout = nn.Dropout(drop_p)
                self.lstm = nn.GRU(
                    n_hidden_lin,
                    n_hidden,
                    n_layers,
                    batch_first=True,
                    dropout=drop_p_gru,
                    bidirectional=bidirectional,
                )

            else:
                self.lstm = nn.GRU(
                    n_embed,
                    n_hidden,
                    n_layers,
                    batch_first=True,
                    dropout=drop_p_gru,
                    bidirectional=bidirectional,
                )
        else:
            if n_hidden_lin > 0:
                self.lin1 = nn.Linear(n_embed, n_hidden_lin)
                self.lstm = nn.LSTM(
                    n_hidden_lin,
                    n_hidden,
                    n_layers,
                    batch_first=True,
                    dropout=drop_p_gru,
                    bidirectional=bidirectional,
                )
            else:
                self.lstm = nn.LSTM(
                    n_embed,
                    n_hidden,
                    n_layers,
                    batch_first=True,
                    dropout=drop_p_gru,
                    bidirectional=bidirectional,
                )
        self.fc = (
            nn.Linear(n_hidden * 2, n_output)
            if bidirectional
            else nn.Linear(n_hidden, n_output)
        )

    def forward(self, input_words):
        # INPUT   :  (batch_size, seq_length)
        if self.n_hidden_lin > 0:
            lin_out = self.lin1(input_words)
            lin_out = tanh(lin_out)
            lin_out = self.dropout(lin_out)
            lstm_out, h = self.lstm(lin_out)  # (batch_size, seq_length, n_hidden)
        else:
            lstm_out, h = self.lstm(input_words)
        fc_out = self.fc(lstm_out)  # (batch_size, seq_length, n_output)
        fc_out = fc_out[
            :, -1, :
        ]  # take only result of end of a sequence (batch_size, n_output)

        return fc_out, h


class RegardBERT(RegardClassifier):
    def __init__(
        self, n_embed, n_hidden_lin, n_hidden_lin_2, n_output, lr, weight_vector, drop_p
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