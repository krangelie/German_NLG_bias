import os
import json

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from src.classifier.utils import store_preds
from src.visualize import plot_conf_matrix


def evaluate(preds, labels, texts_test, classes, name_str, output_path, plot=False):

    conf_matrix_npy, acc_per_class = get_conf_matrix(classes, preds, labels)
    results_dict = get_metrics(preds, labels)
    results_dict["acc_per_class"] = acc_per_class
    print(f"Test Accuracy per class: {acc_per_class}")
    print(f"Test Accuracy averaged: {results_dict['accuracy']}")
    print(f"Test F1-score macro-averaged: {results_dict['f1']}")
    #print(f"Test F1-score micro-averaged: {mean_f1_micro}")
    if plot:
        os.makedirs(output_path, exist_ok=True)
        plot_conf_matrix(conf_matrix_npy, output_path, f"conf_matrix_{name_str}")

    store_preds(output_path, name_str, preds, labels, texts_test)
    print(results_dict)
    with open(os.path.join(output_path, f"results_{name_str}.json"), "w") as outfile:
        json.dump(results_dict, outfile)
    np.save(os.path.join(output_path, f"conf_matrix.npy"), conf_matrix_npy)
    return results_dict['accuracy'], results_dict, conf_matrix_npy


def gather_preds_and_labels(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)
    preds_all = []
    labels_all = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        test_outputs = model(inputs)
        if isinstance(test_outputs, tuple):
            test_outputs = test_outputs[0]
        logits = F.log_softmax(test_outputs, dim=1)
        preds = torch.argmax(logits, dim=1)
        preds_all += preds
        labels_all += labels
    labels_all = torch.tensor(labels_all).cpu().numpy()
    preds_all = torch.FloatTensor(preds_all).cpu().numpy()
    return preds_all, labels_all


def get_conf_matrix(classes, preds, labels):
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes))
    num_per_class = {c: 0 for c in classes}
    for t, p in zip(labels, preds):
        num_per_class[int(p.item())] += 1
        confusion_matrix[int(t), int(p)] += 1
    print(f"Confusion matrix: {confusion_matrix}")
    acc_list = confusion_matrix.diagonal() / confusion_matrix.sum(1)
    acc_per_class = dict(zip([str(c) for c in classes], acc_list))
    print(f"Accuracy per class:", acc_per_class)
    return confusion_matrix, acc_per_class


def get_metrics(preds, labels):
    average = "macro"
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds, average=average)
    precision = precision_score(y_true=labels, y_pred=preds, average=average)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
