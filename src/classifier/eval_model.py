import os
import json
from pprint import pprint

import hydra
import hydra.utils
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    classification_report, plot_confusion_matrix

from src.classifier.dataset import get_dataloader
from src.classifier.get_classifier_or_embedding import load_pretrained_sklearn, load_torch_model

from src.classifier.utils import store_preds
from src.visualize import plot_conf_matrix, aggregate_metrics


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


def evaluate_sklearn(
    embedding_name, model_type, model, X_test, Y_test, texts_test, plot_path
):
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, output_dict=True)
    pprint(report)

    plot_confusion_matrix(model, X_test, Y_test)

    name_str = f"{embedding_name}_{model_type}"
    plot_path = hydra.utils.to_absolute_path(plot_path)
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"conf_matrix_{name_str}.png"))
    with open(os.path.join(plot_path, f"report_{name_str}.json"), "w") as outfile:
        json.dump(report, outfile)

    store_preds(plot_path, name_str, Y_pred, Y_test, texts_test)
    acc = report["accuracy"]  # ["f1-score"]
    print("Storing evaluation results at", plot_path)
    return acc


def evaluate_on_testset(cfg, X_test, Y_test, texts_test):
    model_path = cfg.classifier_mode.model_path
    model_path = hydra.utils.to_absolute_path(model_path)
    results_path = cfg.classifier_mode.results_path
    results_path = hydra.utils.to_absolute_path(results_path)
    print(len(X_test))

    dest = os.path.join(
        results_path,
        cfg.classifier.name,
    )

    os.makedirs(dest, exist_ok=True)
    name_str = f"{cfg.embedding.name}_{cfg.classifier.name}"

    if cfg.classifier.name == "xgb":
        model = load_pretrained_sklearn(model_path)
        evaluate_sklearn(
            cfg.embedding.name,
            cfg.classifier.name,
            model,
            X_test,
            Y_test,
            texts_test,
            dest,
        )
    else:
        if cfg.classifier_mode.cv:
            results_dicts, conf_matrices = [], []
            for subfolder in os.listdir(model_path):
                for file in os.listdir(os.path.join(model_path, subfolder)):
                    model_path_i = os.path.join(model_path, subfolder, file)
                    if os.path.isfile(model_path_i):
                        dest_i = os.path.join(model_path, subfolder, "eval_results")
                        accuracy, results_dict, conf_matrix_npy = eval_single_model(cfg, X_test,
                                                                                    Y_test,
                                                                                    texts_test,
                                                                                    name_str,
                                                                                    dest_i, model_path_i)
                        results_dicts.append(results_dict)
                        conf_matrices.append(conf_matrix_npy)

            results_all, avg_conf = aggregate_metrics(results_dicts, conf_matrices, model_path)
            print("Results - (Mean, SD):", results_all)
            print("Avg. confusion matrix", avg_conf)

        else:
            eval_single_model(cfg, X_test, Y_test, texts_test, name_str, dest, model_path)


def eval_single_model(cfg, X_test, Y_test, texts_test, name_str, dest, model_path):
    test_loader = get_dataloader(X_test, Y_test, cfg.classifier_mode.batch_size, shuffle=False)
    model = load_torch_model(model_path, cfg.classifier.name, logger=None)
    model.to("cpu")
    model.eval()
    preds, labels = gather_preds_and_labels(model, test_loader)
    accuracy, results_dict, conf_matrix_npy = evaluate(preds, labels, texts_test,
                                                       classes=set(Y_test), name_str=name_str,
                                                       output_path=dest, plot=True)
    return accuracy, results_dict, conf_matrix_npy