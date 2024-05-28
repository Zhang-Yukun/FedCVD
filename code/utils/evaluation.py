import pandas as pd
import torch
import sklearn.metrics as metrics
import numpy as np
import json


def get_pred_label(pred_score: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        one = torch.ones_like(pred_score, dtype=torch.float32)
        zero = torch.zeros_like(pred_score, dtype=torch.float32)
        pred_label = torch.where(pred_score >= 0.5, one, zero)
    return pred_label


def transfer_tensor_to_numpy(tensor):
    return tensor.cpu().numpy()


def calculate_accuracy_every_label(pred_label, true_label, normalize=False):
    pred_label = pred_label.flatten()
    true_label = true_label.flatten()
    return metrics.accuracy_score(true_label, pred_label, normalize=normalize)


"""
将标签作为整体计算准确率, 当所有标签分类正确时记为正确
"""


def calculate_accuracy(pred_label, true_label, normalize=False):
    return metrics.accuracy_score(true_label, pred_label, normalize=normalize)


"""
计算每一个标签的准确率
"""


def calculate_accuracy_per_label(pred_label, true_label, normalize=False):
    pred_label = pred_label.T
    true_label = true_label.T
    n_classes = pred_label.shape[0]
    per_accuracy = np.zeros((n_classes, ), dtype=float)
    for i in range(n_classes):
        per_accuracy[i] = metrics.accuracy_score(true_label[i], pred_label[i], normalize=normalize)
    return per_accuracy


def calculate_multilabel_confusion_matrix_info(pred_label, true_label):
    mcm = metrics.multilabel_confusion_matrix(true_label, pred_label)
    tn, tp, fn, fp = mcm[:, 0, 0], mcm[:, 1, 1], mcm[:, 1, 0], mcm[:, 0, 1]
    return tn, tp, fn, fp


def calculate_miss_rate(tp, fn):
    miss_rate = fn / (tp + fn + 1e-7)
    return miss_rate


def calculate_specificity(fp, tn):
    specificity = tn / (fp + tn + 1e-7)
    return specificity


def calculate_fall_out(fp, tn):
    fall_out = fp / (fp + tn + 1e-7)
    return fall_out


def calculate_multilabel_metric(tn, tp, fn, fp):
    pfp = np.where((tp + fp) == 0, 1, fp)
    rfn = np.where((tp + fn) == 0, 1, fn)
    mtp = np.where((tp + fn) == 0, 1, tp)
    sfp = np.where((fp + tn) == 0, 1, fp)
    ftn = np.where((fp + tn) == 0, 1, tn)

    precision = tp / (tp + pfp)
    recall = tp / (tp + rfn)
    miss_rate = fn / (mtp + fn)
    specificity = tn / (sfp + tn)
    fall_out = fp / (fp + ftn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        'precision': precision,
        'recall': recall,
        'miss_rate': miss_rate,
        'specificity': specificity,
        'fall_out': fall_out,
        'accuracy': accuracy
    }


"""
计算每一个标签的 F1 分数
"""


def calculate_f1_score(pred_label, true_label, average=None, zero_division=0):
    return metrics.f1_score(true_label, pred_label, average=average, zero_division=zero_division)


"""
计算每一个标签的精确度
"""


def calculate_precision(pred_label, true_label, average=None, zero_division=0):
    return metrics.precision_score(true_label, pred_label, average=average, zero_division=zero_division)


"""
计算每一个标签的召回率
"""


def calculate_recall(pred_label, true_label, average=None, zero_division=0):
    return metrics.recall_score(true_label, pred_label, average=average, zero_division=zero_division)


"""
计算整体的汉明损失
"""


def calculate_hamming_loss(pred_label, true_label):
    return metrics.hamming_loss(true_label, pred_label)


def calculate_average_precision_score(pred_score, true_label, average=None):
    return metrics.average_precision_score(true_label, pred_score, average=average)


def calculate_precision_recall_fscore_support(pred_label, true_label, average=None):
    return metrics.precision_recall_fscore_support(true_label, pred_label, average=average)


def calculate_roc_auc_score(pred_score, true_label, average=None):
    class_sum = np.sum(true_label, axis=0)
    cal_col = []
    uncal_col = []
    for col in range(len(class_sum)):
        if class_sum[col] == 0:
            uncal_col.append(col)
        else:
            cal_col.append(col)
    if not uncal_col:
        return metrics.roc_auc_score(true_label, pred_score, average=average)
    true_label_copy = true_label.copy()
    true_label_copy[0][uncal_col] = 1
    roc_auc_score = metrics.roc_auc_score(true_label_copy, pred_score, average=average)
    roc_auc_score[uncal_col] = np.nan
    return roc_auc_score


def calculate_multilabel_metrics(pred_score, pred_label, true_label, average=None, normalize=True, zero_division=0):
    accuracy = calculate_accuracy(pred_label, true_label, normalize=normalize)
    per_accuracy = calculate_accuracy_per_label(pred_label, true_label, normalize=normalize)
    precision = calculate_precision(pred_label, true_label, average=average, zero_division=zero_division)
    recall = calculate_recall(pred_label, true_label, average=average, zero_division=zero_division)
    f1_score = calculate_f1_score(pred_label, true_label, average=average, zero_division=zero_division)
    micro_f1 = calculate_f1_score(pred_label, true_label, average="micro", zero_division=zero_division)
    average_precision_score = calculate_average_precision_score(pred_score, true_label, average=average)
    roc_auc_score = calculate_roc_auc_score(pred_score, true_label, average=average)
    hamming_loss = calculate_hamming_loss(pred_label, true_label)

    tn, tp, fn, fp = calculate_multilabel_confusion_matrix_info(pred_label, true_label)
    missing_rate = 1 - recall
    specificity = calculate_specificity(fp, tn)
    fall_out = 1 - specificity
    false_alarm = 1 - precision

    metric_dict = {'accuracy': accuracy,
                   'per_accuracy': per_accuracy.tolist(),
                   'precision': precision.tolist(),
                   'recall': recall.tolist(),
                   'f1_score': f1_score.tolist(),
                   'micro_f1': micro_f1,
                   'average_precision_score': average_precision_score.tolist(),
                   'roc_auc_score': roc_auc_score.tolist(),
                   'hamming_loss': hamming_loss,
                   'missing_rate': missing_rate.tolist(),
                   'specificity': specificity.tolist(),
                   'fall_out': fall_out.tolist(),
                   'false_alarm': false_alarm.tolist()
                   }
    return metric_dict


def transfer_metrics_to_dataframe(metric_dict: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(metric_dict)


class Accumulator:
    """
    For accumulating sums over `n` variables
    """
    def __init__(self, n: int = 1):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [x + float(y) for x, y in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Evaluator:
    def __init__(self):
        self.data = {
            "train": {},
            "local_test": {},
            "global_test": {}
        }

    def evaluate(self, mode, epoch, pred_score, pred_label, true_label):
        raise NotImplementedError()

    def save(self, path):
        with open(path, "w") as file:
            file.write(json.dumps(self.data))


class MultiLabelEvaluator(Evaluator):
    def add_dict(self, mode, epoch, metric_dict):
        self.data[mode][epoch] = metric_dict

    def evaluate(self, mode, epoch, pred_score, pred_label, true_label):
        metric_dict = calculate_multilabel_metrics(pred_score, pred_label, true_label)
        self.data[mode][epoch] = metric_dict


class FedClientMultiLabelEvaluator(Evaluator):
    def add_dict(self, mode, cround, epoch, metric_dict):
        if cround not in self.data[mode].keys():
            self.data[mode][cround] = {}
        self.data[mode][cround][epoch] = metric_dict

class FedServerMultiLabelEvaluator(Evaluator):
    def add_dict(self, mode, cround, metric_dict):
        self.data[mode][cround] = metric_dict