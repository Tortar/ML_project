
import torch
from torchmetrics.classification import BinaryConfusionMatrix

confmat = BinaryConfusionMatrix()

def confusion_matrix(y_true, y_pred, normalize="all"):
	return confmat(y_pred, y_true)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
