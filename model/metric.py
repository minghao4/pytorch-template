import torch
from torch import Tensor


def my_metric(output: Tensor, target: Tensor) -> float:
    with torch.no_grad():
        pred: Tensor = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)  # mypy error, len(Tensor) always returns # of first dim
        correct: float = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output: Tensor, target: Tensor, k: int = 3) -> float:
    with torch.no_grad():
        pred: Tensor = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct: float = 0

        i: int
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()

    return correct / len(target)
