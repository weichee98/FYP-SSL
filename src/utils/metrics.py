from os import stat
import torch
import numpy as np


class EMA:
    """
    EMA(t) = val(t) * k + EMA(t - 1) * (1 - k)
    """

    def __init__(self, k):
        """
        k: float or None
            A value in range (0, 1]
        """
        self._iter = 0
        self._ema = None
        self.k = k

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        if k is None:
            self._k = 1
        elif isinstance(k, float):
            if 0 < k <= 1:
                self._k = k
            else:
                raise ValueError("k must be within (0, 1]")
        else:
            raise TypeError("k must be float or None")

    @property
    def iter(self):
        return self._iter

    def update(self, *args):
        if self._ema is None:
            self._ema = np.array(args)
        elif isinstance(self._ema, dict):
            raise TypeError(
                "invalid usage of update method, "
                "please use update_dict method to update instead"
            )
        else:
            self._ema = self.k * np.array(args) + (1 - self.k) * self._ema
        self._iter += 1
        return tuple(self._ema)

    def update_dict(self, item_dict):
        if not isinstance(item_dict, dict):
            raise TypeError(
                "invalid usage of update_dict method, "
                "please use update method to update instead"
            )
        if self._ema is None:
            self._ema = item_dict.copy()
        elif not isinstance(self._ema, dict):
            raise TypeError(
                "invalid usage of update_dict method, "
                "please use update method to update instead"
            )
        else:
            for key in self._ema:
                self._ema[key] = self.k * item_dict[key] + (1 - self.k) * self._ema[key]
        self._iter += 1
        return self._ema.copy()

    
class ClassificationMetrics:
    """
    https://www.labtestsonline.org.au/understanding/test-accuracy-and-reliability/how-reliable-is-pathology-testing

    Only for 2 class classification (diseased and controlled).
        - class 1 = diseased
        - class 0 = controlled
    """

    @staticmethod
    def _check_y(y):
        assert 1 <= y.ndim <= 2, "y.ndim must be 1 or 2, but given {}".format(y.ndim)
        if y.ndim == 2:
            assert y.size(1) == 2, "dim 1 of y must have size 2, but given size {}".format(y.size(1))
            y = y.argmax(dim=1)
        elif y.ndim == 1:
            assert torch.all((y == 0) | (y == 1)), "y can only contain 0 or 1"
        return y

    def _check_args(function):
        def wrapper(y_true, y_pred):
            with torch.no_grad():
                y_pred = __class__._check_y(y_pred)
                y_true = __class__._check_y(y_true)
                return function(y_true, y_pred)
        return wrapper

    @staticmethod
    @_check_args
    def accuracy(y_true, y_pred):
        correct = y_pred == y_true
        accuracy = correct.float().mean()
        return accuracy

    @staticmethod
    @_check_args
    def tnr(y_true, y_pred):
        """
        True Negative Rate

        The ability of a test to correctly identify people without the disease,
        also called:
            - specificity
            - selectivity

        TN / (TN + FP)
        """
        tn = ((y_true == 0) & (y_pred == 0)).float().sum()
        fp = ((y_true == 0) & (y_pred == 1)).float().sum()
        return tn / (tn + fp)

    @staticmethod
    @_check_args
    def tpr(y_true, y_pred):
        """
        True Positive Rate

        The ability of a test to correctly identify patients with a disease,
        also called:
            - sensitivity
            - recall
            - hit rate

        TP / (TP + FN)
        """
        tp = ((y_true == 1) & (y_pred == 1)).float().sum()
        fn = ((y_true == 1) & (y_pred == 0)).float().sum()
        return tp / (tp + fn)

    @staticmethod
    @_check_args
    def ppv(y_true, y_pred):
        """
        Positive Predictive Value

        The reliability of a test when it identifies a patient as having a disease,
        also called:
            - precision

        TP / (TP + FP)
        """
        tp = ((y_true == 1) & (y_pred == 1)).float().sum()
        fp = ((y_true == 0) & (y_pred == 1)).float().sum()
        return tp / (tp + fp)

    @staticmethod
    @_check_args
    def npv(y_true, y_pred):
        """
        Negative Predictive Value

        The reliability of a test when it identifies a patient as not having a disease

        TN / (TN + FN)
        """
        tn = ((y_true == 0) & (y_pred == 0)).float().sum()
        fn = ((y_true == 1) & (y_pred == 0)).float().sum()
        return tn / (tn + fn)

    @staticmethod
    @_check_args
    def fpr(y_true, y_pred):
        """
        False Positive Rate

        FP / (TN + FP)
        """
        tn = ((y_true == 0) & (y_pred == 0)).float().sum()
        fp = ((y_true == 0) & (y_pred == 1)).float().sum()
        return fp / (tn + fp)

    @staticmethod
    @_check_args
    def fnr(y_true, y_pred):
        """
        False Negative Rate

        FN / (TP + FN)
        """
        tp = ((y_true == 1) & (y_pred == 1)).float().sum()
        fn = ((y_true == 1) & (y_pred == 0)).float().sum()
        return fn / (tp + fn)

    @staticmethod
    @_check_args
    def fdr(y_true, y_pred):
        """
        False Discovery Rate

        The probability of a test identifying a patient as having a disease incorrectly

        FP / (TP + FP)
        """
        tp = ((y_true == 1) & (y_pred == 1)).float().sum()
        fp = ((y_true == 0) & (y_pred == 1)).float().sum()
        return fp / (tp + fp)

    @staticmethod
    @_check_args
    def fomr(y_true, y_pred):
        """
        False Ommision Rate

        The probability of a test identifying a patient as not having a disease incorrectly

        FN / (TN + FN)
        """
        tn = ((y_true == 0) & (y_pred == 0)).float().sum()
        fn = ((y_true == 1) & (y_pred == 0)).float().sum()
        return fn / (tn + fn)

    @staticmethod
    @_check_args
    def f1_score(y_true, y_pred):
        """
        2TP / (2TP + FP + FN)
        """
        tp = ((y_true == 1) & (y_pred == 1)).float().sum()
        fp = ((y_true == 0) & (y_pred == 1)).float().sum()
        fn = ((y_true == 1) & (y_pred == 0)).float().sum()
        return 2 * tp / (2 * tp + fp + fn)


class CummulativeClassificationMetrics:
    """
    Only for 2 class classification (diseased and controlled).
        - class 1 = diseased
        - class 0 = controlled
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.total = 0

    @staticmethod
    def _check_y(y):
        assert 1 <= y.ndim <= 2, "y.ndim must be 1 or 2, but given {}".format(y.ndim)
        if y.ndim == 2:
            assert y.size(1) == 2, "dim 1 of y must have size 2, but given size {}".format(y.size(1))
            y = y.argmax(dim=1)
        elif y.ndim == 1:
            assert torch.all((y == 0) | (y == 1)), "y can only contain 0 or 1"
        return y

    def update_batch(self, y_true, y_pred):
        y_pred = self._check_y(y_pred)
        y_true = self._check_y(y_true)
        self.tp += ((y_true == 1) & (y_pred == 1)).float().sum()
        self.tn += ((y_true == 0) & (y_pred == 0)).float().sum()
        self.fp += ((y_true == 0) & (y_pred == 1)).float().sum()
        self.fn += ((y_true == 1) & (y_pred == 0)).float().sum()
        self.total += y_true.size(0)

    @property
    def accuracy(self):
        return (self.tp + self.tn) / self.total

    @property
    def tnr(self):
        return self.tn / (self.tn + self.fp)

    @property
    def tpr(self):
        return self.tp / (self.tp + self.fn)

    @property
    def ppv(self):
        return self.tp / (self.p + self.fp)

    @property
    def npv(self):
        return self.tn / (self.tn + self.fn)

    @property
    def fpr(self):
        return self.fp / (self.tn + self.fp)

    @property
    def fnr(self):
        return self.fn / (self.tp + self.fn)

    @property
    def fdr(self):
        return self.p / (self.tp + self.fp)

    @property
    def fomr(self):
        return self.fn / (self.tn + self.fn)

    @property
    def f1_score(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)