import torch


class ClassificationMetrics:
    """
    https://www.labtestsonline.org.au/understanding/test-accuracy-and-reliability/how-reliable-is-pathology-testing

    Only for 2 class classification (diseased and controlled).
        - class 1 = diseased
        - class 0 = controlled
    """

    @staticmethod
    def _check_y(y):
        assert 1 <= y.ndim <= 2, "y.ndim must be 1 or 2, but given {}".format(
            y.ndim
        )
        if y.ndim == 2:
            assert (
                y.size(1) == 2
            ), "dim 1 of y must have size 2, but given size {}".format(
                y.size(1)
            )
            y = y.argmax(dim=1)
        elif y.ndim == 1:
            assert torch.all((y == 0) | (y == 1)), "y can only contain 0 or 1"
        return y

    def _check_args(function):
        def wrapper(y_true, y_pred) -> torch.Tensor:
            with torch.no_grad():
                y_pred = __class__._check_y(y_pred)
                y_true = __class__._check_y(y_true)
                return function(y_true, y_pred)

        return wrapper

    @staticmethod
    @_check_args
    def accuracy(y_true, y_pred) -> torch.Tensor:
        correct = y_pred == y_true
        accuracy = correct.float().mean()
        return accuracy

    @staticmethod
    @_check_args
    def tnr(y_true, y_pred) -> torch.Tensor:
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
    def tpr(y_true, y_pred) -> torch.Tensor:
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
    def ppv(y_true, y_pred) -> torch.Tensor:
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
    def npv(y_true, y_pred) -> torch.Tensor:
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
    def fpr(y_true, y_pred) -> torch.Tensor:
        """
        False Positive Rate

        FP / (TN + FP)
        """
        tn = ((y_true == 0) & (y_pred == 0)).float().sum()
        fp = ((y_true == 0) & (y_pred == 1)).float().sum()
        return fp / (tn + fp)

    @staticmethod
    @_check_args
    def fnr(y_true, y_pred) -> torch.Tensor:
        """
        False Negative Rate

        FN / (TP + FN)
        """
        tp = ((y_true == 1) & (y_pred == 1)).float().sum()
        fn = ((y_true == 1) & (y_pred == 0)).float().sum()
        return fn / (tp + fn)

    @staticmethod
    @_check_args
    def fdr(y_true, y_pred) -> torch.Tensor:
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
    def fomr(y_true, y_pred) -> torch.Tensor:
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
    def f1_score(y_true, y_pred) -> torch.Tensor:
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
        assert 1 <= y.ndim <= 2, "y.ndim must be 1 or 2, but given {}".format(
            y.ndim
        )
        if y.ndim == 2:
            assert (
                y.size(1) == 2
            ), "dim 1 of y must have size 2, but given size {}".format(
                y.size(1)
            )
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
    def accuracy(self) -> torch.Tensor:
        return (self.tp + self.tn) / self.total

    @property
    def tnr(self) -> torch.Tensor:
        return self.tn / (self.tn + self.fp)

    @property
    def tpr(self) -> torch.Tensor:
        return self.tp / (self.tp + self.fn)

    @property
    def ppv(self) -> torch.Tensor:
        return self.tp / (self.tp + self.fp)

    @property
    def npv(self) -> torch.Tensor:
        return self.tn / (self.tn + self.fn)

    @property
    def fpr(self) -> torch.Tensor:
        return self.fp / (self.tn + self.fp)

    @property
    def fnr(self) -> torch.Tensor:
        return self.fn / (self.tp + self.fn)

    @property
    def fdr(self) -> torch.Tensor:
        return self.fp / (self.tp + self.fp)

    @property
    def fomr(self) -> torch.Tensor:
        return self.fn / (self.tn + self.fn)

    @property
    def f1_score(self) -> torch.Tensor:
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
