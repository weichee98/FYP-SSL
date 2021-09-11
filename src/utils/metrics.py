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
        else:
            self._ema = self.k * np.array(args) + (1 - self.k) * self._ema
        self._iter += 1
        return tuple(self._ema)

    