import numpy as np
from itertools import pairwise


class BNN_Gaussian:

    def __init__(self, layer_sizes, sigma_weight=1.0, sigma_bias=1.0):
        self.dims = tuple(layer_sizes[1:])
        self.shapes = tuple((a, b) for a, b in pairwise(layer_sizes))
        self.size = sum(self.dims) + sum(a * b for a, b in self.shapes)
        self.sigma_weight = sigma_weight
        self.sigma_bias = sigma_bias
        self.weight = None
        self.bias = None
        self.n_weight = sum(a * b for a, b in self.shapes)
        self.n_bias = sum(self.dims)

    def sample_weight(self):
        return [self.sigma_weight * np.random.randn(*shape) for shape in self.shapes]

    def sample_bias(self):
        return [self.sigma_bias * np.random.randn(dim) for dim in self.dims]

    def log_pdf(self, weight=None, bias=None):
        if weight is None: weight = self.weight
        if bias is None: bias = self.bias
        assert weight is not None
        assert bias is not None
        quad = -0.5 * (sum((w**2).sum() for w in weight) / self.sigma_weight**2 + \
                       sum((b**2).sum() for b in bias) / self.sigma_bias**2)
        lift = -0.5 * self.size * np.log(2 * np.pi) \
               - self.n_weight * np.log(self.sigma_weight) \
               - self.n_bias * np.log(self.sigma_bias)
        return lift + quad

    def norm_term(self, n):
        return np.sqrt(self.sigma_weight**2 * n + self.sigma_bias**2)
        
    def __call__(self, x, weight=None, bias=None):
        if weight is None: weight = self.weight or self.sample_weight()
        if bias is None: bias = self.bias or self.sample_bias()
        for w, b in zip(weight[:-1], bias[:-1]):
            z = (x @ w + b) / self.norm_term(len(w))
            x = np.tanh(z)
        z = (x @ weight[-1] + bias[-1]) / self.norm_term(len(weight[-1]))
        return z