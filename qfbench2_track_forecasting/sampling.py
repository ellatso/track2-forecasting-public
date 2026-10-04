"""## Executive summary (read this first)

Generate reproducible joint normal innovations with antithetic or Sobol sampling.
Sobol uses a full power-of-two design and keeps all points. No fitting or scoring
is done here. Each column is one asset's innovation over one disjoint interval.
"""

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc


class InnovationSampler:
    def __init__(self, seed, draws, assets, intervals, method):
        dimension = assets * intervals
        if method == "sobol":
            if draws <= 0 or draws & (draws - 1):
                raise ValueError("Sobol draws must be a power of two")
            engine = qmc.Sobol(dimension, scramble=True, rng=np.random.default_rng(seed))
            uniform = engine.random_base2(int(np.log2(draws)))
            values = ndtri(np.clip(uniform, np.finfo(float).eps, 1 - np.finfo(float).eps))
        elif method == "antithetic":
            if draws % 2:
                raise ValueError("Antithetic draws must be even")
            half = np.random.default_rng(seed).standard_normal((draws // 2, dimension))
            values = np.concatenate([half, -half], axis=0)
        else:
            raise ValueError("Unknown innovation sampling method")
        self.values = values.reshape(draws, intervals, assets)
        self.position = 0

    def standard_normal(self, shape):
        if shape != (self.values.shape[0], self.values.shape[2]):
            raise ValueError("Innovation shape changed")
        if self.position >= self.values.shape[1]:
            raise ValueError("Innovation interval budget exhausted")
        result = self.values[:, self.position, :]
        self.position += 1
        return result
