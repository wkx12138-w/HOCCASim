"""Minimal Gymnasium-compatible space shims."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as _gym

    GYMNASIUM_AVAILABLE = True
except Exception:
    _gym = None
    GYMNASIUM_AVAILABLE = False

__all__ = [
    "Box", "Discrete", "MultiBinary", "Dict", "GYMNASIUM_AVAILABLE", "spaces_module",
]

if GYMNASIUM_AVAILABLE:

    Box = _gym.spaces.Box
    Discrete = _gym.spaces.Discrete
    MultiBinary = _gym.spaces.MultiBinary
    Dict = _gym.spaces.Dict

    def spaces_module() -> Any:
        return _gym.spaces

else:

    class Space:
        def __init__(self, shape, dtype, seed=None):
            self.shape = tuple(int(s) for s in shape)
            self.dtype = np.dtype(dtype)
            self._rng = np.random.default_rng(seed)

        @property
        def np_random(self):
            return self._rng

        def seed(self, seed=None):
            self._rng = np.random.default_rng(seed)

        def sample(self):
            raise NotImplementedError

        def contains(self, x):
            raise NotImplementedError

    class Box(Space):
        def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
            low = np.asarray(low, dtype=np.float64)
            high = np.asarray(high, dtype=np.float64)
            if shape is None:
                shape = low.shape
            low = np.broadcast_to(low, shape)
            high = np.broadcast_to(high, shape)
            super().__init__(tuple(shape), dtype, seed)
            self.low = low.astype(self.dtype)
            self.high = high.astype(self.dtype)

        def sample(self):
            return self._rng.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and bool(
                np.all(x >= self.low - 1e-9) and np.all(x <= self.high + 1e-9)
            )

    class Discrete(Space):
        def __init__(self, n, seed=None, start=0):
            super().__init__((), np.int64, seed)
            self.n = int(n)
            self.start = int(start)

        def sample(self):
            return int(self._rng.integers(self.start, self.start + self.n))

        def contains(self, x):
            try:
                return self.start <= int(x) < self.start + self.n
            except (TypeError, ValueError):
                return False

    class MultiBinary(Space):
        def __init__(self, n, seed=None):
            super().__init__((int(n),), np.int8, seed)
            self.n = int(n)

        def sample(self):
            return self._rng.integers(0, 2, size=self.n).astype(np.int8)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and bool(np.all((x == 0) | (x == 1)))

    class Dict(Space):
        def __init__(self, spaces, seed=None):
            super().__init__((), np.float32, seed)
            self.spaces = dict(spaces)

        def sample(self):
            return {k: v.sample() for k, v in self.spaces.items()}

        def contains(self, x):
            if not isinstance(x, dict) or set(x) != set(self.spaces):
                return False
            return all(self.spaces[k].contains(x[k]) for k in self.spaces)

    def spaces_module() -> Any:
        import types

        return types.SimpleNamespace(
            Box=Box, Discrete=Discrete, MultiBinary=MultiBinary, Dict=Dict
        )
