import mlx.core as mx
import mlx.optimizers as optim
from typing import List


class AdamW(optim.Adam):
    def __init__(
        self,
        learning_rate: float,
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(learning_rate=learning_rate, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def apply_single(self, gradient: mx.array, parameter: mx.array, state):
        parameter -= self.weight_decay * self.learning_rate * parameter
        return super().apply_single(gradient, parameter, state)

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate
