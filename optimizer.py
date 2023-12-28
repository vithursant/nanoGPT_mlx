from typing import List

import mlx.core as mx
import mlx.optimizers as optim


class AdamW(optim.Adam):
    r"""Implementation of the AdamW optimizer [1].

    Following the above convention, in contrast with [1], we do not use bias
    correction in the first and second moments for AdamW. We update the weights
    with a weight_decay (:math:`\lambda`) value:

    [1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay
    regularization. ICLR 2019.

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \alpha (\frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}} + \lambda w_t)

    Args:
        learning_rate (float): The learning rate :math:`\alpha`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
        weight_decay (float, optional): The weight decay :math:`\lambda`.
          Default: ``0``.
    """
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
