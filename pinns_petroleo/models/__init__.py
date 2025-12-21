"""
Módulo de Modelos para PINNs em Petrofísica

Contém:
- FisicaPuraModel: Modelo baseado em equações físicas
- NNPuraModel: Rede Neural pura (data-driven)
- PINNModel: Physics-Informed Neural Network
"""

from .fisica_pura import FisicaPuraModel, calcular_correcao_argila
from .nn_pura import NNPuraModel
from .pinn import PINNModel

__all__ = [
    'FisicaPuraModel',
    'NNPuraModel',
    'PINNModel',
    'calcular_correcao_argila'
]
