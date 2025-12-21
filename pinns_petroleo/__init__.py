"""
PINNs em Petrofísica
====================

Implementação de Physics-Informed Neural Networks para Previsão de Porosidade

Módulos principais:
- models: FisicaPuraModel, NNPuraModel, PINNModel
- utils: DataPreprocessor, métricas, visualizações

Autores:
- Edmilson Delfim Praia
- Cirilo Cauxeiro

Exemplo de uso:
--------------
>>> from models import NNPuraModel
>>> from utils import generate_synthetic_data, DataPreprocessor
>>>
>>> # Gerar dados
>>> df = generate_synthetic_data(n_samples=5000)
>>>
>>> # Preprocessar
>>> preprocessor = DataPreprocessor()
>>> result = preprocessor.prepare_data_from_dataframe(df)
>>> X_train, X_val, X_test, y_train, y_val, y_test = result[:6]
>>>
>>> # Treinar modelo
>>> model = NNPuraModel(input_dim=4)
>>> model.fit(X_train, y_train, X_val, y_val, epochs=100)
>>> predictions = model.predict(X_test)
"""

__version__ = '1.0.0'
__author__ = 'Edmilson Delfim Praia, Cirilo Cauxeiro'

# Importações principais
from .models import FisicaPuraModel, NNPuraModel, PINNModel
from .utils import DataPreprocessor, generate_synthetic_data
from .utils import calculate_metrics, plot_scatter_prediction

__all__ = [
    'FisicaPuraModel',
    'NNPuraModel',
    'PINNModel',
    'DataPreprocessor',
    'generate_synthetic_data',
    'calculate_metrics',
    'plot_scatter_prediction'
]
