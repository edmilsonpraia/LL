"""
Módulo de Utilitários para PINNs em Petrofísica

Contém:
- data_preprocessing: Carregamento e pré-processamento de dados
- metrics: Cálculo de métricas e avaliação
- visualizations: Funções de visualização
"""

from .data_preprocessing import DataPreprocessor, generate_synthetic_data
from .metrics import (
    calculate_metrics,
    print_metrics,
    compare_models,
    evaluate_model_comprehensive
)
from .visualizations import (
    plot_scatter_prediction,
    plot_learning_curves,
    plot_pinn_loss_decomposition,
    plot_residuals_analysis,
    plot_feature_importance,
    plot_correlation_matrix,
    plot_model_comparison_bar,
    plot_lambda_sensitivity,
    plot_well_log_comparison
)

__all__ = [
    # Data preprocessing
    'DataPreprocessor',
    'generate_synthetic_data',

    # Metrics
    'calculate_metrics',
    'print_metrics',
    'compare_models',
    'evaluate_model_comprehensive',

    # Visualizations
    'plot_scatter_prediction',
    'plot_learning_curves',
    'plot_pinn_loss_decomposition',
    'plot_residuals_analysis',
    'plot_feature_importance',
    'plot_correlation_matrix',
    'plot_model_comparison_bar',
    'plot_lambda_sensitivity',
    'plot_well_log_comparison'
]
