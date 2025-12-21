"""
Funções de Avaliação e Métricas
- Cálculo de métricas (MSE, MAE, R², RMSE)
- Análise de resíduos
- Comparação de modelos

Autor: Edmilson Delfim Praia
Co-autor: Cirilo Cauxeiro
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      model_name: str = "Modelo") -> Dict[str, float]:
    """
    Calcula métricas de desempenho.

    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        model_name: Nome do modelo (para logging)

    Returns:
        Dicionário com métricas
    """
    # Garantir arrays 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calcular métricas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Métricas adicionais
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # MAPE (%)

    # Resíduos
    residuals = y_true - y_pred
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)

    metrics = {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'residuals_mean': residuals_mean,
        'residuals_std': residuals_std,
        'n_samples': len(y_true)
    }

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = None):
    """
    Imprime métricas de forma formatada.

    Args:
        metrics: Dicionário com métricas
        title: Título opcional
    """
    if title:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    print(f"\nModelo: {metrics.get('model_name', 'N/A')}")
    print(f"N amostras: {metrics.get('n_samples', 'N/A')}")
    print(f"\nMétricas de Performance:")
    print(f"  R²    : {metrics['r2']:.4f}")
    print(f"  RMSE  : {metrics['rmse']:.4f}")
    print(f"  MAE   : {metrics['mae']:.4f}")
    print(f"  MSE   : {metrics['mse']:.4f}")
    print(f"  MAPE  : {metrics['mape']:.2f}%")
    print(f"\nAnálise de Resíduos:")
    print(f"  Média : {metrics['residuals_mean']:.6f}")
    print(f"  Std   : {metrics['residuals_std']:.4f}")


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compara múltiplos modelos.

    Args:
        results: Dicionário com resultados de diferentes modelos
                 {nome_modelo: dicionário_métricas}

    Returns:
        DataFrame com comparação
    """
    comparison_data = []

    for model_name, metrics in results.items():
        comparison_data.append({
            'Modelo': model_name,
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'MSE': metrics['mse'],
            'MAPE (%)': metrics['mape']
        })

    df_comparison = pd.DataFrame(comparison_data)

    # Ordenar por R² (decrescente)
    df_comparison = df_comparison.sort_values('R²', ascending=False)

    return df_comparison


def calculate_residuals_statistics(y_true: np.ndarray,
                                    y_pred: np.ndarray) -> Dict:
    """
    Calcula estatísticas detalhadas dos resíduos.

    Args:
        y_true: Valores reais
        y_pred: Valores preditos

    Returns:
        Dicionário com estatísticas dos resíduos
    """
    residuals = y_true - y_pred

    stats = {
        'mean': np.mean(residuals),
        'median': np.median(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'q25': np.percentile(residuals, 25),
        'q75': np.percentile(residuals, 75),
        'skewness': pd.Series(residuals).skew(),
        'kurtosis': pd.Series(residuals).kurtosis()
    }

    return stats


def identify_worst_predictions(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                X: np.ndarray = None,
                                top_n: int = 10,
                                feature_names: List[str] = None) -> pd.DataFrame:
    """
    Identifica as piores predições (failure analysis).

    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        X: Features (opcional)
        top_n: Número de piores casos a retornar
        feature_names: Nomes das features

    Returns:
        DataFrame com os piores casos
    """
    # Calcular erro absoluto
    abs_error = np.abs(y_true - y_pred)

    # Índices dos piores casos
    worst_indices = np.argsort(abs_error)[-top_n:][::-1]

    # Criar DataFrame
    worst_cases = pd.DataFrame({
        'Index': worst_indices,
        'True_Value': y_true[worst_indices],
        'Predicted_Value': y_pred[worst_indices],
        'Absolute_Error': abs_error[worst_indices],
        'Relative_Error (%)': (abs_error[worst_indices] / (y_true[worst_indices] + 1e-10)) * 100
    })

    # Adicionar features se disponíveis
    if X is not None and feature_names is not None:
        for i, feature_name in enumerate(feature_names):
            worst_cases[feature_name] = X[worst_indices, i]

    return worst_cases


def calculate_correlation_matrix(X: np.ndarray,
                                  y: np.ndarray,
                                  feature_names: List[str] = None) -> pd.DataFrame:
    """
    Calcula matriz de correlação entre features e target.

    Args:
        X: Features
        y: Target
        feature_names: Nomes das features

    Returns:
        DataFrame com matriz de correlação
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    # Criar DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y

    # Calcular correlação
    corr_matrix = df.corr()

    return corr_matrix


def calculate_feature_importance_permutation(model,
                                              X_test: np.ndarray,
                                              y_test: np.ndarray,
                                              feature_names: List[str] = None,
                                              n_repeats: int = 10,
                                              random_seed: int = 42) -> Dict[str, float]:
    """
    Calcula importância de features usando permutação.

    Args:
        model: Modelo treinado (deve ter método predict)
        X_test: Features de teste
        y_test: Target de teste
        feature_names: Nomes das features
        n_repeats: Número de repetições para cada permutação
        random_seed: Seed para reprodutibilidade

    Returns:
        Dicionário com importância de cada feature
    """
    np.random.seed(random_seed)

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]

    # Score baseline
    baseline_predictions = model.predict(X_test)
    baseline_score = r2_score(y_test, baseline_predictions)

    importances = {}

    for i, feature_name in enumerate(feature_names):
        scores = []

        for _ in range(n_repeats):
            # Copiar X_test
            X_permuted = X_test.copy()

            # Permutar feature i
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

            # Fazer predição
            permuted_predictions = model.predict(X_permuted)
            permuted_score = r2_score(y_test, permuted_predictions)

            # Importância = queda no score
            importance = baseline_score - permuted_score
            scores.append(importance)

        # Média das importâncias
        importances[feature_name] = np.mean(scores)

    return importances


def evaluate_model_comprehensive(model,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  model_name: str = "Modelo",
                                  feature_names: List[str] = None,
                                  verbose: bool = True) -> Dict:
    """
    Avaliação abrangente de um modelo.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        model_name: Nome do modelo
        feature_names: Nomes das features
        verbose: Se True, imprime resultados

    Returns:
        Dicionário com todos os resultados
    """
    # Fazer predições
    y_pred = model.predict(X_test)

    # Métricas principais
    metrics = calculate_metrics(y_test, y_pred, model_name)

    # Estatísticas de resíduos
    residuals_stats = calculate_residuals_statistics(y_test, y_pred)

    # Piores predições
    worst_cases = identify_worst_predictions(
        y_test, y_pred, X_test, top_n=10, feature_names=feature_names
    )

    # Importância de features (se possível)
    try:
        feature_importance = calculate_feature_importance_permutation(
            model, X_test, y_test, feature_names, n_repeats=5
        )
    except Exception as e:
        feature_importance = None
        if verbose:
            print(f"Não foi possível calcular importância de features: {e}")

    results = {
        'metrics': metrics,
        'residuals_stats': residuals_stats,
        'worst_cases': worst_cases,
        'feature_importance': feature_importance
    }

    if verbose:
        print_metrics(metrics, title=f"AVALIAÇÃO COMPLETA: {model_name}")

        print("\n" + "=" * 60)
        print("ESTATÍSTICAS DOS RESÍDUOS")
        print("=" * 60)
        for key, value in residuals_stats.items():
            print(f"  {key:12s}: {value:10.6f}")

        if feature_importance:
            print("\n" + "=" * 60)
            print("IMPORTÂNCIA DAS FEATURES (Permutação)")
            print("=" * 60)
            for feature, importance in sorted(feature_importance.items(),
                                              key=lambda x: x[1], reverse=True):
                print(f"  {feature:12s}: {importance:8.4f}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO MÓDULO DE MÉTRICAS")
    print("=" * 60)

    # Gerar dados de teste
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.rand(n_samples) * 0.4
    y_pred = y_true + np.random.normal(0, 0.05, n_samples)  # Predições com ruído

    # Teste 1: Métricas básicas
    print("\nTESTE 1: Métricas Básicas")
    metrics = calculate_metrics(y_true, y_pred, "Modelo Teste")
    print_metrics(metrics)

    # Teste 2: Estatísticas de resíduos
    print("\n" + "=" * 60)
    print("TESTE 2: Estatísticas de Resíduos")
    print("=" * 60)
    residuals_stats = calculate_residuals_statistics(y_true, y_pred)
    for key, value in residuals_stats.items():
        print(f"  {key:12s}: {value:10.6f}")

    # Teste 3: Piores predições
    print("\n" + "=" * 60)
    print("TESTE 3: Piores Predições (Top 5)")
    print("=" * 60)
    X_test = np.random.randn(n_samples, 4)
    worst = identify_worst_predictions(
        y_true, y_pred, X_test, top_n=5,
        feature_names=['GR', 'RHOB', 'DT', 'ILD']
    )
    print(worst)

    # Teste 4: Comparação de modelos
    print("\n" + "=" * 60)
    print("TESTE 4: Comparação de Modelos")
    print("=" * 60)

    y_pred_model2 = y_true + np.random.normal(0, 0.08, n_samples)
    y_pred_model3 = y_true + np.random.normal(0, 0.03, n_samples)

    results = {
        'Modelo 1': calculate_metrics(y_true, y_pred, "Modelo 1"),
        'Modelo 2': calculate_metrics(y_true, y_pred_model2, "Modelo 2"),
        'Modelo 3': calculate_metrics(y_true, y_pred_model3, "Modelo 3")
    }

    comparison = compare_models(results)
    print(comparison)

    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
