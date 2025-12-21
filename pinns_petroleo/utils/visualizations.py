"""
Funções de Visualização
- Gráficos de dispersão (Predicted vs Observed)
- Curvas de aprendizado
- Análise de resíduos
- Comparação de modelos
- Análise SHAP

Autor: Edmilson Delfim Praia
Co-autor: Cirilo Cauxeiro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURE_DPI, MODEL_COLORS

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = FIGURE_DPI
plt.rcParams['font.size'] = 10


def plot_scatter_prediction(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             model_name: str = "Modelo",
                             r2: float = None,
                             rmse: float = None,
                             save_path: Optional[str] = None,
                             show: bool = True):
    """
    Gráfico de dispersão: Predito vs Observado.

    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        model_name: Nome do modelo
        r2: Coeficiente R² (opcional)
        rmse: RMSE (opcional)
        save_path: Caminho para salvar figura
        show: Se True, exibe o gráfico
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.3, s=20, edgecolors='k', linewidths=0.5)

    # Linha de identidade (perfeição)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Linha de Identidade')

    # Configurações
    ax.set_xlabel('Porosidade Observada (v/v)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Porosidade Predita (v/v)', fontsize=12, fontweight='bold')
    ax.set_title(f'Modelo: {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Adicionar métricas no gráfico
    textstr = ''
    if r2 is not None:
        textstr += f'R² = {r2:.4f}\n'
    if rmse is not None:
        textstr += f'RMSE = {rmse:.4f}'

    if textstr:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curves(history: Dict,
                          model_name: str = "Modelo",
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    Plota curvas de aprendizado (loss vs epochs).

    Args:
        history: Dicionário com histórico de treinamento
        model_name: Nome do modelo
        save_path: Caminho para salvar
        show: Se True, exibe o gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loss de treino
    if 'loss' in history:
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], 'b-', label='Treino', linewidth=2)

    # Loss de validação
    if 'val_loss' in history:
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validação', linewidth=2)

    ax.set_xlabel('Épocas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title(f'Curvas de Aprendizado - {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_pinn_loss_decomposition(history: Dict,
                                  save_path: Optional[str] = None,
                                  show: bool = True):
    """
    Plota decomposição das perdas da PINN (dados + física).

    Args:
        history: Histórico com loss_data e loss_physics
        save_path: Caminho para salvar
        show: Se True, exibe
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['loss']) + 1)

    # Gráfico 1: Todas as perdas
    ax1.plot(epochs, history['loss'], 'k-', label='Loss Total', linewidth=2)
    ax1.plot(epochs, history['loss_data'], 'b-', label='Loss Dados', linewidth=2)
    ax1.plot(epochs, history['loss_physics'], 'r-', label='Loss Física', linewidth=2)

    ax1.set_xlabel('Épocas', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Decomposição da Loss - PINN', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Razão Loss Física / Loss Dados
    ratio = np.array(history['loss_physics']) / (np.array(history['loss_data']) + 1e-10)
    ax2.plot(epochs, ratio, 'g-', linewidth=2)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Equilíbrio (1:1)')

    ax2.set_xlabel('Épocas', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Razão: Loss Física / Loss Dados', fontsize=12, fontweight='bold')
    ax2.set_title('Balanceamento: Física vs Dados', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals_analysis(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             model_name: str = "Modelo",
                             save_path: Optional[str] = None,
                             show: bool = True):
    """
    Análise de resíduos com múltiplos gráficos.

    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        model_name: Nome do modelo
        save_path: Caminho para salvar
        show: Se True, exibe
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.3, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Valores Preditos', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Resíduos', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Resíduos vs Preditos', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Histogram of Residuals
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Resíduos', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequência', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Distribuição dos Resíduos', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normalidade)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Residuals over Index
    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.3, s=20)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Índice da Amostra', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Resíduos', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Resíduos ao Longo das Amostras', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Análise de Resíduos - {model_name}', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(importance_dict: Dict[str, float],
                             title: str = "Importância das Features",
                             save_path: Optional[str] = None,
                             show: bool = True):
    """
    Plota importância de features.

    Args:
        importance_dict: Dicionário {feature_name: importance_value}
        title: Título do gráfico
        save_path: Caminho para salvar
        show: Se True, exibe
    """
    # Ordenar por importância
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_features]
    importances = [item[1] for item in sorted_features]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot
    bars = ax.barh(features, importances, color='skyblue', edgecolor='black')

    # Colorir barras
    for i, bar in enumerate(bars):
        if importances[i] < 0:
            bar.set_color('salmon')

    ax.set_xlabel('Importância', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_correlation_matrix(corr_matrix: pd.DataFrame,
                             save_path: Optional[str] = None,
                             show: bool = True):
    """
    Plota matriz de correlação como heatmap.

    Args:
        corr_matrix: Matriz de correlação (DataFrame)
        save_path: Caminho para salvar
        show: Se True, exibe
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, fmt='.2f')

    ax.set_title('Matriz de Correlação', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison_bar(comparison_df: pd.DataFrame,
                               metric: str = 'R²',
                               save_path: Optional[str] = None,
                               show: bool = True):
    """
    Compara modelos usando gráfico de barras.

    Args:
        comparison_df: DataFrame com comparação de modelos
        metric: Métrica a visualizar
        save_path: Caminho para salvar
        show: Se True, exibe
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot
    bars = ax.bar(comparison_df['Modelo'], comparison_df[metric],
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)

    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Comparação de Modelos - {metric}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_lambda_sensitivity(lambda_values: List[float],
                             r2_values: List[float],
                             rmse_values: List[float],
                             save_path: Optional[str] = None,
                             show: bool = True):
    """
    Análise de sensibilidade do hiperparâmetro lambda.

    Args:
        lambda_values: Lista de valores de lambda testados
        r2_values: Lista de R² correspondentes
        rmse_values: Lista de RMSE correspondentes
        save_path: Caminho para salvar
        show: Se True, exibe
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # R² vs Lambda
    ax1.plot(lambda_values, r2_values, 'bo-', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('λ (Lambda Physics)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax1.set_title('R² vs Lambda', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # RMSE vs Lambda
    ax2.plot(lambda_values, rmse_values, 'ro-', linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('λ (Lambda Physics)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('RMSE vs Lambda', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_well_log_comparison(depth: np.ndarray,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model_name: str = "Modelo",
                              save_path: Optional[str] = None,
                              show: bool = True):
    """
    Compara perfil de porosidade real vs predito.

    Args:
        depth: Profundidade
        y_true: Porosidade real
        y_pred: Porosidade predita
        model_name: Nome do modelo
        save_path: Caminho para salvar
        show: Se True, exibe
    """
    fig, ax = plt.subplots(figsize=(6, 10))

    # Plotar perfis
    ax.plot(y_true, depth, 'k-', label='Real (NPHI)', linewidth=2)
    ax.plot(y_pred, depth, 'r-', label='Predito', linewidth=2, alpha=0.7)

    ax.set_xlabel('Porosidade (v/v)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profundidade (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Perfil de Porosidade - {model_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Inverter eixo Y (profundidade aumenta para baixo)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO MÓDULO DE VISUALIZAÇÕES")
    print("=" * 60)

    # Gerar dados de teste
    np.random.seed(42)
    n_samples = 500

    y_true = np.random.rand(n_samples) * 0.4
    y_pred = y_true + np.random.normal(0, 0.05, n_samples)

    # Teste 1: Scatter plot
    print("\nGerando scatter plot...")
    plot_scatter_prediction(y_true, y_pred, "NN Pura", r2=0.95, rmse=0.03, show=False)

    # Teste 2: Curvas de aprendizado
    print("Gerando curvas de aprendizado...")
    history = {
        'loss': [0.1, 0.08, 0.06, 0.05, 0.04, 0.03],
        'val_loss': [0.12, 0.09, 0.07, 0.06, 0.05, 0.04]
    }
    plot_learning_curves(history, "NN Pura", show=False)

    # Teste 3: Análise de resíduos
    print("Gerando análise de resíduos...")
    plot_residuals_analysis(y_true, y_pred, "NN Pura", show=False)

    # Teste 4: Importância de features
    print("Gerando gráfico de importância...")
    importance = {'GR': 0.15, 'RHOB': 0.65, 'DT': 0.18, 'ILD': 0.02}
    plot_feature_importance(importance, show=False)

    print("\n" + "=" * 60)
    print("TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
    print("=" * 60)
