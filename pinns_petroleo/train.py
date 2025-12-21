"""
Script de Treinamento Principal
Treina e compara três modelos:
1. Física Pura (Equação de Densidade)
2. NN Pura (Data-Driven)
3. PINN (Physics-Informed)

Autor: Edmilson Delfim Praia
Co-autor: Cirilo Cauxeiro
"""

import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime

# Importar módulos do projeto
from models import FisicaPuraModel, NNPuraModel, PINNModel
from utils import (
    DataPreprocessor, generate_synthetic_data,
    calculate_metrics, print_metrics, compare_models,
    plot_scatter_prediction, plot_learning_curves,
    plot_pinn_loss_decomposition, plot_residuals_analysis,
    plot_feature_importance, plot_model_comparison_bar
)
from config import (
    INPUT_FEATURES, TARGET_FEATURE, RESULTS_DIR, MODELS_DIR,
    RANDOM_SEED, LAMBDA_PHYSICS_DEFAULT
)


def train_all_models(data_source: str = 'synthetic',
                     data_path: str = None,
                     n_samples: int = 10000,
                     verbose: bool = True):
    """
    Treina todos os modelos e gera análises comparativas.

    Args:
        data_source: 'synthetic', 'csv', ou 'las'
        data_path: Caminho para arquivo de dados (se não for synthetic)
        n_samples: Número de amostras (para dados sintéticos)
        verbose: Se True, imprime progresso
    """

    # Criar timestamp para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(RESULTS_DIR, f"experiment_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)

    if verbose:
        print("\n" + "=" * 80)
        print("TREINAMENTO DE MODELOS PINN PARA PREVISÃO DE POROSIDADE")
        print("=" * 80)
        print(f"Experimento: {timestamp}")
        print(f"Resultados serão salvos em: {results_folder}")

    # ========================================================================
    # 1. CARREGAR E PREPROCESSAR DADOS
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("ETAPA 1: CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS")
        print("=" * 80)

    preprocessor = DataPreprocessor()

    if data_source == 'synthetic':
        if verbose:
            print(f"\nGerando dados sintéticos ({n_samples} amostras)...")
        df = generate_synthetic_data(n_samples=n_samples, random_seed=RANDOM_SEED)

    elif data_source == 'csv':
        if verbose:
            print(f"\nCarregando dados de CSV: {data_path}")
        df = preprocessor.load_csv_file(data_path)

    elif data_source == 'las':
        if verbose:
            print(f"\nCarregando dados de LAS: {data_path}")
        df = preprocessor.load_las_file(data_path)

    else:
        raise ValueError(f"data_source inválido: {data_source}")

    # Preparar dados
    result = preprocessor.prepare_data_from_dataframe(
        df, normalize=True, quality_control=True
    )

    X_train_norm, X_val_norm, X_test_norm, \
    y_train_norm, y_val_norm, y_test_norm, \
    X_train, X_val, X_test, \
    y_train, y_val, y_test = result

    if verbose:
        print(f"\n✓ Dados preparados com sucesso!")
        print(f"  - Treino: {len(X_train)} amostras")
        print(f"  - Validação: {len(X_val)} amostras")
        print(f"  - Teste: {len(X_test)} amostras")

    # ========================================================================
    # 2. MODELO 1: FÍSICA PURA
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("ETAPA 2: TREINANDO MODELO DE FÍSICA PURA")
        print("=" * 80)

    model_fisica = FisicaPuraModel()
    y_pred_fisica = model_fisica.predict(X_test, method='densidade')

    metrics_fisica = calculate_metrics(y_test, y_pred_fisica, "Física Pura")

    if verbose:
        print_metrics(metrics_fisica, "RESULTADOS - FÍSICA PURA")

    # Salvar gráfico
    plot_scatter_prediction(
        y_test, y_pred_fisica,
        model_name="Física Pura (Densidade)",
        r2=metrics_fisica['r2'],
        rmse=metrics_fisica['rmse'],
        save_path=os.path.join(results_folder, 'scatter_fisica_pura.png'),
        show=False
    )

    # ========================================================================
    # 3. MODELO 2: NN PURA
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("ETAPA 3: TREINANDO MODELO DE REDE NEURAL PURA")
        print("=" * 80)

    model_nn = NNPuraModel(input_dim=len(INPUT_FEATURES))

    if verbose:
        print("\nArquitetura da NN:")
        model_nn.get_model_summary()

    if verbose:
        print("\nIniciando treinamento...")

    history_nn = model_nn.fit(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        verbose=1 if verbose else 0
    )

    # Predições (desnormalizar)
    y_pred_nn_norm = model_nn.predict(X_test_norm)
    y_pred_nn = preprocessor.inverse_transform_y(y_pred_nn_norm)

    metrics_nn = calculate_metrics(y_test, y_pred_nn, "NN Pura")

    if verbose:
        print_metrics(metrics_nn, "RESULTADOS - NN PURA")

    # Salvar modelo
    model_path_nn = os.path.join(MODELS_DIR, f'nn_pura_{timestamp}.h5')
    model_nn.save_model(model_path_nn)

    # Salvar gráficos
    plot_scatter_prediction(
        y_test, y_pred_nn,
        model_name="NN Pura",
        r2=metrics_nn['r2'],
        rmse=metrics_nn['rmse'],
        save_path=os.path.join(results_folder, 'scatter_nn_pura.png'),
        show=False
    )

    plot_learning_curves(
        history_nn.history,
        model_name="NN Pura",
        save_path=os.path.join(results_folder, 'learning_curves_nn.png'),
        show=False
    )

    plot_residuals_analysis(
        y_test, y_pred_nn,
        model_name="NN Pura",
        save_path=os.path.join(results_folder, 'residuals_nn.png'),
        show=False
    )

    # ========================================================================
    # 4. MODELO 3: PINN
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("ETAPA 4: TREINANDO MODELO PINN")
        print("=" * 80)

    model_pinn = PINNModel(
        input_dim=len(INPUT_FEATURES),
        lambda_physics=LAMBDA_PHYSICS_DEFAULT
    )

    if verbose:
        print(f"\nλ (lambda_physics) = {LAMBDA_PHYSICS_DEFAULT}")
        print("\nIniciando treinamento...")

    history_pinn = model_pinn.fit(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        verbose=1 if verbose else 0
    )

    # Predições (desnormalizar)
    y_pred_pinn_norm = model_pinn.predict(X_test_norm)
    y_pred_pinn = preprocessor.inverse_transform_y(y_pred_pinn_norm)

    metrics_pinn = calculate_metrics(y_test, y_pred_pinn, "PINN")

    if verbose:
        print_metrics(metrics_pinn, "RESULTADOS - PINN")

    # Salvar modelo
    model_path_pinn = os.path.join(MODELS_DIR, f'pinn_{timestamp}.h5')
    model_pinn.save_model(model_path_pinn)

    # Salvar gráficos
    plot_scatter_prediction(
        y_test, y_pred_pinn,
        model_name="PINN",
        r2=metrics_pinn['r2'],
        rmse=metrics_pinn['rmse'],
        save_path=os.path.join(results_folder, 'scatter_pinn.png'),
        show=False
    )

    plot_pinn_loss_decomposition(
        history_pinn,
        save_path=os.path.join(results_folder, 'pinn_loss_decomposition.png'),
        show=False
    )

    plot_residuals_analysis(
        y_test, y_pred_pinn,
        model_name="PINN",
        save_path=os.path.join(results_folder, 'residuals_pinn.png'),
        show=False
    )

    # ========================================================================
    # 5. COMPARAÇÃO DE MODELOS
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("ETAPA 5: COMPARAÇÃO FINAL DE MODELOS")
        print("=" * 80)

    results = {
        'Física Pura': metrics_fisica,
        'NN Pura': metrics_nn,
        'PINN': metrics_pinn
    }

    comparison_df = compare_models(results)

    if verbose:
        print("\n" + comparison_df.to_string(index=False))

    # Salvar comparação
    comparison_df.to_csv(
        os.path.join(results_folder, 'model_comparison.csv'),
        index=False
    )

    # Gráfico de comparação
    plot_model_comparison_bar(
        comparison_df,
        metric='R²',
        save_path=os.path.join(results_folder, 'comparison_r2.png'),
        show=False
    )

    plot_model_comparison_bar(
        comparison_df,
        metric='RMSE',
        save_path=os.path.join(results_folder, 'comparison_rmse.png'),
        show=False
    )

    # ========================================================================
    # 6. SALVAR RESULTADOS
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("ETAPA 6: SALVANDO RESULTADOS")
        print("=" * 80)

    # Salvar métricas em JSON
    import json

    results_summary = {
        'timestamp': timestamp,
        'data_source': data_source,
        'n_samples_train': len(X_train),
        'n_samples_val': len(X_val),
        'n_samples_test': len(X_test),
        'metrics': {
            'Física Pura': {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in metrics_fisica.items()},
            'NN Pura': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in metrics_nn.items()},
            'PINN': {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in metrics_pinn.items()}
        }
    }

    with open(os.path.join(results_folder, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)

    if verbose:
        print(f"\n✓ Resultados salvos em: {results_folder}")
        print(f"  - Modelos salvos em: {MODELS_DIR}")
        print(f"  - Gráficos e análises: {results_folder}")

    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)

    return results_summary, comparison_df


def main():
    """Função principal com argumentos de linha de comando."""

    parser = argparse.ArgumentParser(
        description='Treinar modelos PINN para previsão de porosidade'
    )

    parser.add_argument(
        '--data-source',
        type=str,
        default='synthetic',
        choices=['synthetic', 'csv', 'las'],
        help='Fonte de dados (synthetic, csv, ou las)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Caminho para arquivo de dados (CSV ou LAS)'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Número de amostras (para dados sintéticos)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso (menos output)'
    )

    args = parser.parse_args()

    # Executar treinamento
    train_all_models(
        data_source=args.data_source,
        data_path=args.data_path,
        n_samples=args.n_samples,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
