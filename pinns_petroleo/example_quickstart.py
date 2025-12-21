"""
Exemplo de Uso Rápido - PINNs em Petrofísica

Este script demonstra o uso básico dos modelos implementados.

Autor: Edmilson Delfim Praia
Co-autor: Cirilo Cauxeiro
"""

import numpy as np
import matplotlib.pyplot as plt

# Importar módulos do projeto
from models import FisicaPuraModel, NNPuraModel, PINNModel
from utils import (
    DataPreprocessor,
    generate_synthetic_data,
    calculate_metrics,
    print_metrics,
    compare_models,
    plot_scatter_prediction,
    plot_learning_curves
)

print("=" * 80)
print("EXEMPLO DE USO RÁPIDO - PINNs EM PETROFÍSICA")
print("=" * 80)

# ============================================================================
# 1. GERAR DADOS SINTÉTICOS
# ============================================================================

print("\n[1/5] Gerando dados sintéticos...")
df = generate_synthetic_data(n_samples=5000, noise_level=0.05, random_seed=42)

print(f"✓ Dados gerados: {df.shape}")
print(f"\nPrimeiras linhas:")
print(df.head())

# ============================================================================
# 2. PRÉ-PROCESSAR DADOS
# ============================================================================

print("\n[2/5] Pré-processando dados...")
preprocessor = DataPreprocessor()

result = preprocessor.prepare_data_from_dataframe(
    df, normalize=True, quality_control=True
)

X_train_norm, X_val_norm, X_test_norm, \
y_train_norm, y_val_norm, y_test_norm, \
X_train, X_val, X_test, \
y_train, y_val, y_test = result

print(f"✓ Dados preparados!")

# ============================================================================
# 3. TREINAR MODELO DE FÍSICA PURA
# ============================================================================

print("\n[3/5] Treinando Modelo de Física Pura...")

model_fisica = FisicaPuraModel()
y_pred_fisica = model_fisica.predict(X_test, method='densidade')

metrics_fisica = calculate_metrics(y_test, y_pred_fisica, "Física Pura")
print_metrics(metrics_fisica)

# ============================================================================
# 4. TREINAR REDE NEURAL PURA
# ============================================================================

print("\n[4/5] Treinando Rede Neural Pura...")

model_nn = NNPuraModel(input_dim=4)

print("\nArquitetura:")
model_nn.get_model_summary()

history_nn = model_nn.fit(
    X_train_norm, y_train_norm,
    X_val_norm, y_val_norm,
    epochs=50,  # Reduzido para exemplo rápido
    verbose=1
)

# Predições (desnormalizar)
y_pred_nn_norm = model_nn.predict(X_test_norm)
y_pred_nn = preprocessor.inverse_transform_y(y_pred_nn_norm)

metrics_nn = calculate_metrics(y_test, y_pred_nn, "NN Pura")
print_metrics(metrics_nn)

# ============================================================================
# 5. TREINAR PINN
# ============================================================================

print("\n[5/5] Treinando PINN...")

model_pinn = PINNModel(input_dim=4, lambda_physics=1.0)

history_pinn = model_pinn.fit(
    X_train_norm, y_train_norm,
    X_val_norm, y_val_norm,
    epochs=50,  # Reduzido para exemplo rápido
    verbose=1
)

# Predições (desnormalizar)
y_pred_pinn_norm = model_pinn.predict(X_test_norm)
y_pred_pinn = preprocessor.inverse_transform_y(y_pred_pinn_norm)

metrics_pinn = calculate_metrics(y_test, y_pred_pinn, "PINN")
print_metrics(metrics_pinn)

# ============================================================================
# 6. COMPARAR MODELOS
# ============================================================================

print("\n" + "=" * 80)
print("COMPARAÇÃO FINAL DE MODELOS")
print("=" * 80)

results = {
    'Física Pura': metrics_fisica,
    'NN Pura': metrics_nn,
    'PINN': metrics_pinn
}

comparison_df = compare_models(results)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 7. VISUALIZAÇÕES
# ============================================================================

print("\n" + "=" * 80)
print("GERANDO VISUALIZAÇÕES")
print("=" * 80)

# Scatter plots
print("\nGerando gráficos de dispersão...")
plot_scatter_prediction(
    y_test, y_pred_fisica,
    model_name="Física Pura",
    r2=metrics_fisica['r2'],
    rmse=metrics_fisica['rmse'],
    show=True
)

plot_scatter_prediction(
    y_test, y_pred_nn,
    model_name="NN Pura",
    r2=metrics_nn['r2'],
    rmse=metrics_nn['rmse'],
    show=True
)

plot_scatter_prediction(
    y_test, y_pred_pinn,
    model_name="PINN",
    r2=metrics_pinn['r2'],
    rmse=metrics_pinn['rmse'],
    show=True
)

# Curvas de aprendizado
print("\nGerando curvas de aprendizado...")
plot_learning_curves(history_nn.history, "NN Pura", show=True)

# ============================================================================
# 8. CONCLUSÃO
# ============================================================================

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO COM SUCESSO!")
print("=" * 80)

# Encontrar o melhor modelo
best_model = comparison_df.iloc[0]['Modelo']
best_r2 = comparison_df.iloc[0]['R²']

print(f"\n🏆 Melhor modelo: {best_model}")
print(f"   R² = {best_r2:.4f}")

print("\n" + "=" * 80)
print("PRÓXIMOS PASSOS:")
print("=" * 80)
print("1. Execute 'python train.py' para treinar com mais épocas")
print("2. Use seus próprios dados com --data-source csv --data-path seu_arquivo.csv")
print("3. Explore os notebooks em notebooks/")
print("4. Leia o README.md para mais informações")
print("=" * 80)
