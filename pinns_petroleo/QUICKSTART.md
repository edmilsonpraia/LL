# Guia de Início Rápido

## 🚀 Começando em 5 Minutos

### Passo 1: Instalar Dependências

```bash
cd c:\Users\user\Desktop\PINNs\pinns_petroleo
pip install -r requirements.txt
```

### Passo 2: Executar Exemplo Rápido

```bash
python example_quickstart.py
```

Este script irá:
- Gerar dados sintéticos
- Treinar os 3 modelos (Física Pura, NN Pura, PINN)
- Comparar resultados
- Gerar visualizações

**Tempo estimado:** 2-3 minutos

---

## 📊 Treinar com Dados Sintéticos Completos

```bash
python train.py --data-source synthetic --n-samples 10000
```

**Saída:**
- Modelos treinados salvos em `models/saved/`
- Gráficos e análises em `results/experiment_YYYYMMDD_HHMMSS/`
- Métricas em `results_summary.json`

---

## 📁 Usar Seus Próprios Dados

### Formato CSV

Crie um arquivo CSV com as colunas:

```csv
GR,RHOB,DT,ILD,NPHI
65.2,2.35,72.5,15.3,0.25
89.1,2.58,55.2,8.7,0.12
...
```

Execute:

```bash
python train.py --data-source csv --data-path "data/seus_dados.csv"
```

### Formato LAS

```bash
python train.py --data-source las --data-path "data/poco_euler.las"
```

---

## 🐍 Uso em Código Python

### Exemplo Mínimo

```python
from models import NNPuraModel
from utils import generate_synthetic_data, DataPreprocessor

# 1. Dados
df = generate_synthetic_data(n_samples=5000)

# 2. Preprocessar
prep = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test, *_ = \
    prep.prepare_data_from_dataframe(df)

# 3. Treinar
model = NNPuraModel(input_dim=4)
model.fit(X_train, y_train, X_val, y_val, epochs=100)

# 4. Prever
predictions = model.predict(X_test)

# 5. Avaliar
from utils import calculate_metrics
metrics = calculate_metrics(y_test, predictions, "Meu Modelo")
print(f"R² = {metrics['r2']:.4f}")
```

---

## 📈 Testar Diferentes Hiperparâmetros

### PINN com Lambda Diferente

```python
from models import PINNModel

# Lambda baixo (mais dados, menos física)
model_low = PINNModel(lambda_physics=0.1)

# Lambda alto (mais física, menos dados)
model_high = PINNModel(lambda_physics=10.0)
```

### NN com Arquitetura Diferente

```python
from models import NNPuraModel

model = NNPuraModel(
    input_dim=4,
    hidden_layers=[256, 128, 64, 32],  # Mais neurônios
    dropout_rate=0.3,                   # Mais regularização
    learning_rate=0.0005                # LR menor
)
```

---

## 🎨 Visualizações Rápidas

```python
from utils import plot_scatter_prediction, plot_learning_curves

# Scatter plot
plot_scatter_prediction(y_test, predictions, "Meu Modelo",
                        r2=0.95, rmse=0.03)

# Curvas de aprendizado
plot_learning_curves(history, "Meu Modelo")
```

---

## ⚡ Dicas de Performance

### Treinar Mais Rápido

1. **Usar GPU**: TensorFlow detecta automaticamente
2. **Reduzir épocas para teste**: `epochs=50`
3. **Aumentar batch size**: `batch_size=64`

### Melhorar Resultados

1. **Mais dados**: `n_samples=20000`
2. **Mais épocas**: `epochs=500`
3. **Early stopping** está ativo por padrão

---

## 🔧 Solução de Problemas Comuns

### Erro: "No module named 'lasio'"

```bash
pip install lasio
```

### Erro: "CUDA out of memory"

Edite `config.py`:
```python
BATCH_SIZE = 16  # Reduzir de 32
```

### Overfitting (val_loss aumenta)

Edite `config.py`:
```python
DROPOUT_RATE = 0.4  # Aumentar regularização
EARLY_STOP_PATIENCE = 15  # Parar mais cedo
```

---

## 📚 Próximos Passos

1. ✅ Execute `example_quickstart.py`
2. ✅ Leia o `README.md` completo
3. ✅ Explore os notebooks em `notebooks/`
4. ✅ Teste com seus próprios dados
5. ✅ Ajuste hiperparâmetros em `config.py`
6. ✅ Experimente diferentes λ para PINN

---

## 💡 Recursos Adicionais

- **README.md**: Documentação completa
- **config.py**: Todas as configurações
- **Notebooks**: Análises detalhadas
- **models/**: Código dos modelos
- **utils/**: Utilitários e visualizações

---

**Dúvidas?** Consulte o README.md ou os comentários no código!
