# PINNs em Petrofísica

## Implementação de Physics-Informed Neural Networks para Previsão de Porosidade

**Autores:**
- Edmilson Delfim Praia
- Cirilo Cauxeiro

---

## 📋 Descrição do Projeto

Este projeto implementa três abordagens diferentes para previsão de porosidade em formações petrolíferas:

1. **Física Pura**: Modelo baseado em equações físicas clássicas (Equação de Densidade, Wyllie, Gardner)
2. **NN Pura**: Rede Neural profunda puramente orientada a dados
3. **PINN**: Physics-Informed Neural Network que combina dados e física

O objetivo é comparar essas abordagens e demonstrar quando cada uma é mais apropriada.

---

## 🏗️ Estrutura do Projeto

```
pinns_petroleo/
│
├── models/                    # Modelos de ML
│   ├── __init__.py
│   ├── fisica_pura.py        # Modelo de física pura
│   ├── nn_pura.py            # Rede Neural pura
│   ├── pinn.py               # Physics-Informed NN
│   └── saved/                # Modelos treinados salvos
│
├── utils/                     # Utilitários
│   ├── __init__.py
│   ├── data_preprocessing.py # Pré-processamento de dados
│   ├── metrics.py            # Métricas de avaliação
│   └── visualizations.py     # Funções de visualização
│
├── data/                      # Dados
│   ├── synthetic/            # Dados sintéticos gerados
│   └── real/                 # Dados reais (LAS, CSV)
│
├── results/                   # Resultados de experimentos
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── scatter_*.png
│       ├── learning_curves_*.png
│       ├── residuals_*.png
│       ├── model_comparison.csv
│       └── results_summary.json
│
├── notebooks/                 # Jupyter Notebooks
│   └── exemplo_uso.ipynb
│
├── config.py                  # Configurações do projeto
├── train.py                   # Script de treinamento principal
├── requirements.txt           # Dependências
└── README.md                  # Este arquivo
```

---

## 🚀 Instalação

### 1. Clonar o Repositório

```bash
cd c:\Users\user\Desktop\PINNs\pinns_petroleo
```

### 2. Criar Ambiente Virtual (Recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## 💻 Uso Rápido

### Treinamento com Dados Sintéticos

```bash
python train.py --data-source synthetic --n-samples 10000
```

### Treinamento com Dados CSV

```bash
python train.py --data-source csv --data-path "data/real/well_data.csv"
```

### Treinamento com Dados LAS

```bash
python train.py --data-source las --data-path "data/real/well_euler.las"
```

### Modo Silencioso

```bash
python train.py --data-source synthetic --quiet
```

---

## 📊 Exemplo de Uso em Python

```python
from models import FisicaPuraModel, NNPuraModel, PINNModel
from utils import DataPreprocessor, generate_synthetic_data
from utils import calculate_metrics, plot_scatter_prediction

# 1. Gerar dados sintéticos
df = generate_synthetic_data(n_samples=5000)

# 2. Preprocessar
preprocessor = DataPreprocessor()
result = preprocessor.prepare_data_from_dataframe(df)
X_train_norm, X_val_norm, X_test_norm, \
y_train_norm, y_val_norm, y_test_norm, \
X_train, X_val, X_test, \
y_train, y_val, y_test = result

# 3. Treinar Modelo de Física Pura
model_fisica = FisicaPuraModel()
y_pred_fisica = model_fisica.predict(X_test, method='densidade')
metrics_fisica = calculate_metrics(y_test, y_pred_fisica, "Física Pura")
print(f"R² Física Pura: {metrics_fisica['r2']:.4f}")

# 4. Treinar NN Pura
model_nn = NNPuraModel(input_dim=4)
model_nn.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm, epochs=100)
y_pred_nn_norm = model_nn.predict(X_test_norm)
y_pred_nn = preprocessor.inverse_transform_y(y_pred_nn_norm)
metrics_nn = calculate_metrics(y_test, y_pred_nn, "NN Pura")
print(f"R² NN Pura: {metrics_nn['r2']:.4f}")

# 5. Treinar PINN
model_pinn = PINNModel(input_dim=4, lambda_physics=1.0)
model_pinn.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm, epochs=100)
y_pred_pinn_norm = model_pinn.predict(X_test_norm)
y_pred_pinn = preprocessor.inverse_transform_y(y_pred_pinn_norm)
metrics_pinn = calculate_metrics(y_test, y_pred_pinn, "PINN")
print(f"R² PINN: {metrics_pinn['r2']:.4f}")

# 6. Visualizar
plot_scatter_prediction(y_test, y_pred_nn, "NN Pura",
                        r2=metrics_nn['r2'], rmse=metrics_nn['rmse'])
```

---

## 🔬 Modelos Implementados

### 1. Física Pura

Implementa três equações fundamentais:

**Equação de Densidade:**
```
φ = (ρ_ma - ρ_b) / (ρ_ma - ρ_fl)
```

**Equação de Wyllie (Sônico):**
```
φ = (Δt - Δt_ma) / (Δt_fl - Δt_ma)
```

**Equação de Gardner:**
```
ρ_b = a * V_p^b
```

**Uso:**
```python
from models import FisicaPuraModel

model = FisicaPuraModel(rho_ma=2.65, rho_fl=1.0)
porosity = model.predict(X, method='densidade')
```

### 2. NN Pura

Rede neural profunda com:
- 4 camadas ocultas: [128, 64, 32, 16] neurônios
- Ativação: Sigmoid
- Dropout: 0.25
- Otimizador: Adam
- Early Stopping e ReduceLROnPlateau

**Uso:**
```python
from models import NNPuraModel

model = NNPuraModel(input_dim=4)
model.fit(X_train, y_train, X_val, y_val, epochs=500)
predictions = model.predict(X_test)
```

### 3. PINN

Mesma arquitetura da NN Pura, mas com função de perda híbrida:

```
L_total = L_dados + λ * L_fisica

onde:
L_dados = MSE(y_true, y_pred)
L_fisica = MSE(y_pred, φ_densidade)
```

**Uso:**
```python
from models import PINNModel

model = PINNModel(input_dim=4, lambda_physics=1.0)
history = model.fit(X_train, y_train, X_val, y_val, epochs=500)
predictions = model.predict(X_test)
```

---

## 📈 Métricas de Avaliação

O projeto calcula as seguintes métricas:

- **R²** (Coeficiente de Determinação): Mede a proporção da variância explicada
- **RMSE** (Root Mean Squared Error): Erro quadrático médio
- **MAE** (Mean Absolute Error): Erro absoluto médio
- **MSE** (Mean Squared Error): Erro quadrático médio
- **MAPE** (Mean Absolute Percentage Error): Erro percentual absoluto médio

---

## 🎨 Visualizações

### Gráficos Disponíveis

1. **Scatter Plot (Predito vs Observado)**
```python
from utils import plot_scatter_prediction
plot_scatter_prediction(y_true, y_pred, "Modelo", r2=0.95, rmse=0.03)
```

2. **Curvas de Aprendizado**
```python
from utils import plot_learning_curves
plot_learning_curves(history.history, "NN Pura")
```

3. **Análise de Resíduos**
```python
from utils import plot_residuals_analysis
plot_residuals_analysis(y_true, y_pred, "NN Pura")
```

4. **Decomposição da Loss PINN**
```python
from utils import plot_pinn_loss_decomposition
plot_pinn_loss_decomposition(history)
```

5. **Importância de Features**
```python
from utils import plot_feature_importance
plot_feature_importance({'GR': 0.15, 'RHOB': 0.65, 'DT': 0.18, 'ILD': 0.02})
```

6. **Comparação de Modelos**
```python
from utils import plot_model_comparison_bar
plot_model_comparison_bar(comparison_df, metric='R²')
```

---

## ⚙️ Configuração

Edite `config.py` para ajustar:

### Parâmetros Físicos
```python
RHO_MA = 2.65  # Densidade da matriz (g/cm³)
RHO_FL = 1.0   # Densidade do fluido (g/cm³)
```

### Arquitetura da Rede
```python
HIDDEN_LAYERS = [128, 64, 32, 16]  # Neurônios por camada
ACTIVATION = 'sigmoid'              # Função de ativação
DROPOUT_RATE = 0.25                 # Taxa de dropout
```

### Treinamento
```python
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 500
EARLY_STOP_PATIENCE = 25
```

### PINN
```python
LAMBDA_PHYSICS_DEFAULT = 1.0  # Peso do termo de física
```

---

## 📝 Formato de Dados

### Features de Entrada (X)
- **GR**: Raios Gama (API units)
- **RHOB**: Densidade da formação (g/cm³)
- **DT**: Tempo de trânsito sônico (μs/ft)
- **ILD**: Resistividade profunda (ohm.m)

### Variável Alvo (y)
- **NPHI**: Porosidade neutrônica (fração v/v, 0-1)

### Exemplo de CSV

```csv
GR,RHOB,DT,ILD,NPHI
65.2,2.35,72.5,15.3,0.25
89.1,2.58,55.2,8.7,0.12
45.3,2.18,88.3,25.6,0.32
...
```

---

## 🧪 Testes

Cada módulo possui testes integrados. Execute:

```bash
# Testar modelo de Física Pura
python models/fisica_pura.py

# Testar NN Pura
python models/nn_pura.py

# Testar PINN
python models/pinn.py

# Testar pré-processamento
python utils/data_preprocessing.py

# Testar métricas
python utils/metrics.py

# Testar visualizações
python utils/visualizations.py
```

---

## 📚 Resultados Esperados

Com dados de alta qualidade (como os do Poço Euler no estudo original):

| Modelo | R² | RMSE | MAE |
|--------|-----|------|-----|
| Física Pura | ~0.85 | ~0.052 | ~0.041 |
| NN Pura | **~0.96** | **~0.027** | **~0.019** |
| PINN | ~0.89 | ~0.045 | ~0.035 |

**Conclusão Chave:** Quando os dados são abundantes e de alta qualidade, a NN Pura tende a superar tanto a Física Pura quanto a PINN.

---

## 🔍 Lições Aprendidas

1. **Qualidade > Quantidade**: Dados consistentes são mais importantes que volume
2. **RHOB é Crítico**: A densidade é a variável mais importante para porosidade
3. **Física Inadequada Prejudica**: Uma PINN com física simplista pode ter desempenho inferior a uma NN Pura
4. **Quando Usar PINN**: PINNs são valiosas quando dados são escassos MAS a física é robusta

---

## 🛠️ Troubleshooting

### Erro: "Module 'lasio' not found"
```bash
pip install lasio
```

### Erro: "CUDA out of memory"
Reduza o `BATCH_SIZE` em `config.py` ou use CPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Overfitting Detectado
- Aumente `DROPOUT_RATE`
- Reduza número de épocas
- Aumente tamanho do conjunto de treino

### Underfitting Detectado
- Aumente `MAX_EPOCHS`
- Aumente número de neurônios
- Reduza `DROPOUT_RATE`

---

## 📖 Referências

1. Raissi, M., et al. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
2. Wyllie, M. R. J., et al. (1956). Elastic wave velocities in heterogeneous media. *Geophysics*, 21(1), 41-70.
3. Gardner, G. H. F., et al. (1974). Formation velocity and density. *Geophysics*, 39(6), 770-780.
4. Mavko, G., et al. (2020). *The Rock Physics Handbook*. Cambridge University Press.

---

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais e de pesquisa.

---

## 👥 Contato

**Edmilson Delfim Praia**
**Cirilo Cauxeiro**

---

## 🙏 Agradecimentos

- Campo EDP (Angola) pelos dados de referência
- Comunidade de Deep Learning e Petrofísica

---

**Última atualização:** Dezembro 2025
