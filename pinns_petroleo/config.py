"""
Configurações do Projeto PINNs em Petrofísica
Autor: Edmilson Delfim Praia
Co-autor: Cirilo Cauxeiro
"""

import os

# ============================================================================
# CONFIGURAÇÕES GERAIS DO PROJETO
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# PARÂMETROS FÍSICOS (PETROFÍSICA)
# ============================================================================

# Equação de Densidade
RHO_MA = 2.65  # Densidade da matriz (g/cm³) - típica para quartzo
RHO_FL = 1.0   # Densidade do fluido (g/cm³) - água

# Equação de Wyllie (Sônico)
DT_MA = 55.5   # Tempo de trânsito na matriz (μs/ft) - quartzo
DT_FL = 189.0  # Tempo de trânsito no fluido (μs/ft) - água

# ============================================================================
# CONFIGURAÇÕES DE DADOS
# ============================================================================

# Features de entrada
INPUT_FEATURES = ['GR', 'RHOB', 'DT', 'ILD']

# Variável alvo
TARGET_FEATURE = 'NPHI'

# Divisão de dados
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Seed para reprodutibilidade
RANDOM_SEED = 42

# Valores válidos (para filtragem de outliers)
VALID_RANGES = {
    'GR': (0, 200),      # API units
    'RHOB': (1.5, 3.0),  # g/cm³
    'DT': (40, 200),     # μs/ft
    'ILD': (0.1, 1000),  # ohm.m
    'NPHI': (0.0, 0.50)  # v/v (fração)
}

# ============================================================================
# ARQUITETURA DA REDE NEURAL
# ============================================================================

# Configuração das camadas (número de neurônios por camada oculta)
HIDDEN_LAYERS = [128, 64, 32, 16]

# Função de ativação
ACTIVATION = 'sigmoid'

# Dropout rate
DROPOUT_RATE = 0.25

# ============================================================================
# HIPERPARÂMETROS DE TREINAMENTO
# ============================================================================

# Otimizador
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'

# Treinamento
BATCH_SIZE = 32
MAX_EPOCHS = 500

# Early Stopping
EARLY_STOP_PATIENCE = 25
EARLY_STOP_MONITOR = 'val_loss'
RESTORE_BEST_WEIGHTS = True

# Learning Rate Scheduler
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 10
LR_MIN = 1e-7

# ============================================================================
# CONFIGURAÇÕES ESPECÍFICAS DA PINN
# ============================================================================

# Hiperparâmetro lambda (balanceamento entre dados e física)
LAMBDA_PHYSICS_DEFAULT = 1.0

# Range para análise de sensibilidade do lambda
LAMBDA_RANGE = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]

# ============================================================================
# CONFIGURAÇÕES DE AVALIAÇÃO
# ============================================================================

# Métricas a calcular
METRICS = ['mse', 'mae', 'r2', 'rmse']

# Número de bins para estratificação
N_STRATIFY_BINS = 10

# ============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# ============================================================================

# Estilo de gráficos
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Cores para diferentes modelos
MODEL_COLORS = {
    'fisica_pura': '#FF6B6B',
    'nn_pura': '#4ECDC4',
    'pinn': '#45B7D1'
}

# ============================================================================
# CONFIGURAÇÕES DE EXPORTAÇÃO
# ============================================================================

# Formato de exportação do modelo
EXPORT_FORMAT = 'onnx'
ONNX_OPSET = 13

# ============================================================================
# CONFIGURAÇÕES DA API (OPCIONAL)
# ============================================================================

API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = False

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
