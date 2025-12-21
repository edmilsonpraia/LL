"""
Script de Teste de Instalação
Verifica se todos os módulos foram instalados corretamente

Execute: python test_installation.py
"""

import sys

print("=" * 80)
print("TESTE DE INSTALAÇÃO - PINNs em Petrofísica")
print("=" * 80)

errors = []
warnings = []

# ============================================================================
# 1. Verificar Python
# ============================================================================

print("\n[1/6] Verificando versão do Python...")
if sys.version_info < (3, 7):
    errors.append("Python 3.7+ é necessário")
else:
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} OK")

# ============================================================================
# 2. Verificar Dependências Principais
# ============================================================================

print("\n[2/6] Verificando dependências principais...")

dependencies = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'tensorflow': 'tensorflow',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn'
}

for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {package} OK")
    except ImportError:
        errors.append(f"{package} não instalado")
        print(f"✗ {package} FALTANDO")

# ============================================================================
# 3. Verificar Dependências Opcionais
# ============================================================================

print("\n[3/6] Verificando dependências opcionais...")

optional_deps = {
    'lasio': 'lasio (para arquivos LAS)',
    'shap': 'shap (para análise SHAP)',
}

for module, desc in optional_deps.items():
    try:
        __import__(module)
        print(f"✓ {desc} OK")
    except ImportError:
        warnings.append(f"{desc} não instalado (opcional)")
        print(f"⚠ {desc} FALTANDO (opcional)")

# ============================================================================
# 4. Verificar Módulos do Projeto
# ============================================================================

print("\n[4/6] Verificando módulos do projeto...")

try:
    from models import FisicaPuraModel, NNPuraModel, PINNModel
    print("✓ Módulo 'models' OK")
except ImportError as e:
    errors.append(f"Módulo 'models' com erro: {e}")
    print(f"✗ Módulo 'models' ERRO")

try:
    from utils import DataPreprocessor, generate_synthetic_data
    print("✓ Módulo 'utils' OK")
except ImportError as e:
    errors.append(f"Módulo 'utils' com erro: {e}")
    print(f"✗ Módulo 'utils' ERRO")

try:
    import config
    print("✓ Módulo 'config' OK")
except ImportError as e:
    errors.append(f"Módulo 'config' com erro: {e}")
    print(f"✗ Módulo 'config' ERRO")

# ============================================================================
# 5. Teste Rápido de Funcionalidade
# ============================================================================

print("\n[5/6] Testando funcionalidade básica...")

try:
    import numpy as np
    from models import FisicaPuraModel

    # Criar modelo
    model = FisicaPuraModel()

    # Dados de teste
    X_test = np.array([[60, 2.35, 70, 10]])

    # Predição
    prediction = model.predict(X_test)

    if 0 <= prediction[0] <= 1:
        print("✓ Modelo de Física Pura funcionando")
    else:
        warnings.append("Predição fora do range esperado")
        print("⚠ Modelo funcionando mas predição suspeita")

except Exception as e:
    errors.append(f"Erro ao testar modelo: {e}")
    print(f"✗ Erro no teste: {e}")

# ============================================================================
# 6. Verificar Estrutura de Diretórios
# ============================================================================

print("\n[6/6] Verificando estrutura de diretórios...")

import os

required_dirs = [
    'models',
    'models/saved',
    'utils',
    'data',
    'results',
    'notebooks'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"✓ {dir_path}/ existe")
    else:
        warnings.append(f"Diretório {dir_path}/ não encontrado")
        print(f"⚠ {dir_path}/ FALTANDO")

# ============================================================================
# Resumo Final
# ============================================================================

print("\n" + "=" * 80)
print("RESUMO DO TESTE")
print("=" * 80)

if not errors and not warnings:
    print("\n✅ TUDO OK! Instalação completa e funcional.")
    print("\nPróximos passos:")
    print("  1. Execute: python example_quickstart.py")
    print("  2. Ou execute: python train.py --data-source synthetic")

elif errors:
    print("\n❌ ERROS ENCONTRADOS:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")

    print("\n🔧 SOLUÇÃO:")
    print("  Execute: pip install -r requirements.txt")

elif warnings and not errors:
    print("\n⚠️ AVISOS (não críticos):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")

    print("\n✓ Sistema funcional, mas alguns recursos opcionais estão faltando.")
    print("  Instale dependências opcionais com: pip install lasio shap")

print("\n" + "=" * 80)
