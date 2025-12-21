"""
Utilitários para Pré-processamento de Dados
- Carregamento de dados LAS
- Limpeza e controle de qualidade
- Normalização e padronização
- Divisão de dados (treino/validação/teste)

Autor: Edmilson Delfim Praia
Co-autor: Cirilo Cauxeiro
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Optional, List
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INPUT_FEATURES, TARGET_FEATURE, TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    RANDOM_SEED, VALID_RANGES, N_STRATIFY_BINS
)


class DataPreprocessor:
    """
    Classe para pré-processamento de dados petrofísicos.
    """

    def __init__(self,
                 input_features: List[str] = None,
                 target_feature: str = TARGET_FEATURE,
                 valid_ranges: Dict = None,
                 random_seed: int = RANDOM_SEED):
        """
        Inicializa o preprocessador.

        Args:
            input_features: Lista de features de entrada
            target_feature: Nome da variável alvo
            valid_ranges: Dicionário com ranges válidos para cada feature
            random_seed: Seed para reprodutibilidade
        """
        self.input_features = input_features if input_features else INPUT_FEATURES
        self.target_feature = target_feature
        self.valid_ranges = valid_ranges if valid_ranges else VALID_RANGES
        self.random_seed = random_seed

        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        self.is_fitted = False

    def load_las_file(self, filepath: str) -> pd.DataFrame:
        """
        Carrega arquivo LAS (Log ASCII Standard).

        Args:
            filepath: Caminho para o arquivo LAS

        Returns:
            DataFrame com os dados do poço
        """
        try:
            import lasio
            las = lasio.read(filepath)
            df = las.df()
            df = df.reset_index()  # Depth como coluna

            print(f"Arquivo LAS carregado: {filepath}")
            print(f"Shape: {df.shape}")
            print(f"Colunas disponíveis: {list(df.columns)}")

            return df

        except ImportError:
            raise ImportError("Biblioteca 'lasio' não instalada. Execute: pip install lasio")

        except Exception as e:
            raise Exception(f"Erro ao carregar arquivo LAS: {str(e)}")

    def load_csv_file(self, filepath: str) -> pd.DataFrame:
        """
        Carrega arquivo CSV.

        Args:
            filepath: Caminho para o arquivo CSV

        Returns:
            DataFrame com os dados
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Arquivo CSV carregado: {filepath}")
            print(f"Shape: {df.shape}")
            print(f"Colunas disponíveis: {list(df.columns)}")
            return df

        except Exception as e:
            raise Exception(f"Erro ao carregar arquivo CSV: {str(e)}")

    def quality_control(self, df: pd.DataFrame,
                        remove_outliers: bool = True,
                        verbose: bool = True) -> pd.DataFrame:
        """
        Executa controle de qualidade nos dados.

        Args:
            df: DataFrame com os dados
            remove_outliers: Se True, remove outliers baseado em valid_ranges
            verbose: Se True, imprime estatísticas

        Returns:
            DataFrame limpo
        """
        df_clean = df.copy()
        n_original = len(df_clean)

        if verbose:
            print("\n" + "=" * 60)
            print("CONTROLE DE QUALIDADE")
            print("=" * 60)
            print(f"Número de amostras original: {n_original}")

        # 1. Remover valores NaN
        all_features = self.input_features + [self.target_feature]
        df_clean = df_clean.dropna(subset=all_features)

        if verbose:
            n_after_nan = len(df_clean)
            removed_nan = n_original - n_after_nan
            print(f"Removidos {removed_nan} amostras com valores NaN ({removed_nan/n_original*100:.2f}%)")

        # 2. Remover outliers baseado em ranges válidos
        if remove_outliers:
            for feature, (min_val, max_val) in self.valid_ranges.items():
                if feature in df_clean.columns:
                    before = len(df_clean)
                    df_clean = df_clean[
                        (df_clean[feature] >= min_val) &
                        (df_clean[feature] <= max_val)
                    ]
                    after = len(df_clean)
                    removed = before - after

                    if verbose and removed > 0:
                        print(f"{feature}: Removidos {removed} outliers "
                              f"(fora do range [{min_val}, {max_val}])")

        # 3. Estatísticas finais
        if verbose:
            n_final = len(df_clean)
            total_removed = n_original - n_final
            print(f"\nTotal removido: {total_removed} ({total_removed/n_original*100:.2f}%)")
            print(f"Amostras finais: {n_final}")
            print("=" * 60)

        return df_clean

    def get_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estatísticas descritivas dos dados.

        Args:
            df: DataFrame com os dados

        Returns:
            DataFrame com estatísticas
        """
        all_features = self.input_features + [self.target_feature]
        stats = df[all_features].describe()
        return stats

    def split_data(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   train_size: float = TRAIN_SIZE,
                   val_size: float = VAL_SIZE,
                   test_size: float = TEST_SIZE,
                   stratify: bool = True) -> Tuple:
        """
        Divide os dados em treino, validação e teste.

        Args:
            X: Features
            y: Target
            train_size: Proporção de treino
            val_size: Proporção de validação
            test_size: Proporção de teste
            stratify: Se True, usa estratificação baseada em bins de porosidade

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Verificar se as proporções somam 1
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("As proporções devem somar 1.0")

        # Estratificação
        stratify_array = None
        if stratify:
            # Criar bins para estratificação
            stratify_array = pd.cut(y, bins=N_STRATIFY_BINS, labels=False)

        # Primeira divisão: treino vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_size + test_size),
            stratify=stratify_array,
            random_state=self.random_seed
        )

        # Segunda divisão: val vs test
        relative_test_size = test_size / (val_size + test_size)

        if stratify:
            stratify_temp = pd.cut(y_temp, bins=N_STRATIFY_BINS, labels=False)
        else:
            stratify_temp = None

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test_size,
            stratify=stratify_temp,
            random_state=self.random_seed
        )

        print(f"\nDivisão de dados:")
        print(f"  Treino: {len(X_train)} amostras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validação: {len(X_val)} amostras ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Teste: {len(X_test)} amostras ({len(X_test)/len(X)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize_data(self,
                       X_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None,
                       X_test: Optional[np.ndarray] = None,
                       y_train: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None,
                       y_test: Optional[np.ndarray] = None) -> Tuple:
        """
        Normaliza os dados usando StandardScaler (X) e MinMaxScaler (y).

        IMPORTANTE: Ajusta os scalers apenas no conjunto de treino!

        Args:
            X_train, X_val, X_test: Features
            y_train, y_val, y_test: Target

        Returns:
            Tupla com dados normalizados
        """
        # Normalizar X (StandardScaler - média 0, std 1)
        X_train_norm = self.scaler_X.fit_transform(X_train)

        X_val_norm = None
        if X_val is not None:
            X_val_norm = self.scaler_X.transform(X_val)

        X_test_norm = None
        if X_test is not None:
            X_test_norm = self.scaler_X.transform(X_test)

        # Normalizar y (MinMaxScaler - entre 0 e 1)
        y_train_norm = None
        y_val_norm = None
        y_test_norm = None

        if y_train is not None:
            y_train_norm = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        if y_val is not None:
            y_val_norm = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        if y_test is not None:
            y_test_norm = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        self.is_fitted = True

        print("\nNormalização aplicada:")
        print(f"  X: StandardScaler (média=0, std=1)")
        print(f"  y: MinMaxScaler (min=0, max=1)")

        return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm

    def inverse_transform_y(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Reverte a normalização do target.

        Args:
            y_normalized: Valores normalizados

        Returns:
            Valores originais
        """
        if not self.is_fitted:
            raise ValueError("Scaler não foi ajustado. Execute normalize_data() primeiro.")

        y_original = self.scaler_y.inverse_transform(y_normalized.reshape(-1, 1)).flatten()
        return y_original

    def prepare_data_from_dataframe(self,
                                     df: pd.DataFrame,
                                     normalize: bool = True,
                                     quality_control: bool = True) -> Tuple:
        """
        Pipeline completo de preparação de dados a partir de um DataFrame.

        Args:
            df: DataFrame com os dados
            normalize: Se True, normaliza os dados
            quality_control: Se True, executa controle de qualidade

        Returns:
            (X_train_norm, X_val_norm, X_test_norm,
             y_train_norm, y_val_norm, y_test_norm,
             X_train, X_val, X_test,
             y_train, y_val, y_test)
        """
        # 1. Controle de qualidade
        if quality_control:
            df = self.quality_control(df, remove_outliers=True, verbose=True)

        # 2. Extrair features e target
        X = df[self.input_features].values
        y = df[self.target_feature].values

        # 3. Dividir dados
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, stratify=True
        )

        # 4. Normalizar (se solicitado)
        if normalize:
            X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = \
                self.normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)

            return (X_train_norm, X_val_norm, X_test_norm,
                    y_train_norm, y_val_norm, y_test_norm,
                    X_train, X_val, X_test,
                    y_train, y_val, y_test)
        else:
            return (X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    X_train, X_val, X_test,
                    y_train, y_val, y_test)


def generate_synthetic_data(n_samples: int = 10000,
                             noise_level: float = 0.05,
                             random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Gera dados sintéticos para testes.

    Args:
        n_samples: Número de amostras
        noise_level: Nível de ruído a adicionar
        random_seed: Seed para reprodutibilidade

    Returns:
        DataFrame com dados sintéticos
    """
    np.random.seed(random_seed)

    # Gerar features
    GR = np.random.uniform(20, 150, n_samples)  # Bimodal: arenito vs folhelho
    RHOB = np.random.uniform(2.0, 2.7, n_samples)
    DT = np.random.uniform(45, 120, n_samples)
    ILD = np.random.uniform(1, 200, n_samples)

    # Gerar porosidade baseada em física + ruído
    # Usando equação de densidade
    NPHI = (2.65 - RHOB) / (2.65 - 1.0)

    # Adicionar influência de GR (argilosidade reduz porosidade efetiva)
    vcl = (GR - 30) / (120 - 30)
    vcl = np.clip(vcl, 0, 1)
    NPHI = NPHI * (1 - 0.3 * vcl)  # Argila reduz porosidade efetiva

    # Adicionar ruído
    NPHI += np.random.normal(0, noise_level, n_samples)
    NPHI = np.clip(NPHI, 0.05, 0.45)

    # Criar DataFrame
    df = pd.DataFrame({
        'GR': GR,
        'RHOB': RHOB,
        'DT': DT,
        'ILD': ILD,
        'NPHI': NPHI
    })

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO MÓDULO DE PRÉ-PROCESSAMENTO")
    print("=" * 60)

    # Gerar dados sintéticos
    print("\nGerando dados sintéticos...")
    df = generate_synthetic_data(n_samples=5000)

    print(f"\nShape: {df.shape}")
    print(f"\nPrimeiras linhas:")
    print(df.head())

    print(f"\nEstatísticas descritivas:")
    print(df.describe())

    # Criar preprocessador
    preprocessor = DataPreprocessor()

    # Testar pipeline completo
    print("\n" + "=" * 60)
    print("TESTANDO PIPELINE COMPLETO")
    print("=" * 60)

    result = preprocessor.prepare_data_from_dataframe(
        df,
        normalize=True,
        quality_control=True
    )

    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, \
    X_train, X_val, X_test, y_train, y_val, y_test = result

    print(f"\nDados normalizados:")
    print(f"  X_train shape: {X_train_norm.shape}")
    print(f"  X_val shape: {X_val_norm.shape}")
    print(f"  X_test shape: {X_test_norm.shape}")

    print(f"\nPorosidade (normalizada):")
    print(f"  Treino - min: {y_train_norm.min():.3f}, max: {y_train_norm.max():.3f}")
    print(f"  Val - min: {y_val_norm.min():.3f}, max: {y_val_norm.max():.3f}")
    print(f"  Teste - min: {y_test_norm.min():.3f}, max: {y_test_norm.max():.3f}")

    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
