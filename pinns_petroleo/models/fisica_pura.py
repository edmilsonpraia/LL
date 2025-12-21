"""
Modelo de Física Pura - Equação de Densidade
Implementa a equação de densidade para previsão de porosidade

Equação: φ = (ρ_ma - ρ_b) / (ρ_ma - ρ_fl)

Autor: Edmilson Delfim Praia
Co-autor: Cirilo Cauxeiro
"""

import numpy as np
from typing import Dict, Tuple
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RHO_MA, RHO_FL, DT_MA, DT_FL


class FisicaPuraModel:
    """
    Modelo baseado em equações físicas de rocha para previsão de porosidade.

    Implementa três equações principais:
    1. Equação de Densidade
    2. Equação de Wyllie (Sônico)
    3. Equação de Gardner
    """

    def __init__(self,
                 rho_ma: float = RHO_MA,
                 rho_fl: float = RHO_FL,
                 dt_ma: float = DT_MA,
                 dt_fl: float = DT_FL):
        """
        Inicializa o modelo de Física Pura.

        Args:
            rho_ma: Densidade da matriz (g/cm³)
            rho_fl: Densidade do fluido (g/cm³)
            dt_ma: Tempo de trânsito na matriz (μs/ft)
            dt_fl: Tempo de trânsito no fluido (μs/ft)
        """
        self.rho_ma = rho_ma
        self.rho_fl = rho_fl
        self.dt_ma = dt_ma
        self.dt_fl = dt_fl

        self.model_name = "Física Pura (Densidade)"

    def densidade_porosity(self, rhob: np.ndarray) -> np.ndarray:
        """
        Calcula porosidade usando a Equação de Densidade.

        φ = (ρ_ma - ρ_b) / (ρ_ma - ρ_fl)

        Args:
            rhob: Array com valores de densidade da formação (g/cm³)

        Returns:
            Array com valores de porosidade (fração v/v)
        """
        phi = (self.rho_ma - rhob) / (self.rho_ma - self.rho_fl)

        # Limitar valores entre 0 e 1
        phi = np.clip(phi, 0.0, 1.0)

        return phi

    def wyllie_porosity(self, dt: np.ndarray) -> np.ndarray:
        """
        Calcula porosidade usando a Equação de Wyllie (Tempo de Trânsito).

        φ = (Δt - Δt_ma) / (Δt_fl - Δt_ma)

        Args:
            dt: Array com valores de tempo de trânsito sônico (μs/ft)

        Returns:
            Array com valores de porosidade (fração v/v)
        """
        phi = (dt - self.dt_ma) / (self.dt_fl - self.dt_ma)

        # Limitar valores entre 0 e 1
        phi = np.clip(phi, 0.0, 1.0)

        return phi

    def gardner_density(self, vp: np.ndarray, a: float = 0.23, b: float = 0.25) -> np.ndarray:
        """
        Calcula densidade usando a Equação de Gardner.

        ρ_b = a * V_p^b

        Args:
            vp: Velocidade da onda compressional (km/s)
            a: Constante empírica (padrão: 0.23)
            b: Constante empírica (padrão: 0.25)

        Returns:
            Array com valores de densidade (g/cm³)
        """
        rho_b = a * (vp ** b)
        return rho_b

    def predict(self, X: np.ndarray, method: str = 'densidade') -> np.ndarray:
        """
        Prediz porosidade baseado nos dados de entrada.

        Args:
            X: Array com features [GR, RHOB, DT, ILD]
            method: Método de cálculo ('densidade', 'wyllie', 'combinado')

        Returns:
            Array com predições de porosidade
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Extrair features relevantes
        # Assumindo ordem: GR, RHOB, DT, ILD
        rhob = X[:, 1]  # Densidade
        dt = X[:, 2]    # Sônico

        if method == 'densidade':
            return self.densidade_porosity(rhob)

        elif method == 'wyllie':
            return self.wyllie_porosity(dt)

        elif method == 'combinado':
            # Média ponderada (densidade tem mais peso)
            phi_den = self.densidade_porosity(rhob)
            phi_wyl = self.wyllie_porosity(dt)
            return 0.7 * phi_den + 0.3 * phi_wyl

        else:
            raise ValueError(f"Método '{method}' não reconhecido. Use 'densidade', 'wyllie' ou 'combinado'.")

    def get_params(self) -> Dict:
        """Retorna os parâmetros do modelo."""
        return {
            'rho_ma': self.rho_ma,
            'rho_fl': self.rho_fl,
            'dt_ma': self.dt_ma,
            'dt_fl': self.dt_fl,
            'model_name': self.model_name
        }

    def set_params(self, **params):
        """Define os parâmetros do modelo."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def __repr__(self):
        return f"FisicaPuraModel(rho_ma={self.rho_ma}, rho_fl={self.rho_fl})"


def calcular_correcao_argila(phi_densidade: np.ndarray,
                             gr: np.ndarray,
                             gr_clean: float = 30.0,
                             gr_clay: float = 120.0) -> np.ndarray:
    """
    Aplica correção de argila na porosidade calculada.

    Args:
        phi_densidade: Porosidade calculada pela equação de densidade
        gr: Valores de raios gama (API)
        gr_clean: GR de arenito limpo (API)
        gr_clay: GR de folhelho puro (API)

    Returns:
        Porosidade corrigida para argila
    """
    # Calcular volume de argila (V_cl)
    # Usando a equação linear de Larionov
    igr = (gr - gr_clean) / (gr_clay - gr_clean)
    igr = np.clip(igr, 0.0, 1.0)

    # Larionov (rochas terciárias não consolidadas)
    vcl = 0.083 * (2 ** (3.7 * igr) - 1.0)
    vcl = np.clip(vcl, 0.0, 1.0)

    # Correção de porosidade
    # Assumindo porosidade de argila = 0.25
    phi_clay = 0.25
    phi_corrigida = phi_densidade - vcl * phi_clay

    return np.clip(phi_corrigida, 0.0, 1.0)


if __name__ == "__main__":
    # Teste rápido do modelo
    print("=" * 60)
    print("TESTE DO MODELO DE FÍSICA PURA")
    print("=" * 60)

    # Criar modelo
    model = FisicaPuraModel()
    print(f"\n{model}")
    print(f"\nParâmetros: {model.get_params()}")

    # Dados de teste
    # [GR, RHOB, DT, ILD]
    X_test = np.array([
        [60, 2.35, 70, 10],   # Arenito com boa porosidade
        [90, 2.55, 55, 5],    # Folhelho
        [50, 2.20, 85, 20],   # Arenito poroso
    ])

    print("\n" + "=" * 60)
    print("DADOS DE TESTE:")
    print("=" * 60)
    print("Input: [GR, RHOB, DT, ILD]")
    print(X_test)

    # Testar diferentes métodos
    for method in ['densidade', 'wyllie', 'combinado']:
        pred = model.predict(X_test, method=method)
        print(f"\n{method.upper()}:")
        print(f"Porosidade predita: {pred}")
        print(f"Porosidade média: {pred.mean():.3f}")

    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
