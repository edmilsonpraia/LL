"""
Modelo de Rede Neural Pura (Data-Driven)
Arquitetura: 4 camadas ocultas [128, 64, 32, 16] neurônios
Ativação: Sigmoid
Dropout: 0.25

Autor: Edmilson Delfim Praia
Co-autor: Cirilo Cauxeiro
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Dict, Tuple, Optional, List
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HIDDEN_LAYERS, ACTIVATION, DROPOUT_RATE, LEARNING_RATE,
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE, EARLY_STOP_MONITOR,
    RESTORE_BEST_WEIGHTS, LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE, LR_MIN
)


class NNPuraModel:
    """
    Modelo de Rede Neural Pura para previsão de porosidade.

    Características:
    - Arquitetura densa profunda
    - Sem restrições físicas
    - Aprendizado puramente orientado a dados
    - Regularização via Dropout
    """

    def __init__(self,
                 input_dim: int = 4,
                 hidden_layers: List[int] = None,
                 activation: str = ACTIVATION,
                 dropout_rate: float = DROPOUT_RATE,
                 learning_rate: float = LEARNING_RATE):
        """
        Inicializa o modelo de Rede Neural Pura.

        Args:
            input_dim: Número de features de entrada
            hidden_layers: Lista com número de neurônios por camada oculta
            activation: Função de ativação
            dropout_rate: Taxa de dropout para regularização
            learning_rate: Taxa de aprendizado inicial
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers if hidden_layers else HIDDEN_LAYERS
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = None
        self.history = None
        self.model_name = "NN Pura"

    def build_model(self) -> keras.Model:
        """
        Constrói a arquitetura da rede neural.

        Returns:
            Modelo Keras compilado
        """
        model = models.Sequential(name="NN_Pura")

        # Camada de entrada
        model.add(layers.Input(shape=(self.input_dim,), name='input'))

        # Camadas ocultas com Dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='glorot_uniform',
                name=f'hidden_{i+1}'
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))

        # Camada de saída (regressão)
        model.add(layers.Dense(1, activation='linear', name='output'))

        # Compilar modelo
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        return model

    def get_callbacks(self,
                      early_stop: bool = True,
                      reduce_lr: bool = True,
                      verbose: int = 1) -> List[callbacks.Callback]:
        """
        Retorna lista de callbacks para treinamento.

        Args:
            early_stop: Se True, adiciona Early Stopping
            reduce_lr: Se True, adiciona ReduceLROnPlateau
            verbose: Nível de verbosidade

        Returns:
            Lista de callbacks
        """
        callback_list = []

        if early_stop:
            early_stopping = callbacks.EarlyStopping(
                monitor=EARLY_STOP_MONITOR,
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=RESTORE_BEST_WEIGHTS,
                verbose=verbose
            )
            callback_list.append(early_stopping)

        if reduce_lr:
            lr_scheduler = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=LR_REDUCE_FACTOR,
                patience=LR_REDUCE_PATIENCE,
                min_lr=LR_MIN,
                verbose=verbose
            )
            callback_list.append(lr_scheduler)

        return callback_list

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            batch_size: int = BATCH_SIZE,
            epochs: int = MAX_EPOCHS,
            verbose: int = 1,
            use_callbacks: bool = True) -> keras.callbacks.History:
        """
        Treina o modelo.

        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            batch_size: Tamanho do batch
            epochs: Número máximo de épocas
            verbose: Nível de verbosidade
            use_callbacks: Se True, usa callbacks

        Returns:
            Histórico de treinamento
        """
        if self.model is None:
            self.build_model()

        # Preparar dados de validação
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Callbacks
        callback_list = self.get_callbacks(verbose=verbose) if use_callbacks else []

        # Treinar
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callback_list,
            verbose=verbose
        )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições.

        Args:
            X: Features de entrada

        Returns:
            Predições de porosidade
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 verbose: int = 0) -> Dict[str, float]:
        """
        Avalia o modelo.

        Args:
            X_test: Features de teste
            y_test: Target de teste
            verbose: Nível de verbosidade

        Returns:
            Dicionário com métricas
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        results = self.model.evaluate(X_test, y_test, verbose=verbose)

        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mse': results[2]
        }

        return metrics

    def save_model(self, filepath: str):
        """Salva o modelo."""
        if self.model is None:
            raise ValueError("Modelo não foi criado.")

        self.model.save(filepath)
        print(f"Modelo salvo em: {filepath}")

    def load_model(self, filepath: str):
        """Carrega o modelo."""
        self.model = keras.models.load_model(filepath)
        print(f"Modelo carregado de: {filepath}")

    def get_model_summary(self):
        """Imprime resumo do modelo."""
        if self.model is None:
            self.build_model()
        return self.model.summary()

    def get_params(self) -> Dict:
        """Retorna os parâmetros do modelo."""
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'model_name': self.model_name
        }

    def __repr__(self):
        return (f"NNPuraModel(input_dim={self.input_dim}, "
                f"hidden_layers={self.hidden_layers}, "
                f"activation='{self.activation}')")


if __name__ == "__main__":
    # Teste rápido do modelo
    print("=" * 60)
    print("TESTE DO MODELO DE REDE NEURAL PURA")
    print("=" * 60)

    # Criar dados sintéticos
    np.random.seed(42)
    n_samples = 1000

    X_train = np.random.randn(n_samples, 4)
    y_train = np.random.rand(n_samples) * 0.4  # Porosidade entre 0 e 0.4

    X_val = np.random.randn(200, 4)
    y_val = np.random.rand(200) * 0.4

    # Criar e treinar modelo
    model = NNPuraModel()
    print(f"\n{model}")
    print(f"\nParâmetros: {model.get_params()}")

    print("\n" + "=" * 60)
    print("ARQUITETURA DO MODELO:")
    print("=" * 60)
    model.get_model_summary()

    print("\n" + "=" * 60)
    print("TREINANDO MODELO (5 épocas para teste)...")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        verbose=1
    )

    # Testar predição
    X_test = np.random.randn(10, 4)
    predictions = model.predict(X_test)

    print("\n" + "=" * 60)
    print("PREDIÇÕES DE TESTE:")
    print("=" * 60)
    print(f"Shape dos dados de entrada: {X_test.shape}")
    print(f"Predições: {predictions}")
    print(f"Porosidade média predita: {predictions.mean():.3f}")

    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
