"""
Modelo PINN (Physics-Informed Neural Network)
Combina aprendizado de dados com restrições físicas

Função de Perda: L(θ) = L_dados(θ) + λ * L_fisica(θ)

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
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE, LAMBDA_PHYSICS_DEFAULT,
    RHO_MA, RHO_FL
)


class PINNModel:
    """
    Physics-Informed Neural Network para previsão de porosidade.

    Características:
    - Mesma arquitetura da NN Pura
    - Função de perda híbrida (dados + física)
    - Integra Equação de Densidade como restrição
    """

    def __init__(self,
                 input_dim: int = 4,
                 hidden_layers: List[int] = None,
                 activation: str = ACTIVATION,
                 dropout_rate: float = DROPOUT_RATE,
                 learning_rate: float = LEARNING_RATE,
                 lambda_physics: float = LAMBDA_PHYSICS_DEFAULT,
                 rho_ma: float = RHO_MA,
                 rho_fl: float = RHO_FL):
        """
        Inicializa o modelo PINN.

        Args:
            input_dim: Número de features de entrada
            hidden_layers: Lista com número de neurônios por camada oculta
            activation: Função de ativação
            dropout_rate: Taxa de dropout
            learning_rate: Taxa de aprendizado
            lambda_physics: Peso do termo de física na função de perda
            rho_ma: Densidade da matriz (g/cm³)
            rho_fl: Densidade do fluido (g/cm³)
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers if hidden_layers else HIDDEN_LAYERS
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.lambda_physics = lambda_physics
        self.rho_ma = rho_ma
        self.rho_fl = rho_fl

        self.model = None
        self.history = None
        self.model_name = "PINN"

        # Para rastrear as perdas separadamente
        self.loss_data_history = []
        self.loss_physics_history = []

    def build_model(self) -> keras.Model:
        """
        Constrói a arquitetura da rede neural (idêntica à NN Pura).

        Returns:
            Modelo Keras
        """
        model = models.Sequential(name="PINN")

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

        # Camada de saída
        model.add(layers.Dense(1, activation='linear', name='output'))

        self.model = model
        return model

    def physics_loss(self, y_pred: tf.Tensor, rhob: tf.Tensor) -> tf.Tensor:
        """
        Calcula a perda baseada na física (Equação de Densidade).

        L_fisica = MSE(φ_NN - φ_densidade)

        onde: φ_densidade = (ρ_ma - ρ_b) / (ρ_ma - ρ_fl)

        Args:
            y_pred: Predições da rede neural (porosidade)
            rhob: Valores de densidade da formação

        Returns:
            Perda de física (escalar)
        """
        # Calcular porosidade pela física
        phi_physics = (self.rho_ma - rhob) / (self.rho_ma - self.rho_fl)

        # Garantir que está no intervalo válido
        phi_physics = tf.clip_by_value(phi_physics, 0.0, 1.0)

        # MSE entre predição da rede e física
        loss_physics = tf.reduce_mean(tf.square(y_pred - phi_physics))

        return loss_physics

    def pinn_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                  rhob: tf.Tensor) -> tf.Tensor:
        """
        Função de perda híbrida da PINN.

        L_total = L_dados + λ * L_fisica

        Args:
            y_true: Valores reais de porosidade
            y_pred: Predições da rede
            rhob: Densidade da formação

        Returns:
            Perda total
        """
        # Perda dos dados (MSE)
        loss_data = tf.reduce_mean(tf.square(y_true - y_pred))

        # Perda da física
        loss_physics = self.physics_loss(y_pred, rhob)

        # Perda total
        loss_total = loss_data + self.lambda_physics * loss_physics

        return loss_total, loss_data, loss_physics

    @tf.function
    def train_step(self, X: tf.Tensor, y: tf.Tensor,
                   optimizer: keras.optimizers.Optimizer) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Executa um passo de treinamento.

        Args:
            X: Features [GR, RHOB, DT, ILD]
            y: Target (porosidade real)
            optimizer: Otimizador

        Returns:
            (loss_total, loss_data, loss_physics)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model(X, training=True)

            # Extrair RHOB (índice 1)
            rhob = X[:, 1:2]

            # Calcular perda
            loss_total, loss_data, loss_physics = self.pinn_loss(y, y_pred, rhob)

        # Backward pass
        gradients = tape.gradient(loss_total, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss_total, loss_data, loss_physics

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            batch_size: int = BATCH_SIZE,
            epochs: int = MAX_EPOCHS,
            verbose: int = 1,
            early_stop_patience: int = EARLY_STOP_PATIENCE) -> Dict:
        """
        Treina o modelo PINN com loop de treinamento customizado.

        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação
            y_val: Target de validação
            batch_size: Tamanho do batch
            epochs: Número máximo de épocas
            verbose: Nível de verbosidade
            early_stop_patience: Paciência para early stopping

        Returns:
            Histórico de treinamento
        """
        if self.model is None:
            self.build_model()

        # Converter para tensores
        X_train_tf = tf.constant(X_train, dtype=tf.float32)
        y_train_tf = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32)

        if X_val is not None:
            X_val_tf = tf.constant(X_val, dtype=tf.float32)
            y_val_tf = tf.constant(y_val.reshape(-1, 1), dtype=tf.float32)

        # Otimizador
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Criar dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tf, y_train_tf))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)

        # Histórico
        history = {
            'loss': [],
            'loss_data': [],
            'loss_physics': [],
            'val_loss': [],
            'val_loss_data': [],
            'val_loss_physics': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        # Loop de treinamento
        for epoch in range(epochs):
            # Treinamento
            epoch_loss = []
            epoch_loss_data = []
            epoch_loss_physics = []

            for X_batch, y_batch in train_dataset:
                loss_total, loss_data, loss_physics = self.train_step(
                    X_batch, y_batch, optimizer
                )
                epoch_loss.append(loss_total.numpy())
                epoch_loss_data.append(loss_data.numpy())
                epoch_loss_physics.append(loss_physics.numpy())

            # Médias da época
            avg_loss = np.mean(epoch_loss)
            avg_loss_data = np.mean(epoch_loss_data)
            avg_loss_physics = np.mean(epoch_loss_physics)

            history['loss'].append(avg_loss)
            history['loss_data'].append(avg_loss_data)
            history['loss_physics'].append(avg_loss_physics)

            # Validação
            if X_val is not None:
                y_pred_val = self.model(X_val_tf, training=False)
                rhob_val = X_val_tf[:, 1:2]

                val_loss_total, val_loss_data, val_loss_physics = self.pinn_loss(
                    y_val_tf, y_pred_val, rhob_val
                )

                history['val_loss'].append(val_loss_total.numpy())
                history['val_loss_data'].append(val_loss_data.numpy())
                history['val_loss_physics'].append(val_loss_physics.numpy())

                # Early stopping
                if val_loss_total.numpy() < best_val_loss:
                    best_val_loss = val_loss_total.numpy()
                    patience_counter = 0
                    # Salvar melhores pesos (simulado)
                    best_weights = self.model.get_weights()
                else:
                    patience_counter += 1

                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"\nEarly stopping na época {epoch+1}")
                    # Restaurar melhores pesos
                    self.model.set_weights(best_weights)
                    break

            # Verbose
            if verbose and (epoch + 1) % 10 == 0:
                msg = (f"Época {epoch+1}/{epochs} - "
                       f"loss: {avg_loss:.4f} "
                       f"(data: {avg_loss_data:.4f}, "
                       f"physics: {avg_loss_physics:.4f})")
                if X_val is not None:
                    msg += (f" - val_loss: {history['val_loss'][-1]:.4f} "
                            f"(data: {history['val_loss_data'][-1]:.4f}, "
                            f"physics: {history['val_loss_physics'][-1]:.4f})")
                print(msg)

        self.history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições.

        Args:
            X: Features de entrada

        Returns:
            Predições de porosidade
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado.")

        X_tf = tf.constant(X, dtype=tf.float32)
        predictions = self.model(X_tf, training=False).numpy()
        return predictions.flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Avalia o modelo.

        Args:
            X_test: Features de teste
            y_test: Target de teste

        Returns:
            Dicionário com métricas
        """
        predictions = self.predict(X_test)

        mse = np.mean((y_test - predictions) ** 2)
        mae = np.mean(np.abs(y_test - predictions))

        # Calcular R²
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }

    def get_params(self) -> Dict:
        """Retorna os parâmetros do modelo."""
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'lambda_physics': self.lambda_physics,
            'rho_ma': self.rho_ma,
            'rho_fl': self.rho_fl,
            'model_name': self.model_name
        }

    def save_model(self, filepath: str):
        """Salva o modelo."""
        if self.model is None:
            raise ValueError("Modelo não foi criado.")
        self.model.save(filepath)
        print(f"Modelo PINN salvo em: {filepath}")

    def load_model(self, filepath: str):
        """Carrega o modelo."""
        self.model = keras.models.load_model(filepath, compile=False)
        print(f"Modelo PINN carregado de: {filepath}")

    def __repr__(self):
        return (f"PINNModel(lambda_physics={self.lambda_physics}, "
                f"hidden_layers={self.hidden_layers})")


if __name__ == "__main__":
    # Teste rápido do modelo
    print("=" * 60)
    print("TESTE DO MODELO PINN")
    print("=" * 60)

    # Criar dados sintéticos
    np.random.seed(42)
    n_samples = 1000

    # [GR, RHOB, DT, ILD]
    X_train = np.column_stack([
        np.random.uniform(30, 150, n_samples),  # GR
        np.random.uniform(2.0, 2.7, n_samples),  # RHOB
        np.random.uniform(50, 100, n_samples),  # DT
        np.random.uniform(1, 100, n_samples)  # ILD
    ])

    # Porosidade baseada na densidade + ruído
    y_train = (2.65 - X_train[:, 1]) / (2.65 - 1.0) + np.random.normal(0, 0.05, n_samples)
    y_train = np.clip(y_train, 0, 0.5)

    X_val = np.column_stack([
        np.random.uniform(30, 150, 200),
        np.random.uniform(2.0, 2.7, 200),
        np.random.uniform(50, 100, 200),
        np.random.uniform(1, 100, 200)
    ])
    y_val = (2.65 - X_val[:, 1]) / (2.65 - 1.0) + np.random.normal(0, 0.05, 200)
    y_val = np.clip(y_val, 0, 0.5)

    # Criar e treinar modelo
    model = PINNModel(lambda_physics=1.0)
    print(f"\n{model}")
    print(f"\nParâmetros: {model.get_params()}")

    print("\n" + "=" * 60)
    print("TREINANDO MODELO PINN (20 épocas para teste)...")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        verbose=1
    )

    # Avaliar
    metrics = model.evaluate(X_val, y_val)
    print("\n" + "=" * 60)
    print("MÉTRICAS DE AVALIAÇÃO:")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
