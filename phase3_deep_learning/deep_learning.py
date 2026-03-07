"""Phase 3: Deep Learning Models - LSTM and Transformer."""

import numpy as np
import pandas as pd
import tensorflow as tf


keras = tf.keras
layers = tf.keras.layers
Sequential = tf.keras.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam
from utils.config import TICKERS, BATCH_SIZE, EPOCHS, LOOKBACK_WINDOW, FORECAST_HORIZONS
from utils.helpers import load_preprocessed_data, save_results, calculate_metrics, directional_accuracy
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class LSTM7DayModel:
    """Many-to-Many LSTM for 7-day short-term momentum forecasting."""
    
    def __init__(self, input_shape, output_steps=7):
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.model = None
        self.history = None
        
    def build(self):
        """Build LSTM architecture."""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.output_steps)  # Output 7-day forecast
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        self.model = model
        print("LSTM 7-Day model built")
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the model."""
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        print(f"LSTM trained for {epochs} epochs")
        
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention - captures attention weights."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=rate
        )
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.attention_weights = None
        
    def call(self, inputs, training=False, return_attention_scores=False):
        # Store attention scores if requested
        if return_attention_scores:
            attn_output, attn_scores = self.att(
                inputs, inputs,
                return_attention_scores=True,
                training=training
            )
            self.attention_weights = attn_scores
        else:
            attn_output = self.att(inputs, inputs, training=training)
        
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class Transformer30DayModel:
    """Transformer model with multi-head attention for 30-day trend forecasting."""
    
    def __init__(self, input_shape, output_steps=30, num_heads=4, ff_dim=128):
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.model = None
        self.history = None
        self.attention_blocks = []
        
    def build(self):
        """Build Transformer architecture with attention layer access."""
        inputs = keras.Input(shape=self.input_shape)
        
        # Embed to 64 dimensions
        x = layers.Dense(64, activation='relu')(inputs)
        
        # Transformer blocks - store references for attention extraction
        self.attention_blocks = []
        for _ in range(2):
            attn_block = TransformerBlock(
                embed_dim=64,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                rate=0.1
            )
            self.attention_blocks.append(attn_block)
            x = attn_block(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        
        # Output: 30-day forecast
        outputs = layers.Dense(self.output_steps)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        print("Transformer 30-Day model built")
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the model."""
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        print(f"Transformer trained for {epochs} epochs")
        
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def get_attention_weights(self, X):
        """Extract attention weights from the model."""
        # Create output model that includes attention block outputs
        intermediate_layer_model = keras.Model(
            inputs=self.model.input,
            outputs=[self.model.output, self.attention_blocks[0].attention_weights]
            if self.attention_blocks[0].attention_weights is not None else self.model.output
        )
        return intermediate_layer_model.predict(X)


def create_multi_step_labels(y, forecast_horizon):
    """Convert single-step labels to multi-step forecast labels."""
    X_multi = []
    y_multi = []
    
    for i in range(len(y) - forecast_horizon):
        X_multi.append(y[i])  # This is already a sequence from X
        y_multi.append(y[i:i + forecast_horizon])
    
    return np.array(X_multi), np.array(y_multi)


def train_lstm_7day(data_dict, ticker):
    """Train 7-day LSTM model."""
    print(f"\n--- LSTM 7-Day Model for {ticker} ---")
    
    X_train_raw = data_dict['X_train']  # Shape: (N,60, 7)
    X_val_raw = data_dict['X_val']
    X_test_raw = data_dict['X_test']
    y_train_raw = data_dict['y_train']  # Shape: (N,)
    y_val_raw = data_dict['y_val']
    y_test_raw = data_dict['y_test']
    
    # Create 7-day multi-step targets
    def create_7day_targets(X, y, horizon=7):
        X_out = []
        y_out = []
        for i in range(len(y) - horizon + 1):
            X_out.append(X[i])
            # Create 7-day target from the next 7 prices
            y_out.append(y[i:i+horizon])
        return np.array(X_out), np.array(y_out)
    
    # Reconstruct price sequences from scaled data (simple 1D difference)
    all_y = np.concatenate([y_train_raw, y_val_raw, y_test_raw])
    
    X_train, y_train = create_7day_targets(X_train_raw, all_y[:len(X_train_raw)+6], horizon=7)
    X_val, y_val = create_7day_targets(X_val_raw, all_y[len(X_train_raw):len(X_train_raw)+len(X_val_raw)+6], horizon=7)
    X_test, y_test = create_7day_targets(X_test_raw, all_y[len(X_train_raw)+len(X_val_raw):], horizon=7)
    
    print(f"  Reshaped: X_train {X_train.shape},  y_train {y_train.shape}")
    
    # Build and train model
    lstm_model = LSTM7DayModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_steps=FORECAST_HORIZONS['short']
    )
    lstm_model.build()
    lstm_model.fit(X_train, y_train, X_val, y_val, epochs=min(EPOCHS, 20))  # Reduced epochs for speed
    
    # Predict on test data
    y_pred = lstm_model.predict(X_test)
    
    # For evaluation, use the first prediction from each 7-day forecast
    y_pred_eval = y_pred[:, 0]
    y_test_eval = y_test[:, 0]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_eval, y_pred_eval)
    dir_acc = directional_accuracy(y_test_eval, y_pred_eval)
    
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"Directional Accuracy: {dir_acc:.4f}")
    
    return {
        'model': lstm_model,
        'predictions': y_pred,
        'actuals': y_test,
        'metrics': metrics,
        'directional_accuracy': dir_acc
    }


def train_transformer_30day(data_dict, ticker):
    """Train 30-day Transformer model."""
    print(f"\n--- Transformer 30-Day Model for {ticker} ---")
    
    X_train_raw = data_dict['X_train']
    X_val_raw = data_dict['X_val']
    X_test_raw = data_dict['X_test']
    y_train_raw = data_dict['y_train']
    y_val_raw = data_dict['y_val']
    y_test_raw = data_dict['y_test']
    
    # Create 30-day multi-step targets
    def create_multiday_targets(X,  y, horizon=30):
        X_out = []
        y_out = []
        for i in range(len(y) - horizon + 1):
            X_out.append(X[i])
            y_out.append(y[i:i+horizon])
        return np.array(X_out), np.array(y_out)
    
    # Reconstruct price sequences
    all_y = np.concatenate([y_train_raw, y_val_raw, y_test_raw])
    
    X_train, y_train = create_multiday_targets(X_train_raw, all_y[:len(X_train_raw)+29], horizon=30)
    X_val, y_val = create_multiday_targets(X_val_raw, all_y[len(X_train_raw):len(X_train_raw)+len(X_val_raw)+29], horizon=30)
    X_test, y_test = create_multiday_targets(X_test_raw, all_y[len(X_train_raw)+len(X_val_raw):], horizon=30)
    
    print(f"  Reshaped: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # Build and train model
    transformer_model = Transformer30DayModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_steps=FORECAST_HORIZONS['long']
    )
    transformer_model.build()
    transformer_model.fit(X_train, y_train, X_val, y_val, epochs=min(EPOCHS, 20))  # Reduced epochs
    
    # Predict on test data
    y_pred = transformer_model.predict(X_test)
    
    # For evaluation, use the first prediction from each 30-day forecast
    y_pred_eval = y_pred[:, 0]
    y_test_eval = y_test[:, 0]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_eval, y_pred_eval)
    dir_acc = directional_accuracy(y_test_eval, y_pred_eval)
    
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"Directional Accuracy: {dir_acc:.4f}")
    
    return {
        'model': transformer_model,
        'predictions': y_pred,
        'actuals': y_test,
        'metrics': metrics,
        'directional_accuracy': dir_acc
    }


def run_phase3(tickers=TICKERS):
    """Run Phase 3 deep learning models."""
    print("\n" + "="*60)
    print("PHASE 3: DEEP LEARNING MODELS")
    print("="*60)
    
    all_results = {}
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Deep Learning Models for {ticker}")
        print(f"{'='*60}")
        
        # Load Phase 1 data
        data_dict = load_preprocessed_data(ticker, phase='phase1')
        if data_dict is None:
            print(f"ERROR: Could not load Phase 1 data for {ticker}")
            continue
        
        # Train models
        lstm_results = train_lstm_7day(data_dict, ticker)
        transformer_results = train_transformer_30day(data_dict, ticker)
        
        all_results[ticker] = {
            'LSTM_7D': lstm_results,
            'Transformer_30D': transformer_results
        }
        
        # Save results
        save_results({
            ticker: {
                'LSTM_RMSE': lstm_results['metrics']['RMSE'],
                'LSTM_MAE': lstm_results['metrics']['MAE'],
                'LSTM_Directional_Accuracy': lstm_results['directional_accuracy'],
                'Transformer_RMSE': transformer_results['metrics']['RMSE'],
                'Transformer_MAE': transformer_results['metrics']['MAE'],
                'Transformer_Directional_Accuracy': transformer_results['directional_accuracy']
            }
        }, f'phase3_deep_learning_{ticker}.json')
    
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE: Deep learning models trained")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    dl_results = run_phase3()
