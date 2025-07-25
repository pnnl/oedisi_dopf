import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os
import pickle


def init_paths(args):
    input = f"training/{args.model}"
    if not os.path.exists(input):
        print("data location doesn't exist")
        exit()

    output = os.path.abspath(f"model/{args.model}")
    os.makedirs(output, exist_ok=True)
    return input, output


class PowerSystemInjectionPredictor:
    def __init__(self, n_buses, n_phases, output_dir):
        """
        Initialize the Power System Injection Predictor

        Args:
            n_buses (int): Number of buses in the system
        """
        self.n_buses = n_buses
        self.n_phases = n_phases
        self.output_dir = output_dir
        self.model = None
        self.injection_scaler = StandardScaler()
        self.nonZeroIdx = None
        self.totInjCol = None

    def _detectAbsentNodes(self, array_2D):
        first_row = array_2D[0]
        columns_to_remove = [
            i for i, value in enumerate(first_row) if value == 0]
        return columns_to_remove

    def _removeAbsentNodes(self, _array, columns_to_remove):
        array_modified = np.delete(_array, columns_to_remove, axis=1)
        return array_modified

    def _modifyAbsentPhases(self, array_2D, base_array=None):
        if base_array is None:
            base_array = array_2D
        columns_to_remove = self._detectAbsentNodes(base_array)
        array_modified = self._removeAbsentNodes(array_2D, columns_to_remove)
        return array_modified

    def prepare_data(self, voltage_data, load_forecast_data, r_data, x_data, injection_data=None):
        """
        Prepare and normalize the input data

        Args:
            voltage_data: Array of shape (n_samples, n_buses, n_phases) - Voltage measurements
            load_forecast_data: Array of shape (n_samples, n_buses, n_phases) - Load forecasts
            r_data: Array of shape (n_samples, n_branches, n_phases, n_phases) - Branch resistance matrices
            x_data: Array of shape (n_samples, n_branches, n_phases, n_phases) - Branch reactance matrices
            injection_data: Array of shape (n_samples, n_buses, n_phases) - Net injections (for training)

        Returns:
            tuple: (X_processed, y_processed) where y can be None if injection_data is None
        """
        # Reshape data for scaling
        voltage_flat = voltage_data.reshape(-1, self.n_buses * self.n_phases)
        load_flat = load_forecast_data.reshape(-1,
                                               self.n_buses * self.n_phases)

        # For impedance components, we need to handle the matrix structure
        n_samples = voltage_data.shape[0]
        n_branches = r_data.shape[1]
        r_flat = r_data.reshape(
            n_samples, n_branches * self.n_phases * self.n_phases)
        x_flat = x_data.reshape(
            n_samples, n_branches * self.n_phases * self.n_phases)

        # Remove Absent phases:
        voltage_flat_mod = self._modifyAbsentPhases(voltage_flat)
        load_flat_mod = self._modifyAbsentPhases(
            load_flat, base_array=voltage_flat)
        r_flat_mod = self._modifyAbsentPhases(r_flat)
        x_flat_mod = self._modifyAbsentPhases(x_flat)

        # Use data as is (no scaling)
        voltage_processed = voltage_flat_mod
        load_processed = load_flat_mod
        r_processed = r_flat_mod
        x_processed = x_flat_mod

        # # Scale the data
        # voltage_scaled = self.voltage_scaler.fit_transform(voltage_flat_mod)
        # load_scaled = self.load_scaler.fit_transform(load_flat_mod)
        # r_scaled = self.r_scaler.fit_transform(r_flat_mod)
        # x_scaled = self.x_scaler.fit_transform(x_flat_mod)

        # Combine features
        X = np.hstack([voltage_processed, load_processed,
                      r_processed, x_processed])

        if injection_data is not None:
            injection_flat = injection_data.reshape(
                -1, self.n_buses * self.n_phases)
            self.nonZeroIdx = np.nonzero(injection_flat[89, :])[0]
            self.totInjCol = injection_flat.shape[1]
            mask = np.zeros(injection_flat.shape[1], dtype=bool)
            mask[self.nonZeroIdx] = True

            # Keep only columns in the list
            injection_filtered = injection_flat[:, mask]

            # injection_flat_mod = self._modifyAbsentPhases(injection_flat, base_array=voltage_flat)
            # y = self.injection_scaler.fit_transform(injection_flat_mod)
            y = injection_filtered

            return X, y
        else:
            return X, None

    def _create_residual_model(self, input_dim, output_dim):
        inputs = keras.Input(shape=(input_dim,))

        # Input normalization
        x = keras.layers.BatchNormalization()(inputs)

        # Initial expansion
        x = keras.layers.Dense(2048, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.15)(x)

        # Residual block 1
        residual = x
        x = keras.layers.Dense(2048, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(2048, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, residual])
        x = keras.layers.Activation('swish')(x)
        x = keras.layers.Dropout(0.1)(x)

        # Compression layer
        x = keras.layers.Dense(1536, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.08)(x)

        # Residual block 2
        residual = x
        x = keras.layers.Dense(1536, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.08)(x)
        x = keras.layers.Dense(1536, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, residual])
        x = keras.layers.Activation('swish')(x)

        # Final compression
        x = keras.layers.Dense(1024, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.05)(x)

        x = keras.layers.Dense(768, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.03)(x)

        x = keras.layers.Dense(512, activation='swish')(x)
        x = keras.layers.BatchNormalization()(x)

        # Output layer
        outputs = keras.layers.Dense(output_dim, activation='relu')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model

    def _create_layer_model(self, input_dim, output_dim):
        """
        Model with built-in normalization layers
        """
        inputs = keras.Input(shape=(input_dim,))

        # Aggressive input normalization
        x = keras.layers.LayerNormalization()(inputs)

        # Very simple architecture
        x = keras.layers.Dense(64, activation='linear')(x)  # Linear first
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(0.7)(x)  # Very high dropout

        x = keras.layers.Dense(32, activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(0.5)(x)

        # Output
        outputs = keras.layers.Dense(output_dim, activation='linear')(x)
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.Activation('relu')(
            outputs)  # Apply relu after BN

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model

    def _create_optimal_model(self, input_dim, output_dim):
        inputs = keras.Input(shape=(input_dim,))

        # Input normalization
        x = keras.layers.BatchNormalization()(inputs)

        # First expansion layer - go wider than input
        x = keras.layers.Dense(2048, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.15)(x)

        # Second layer - maintain width
        x = keras.layers.Dense(1500, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.12)(x)

        # Third layer - gradual compression
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)

        # Fourth layer - continue compression
        x = keras.layers.Dense(768, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.08)(x)

        # Fifth layer - approach output size
        x = keras.layers.Dense(512, activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)

        # Pre-output layer
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)

        # Output layer (no activation for regression, or use appropriate activation)
        outputs = keras.layers.Dense(output_dim, activation='relu')(
            x)  # or activation='relu' for non-negative

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_improved_model(self, input_dim, output_dim):
        """
        Improved model architecture for better generalization
        """
        inputs = keras.Input(shape=(input_dim,))

        # Add input normalization (helps without explicit scaling)
        x = keras.layers.BatchNormalization()(inputs)

        # Reduce model complexity to prevent overfitting
        x = keras.layers.Dense(1024, activation='gelu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)  # Higher dropout

        x = keras.layers.Dense(512, activation='gelu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)

        x = keras.layers.Dense(256, activation='gelu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

        x = keras.layers.Dense(128, activation='gelu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)

        # Output layer
        outputs = keras.layers.Dense(output_dim)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model

    def build_model(self, input_dim):
        """
        Build the neural network model

        Args:
            input_dim (int): Dimension of input features

        Returns:
            keras.Model: Built neural network model
        """
        # output_dim = self.n_buses * self.n_phases
        output_dim = len(self.nonZeroIdx)

        # model = self._create_residual_model(input_dim, output_dim)
        # model = self._create_layer_model(input_dim, output_dim)
        model = self._create_optimal_model(
            input_dim, output_dim)  # works better

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )

        # model = self._build_improved_model(input_dim, output_dim)
        # model.compile(
        #     optimizer=keras.optimizers.AdamW(
        #         learning_rate=0.0001,  # Lower learning rate
        #         weight_decay=1e-4  # Add weight decay
        #     ),
        #     loss='huber',  # More robust loss function
        #     metrics=['mae', 'mse']
        # )

        return model

    def train(self, voltage_data, load_forecast_data, r_data, x_data, injection_data,
              epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the model to predict net injections

        Args:
            voltage_data: Voltage measurements
            load_forecast_data: Load forecasts
            r_data: Branch resistance matrices
            x_data: Branch reactance matrices
            injection_data: Net injection values (ground truth)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            validation_split (float): Fraction of data to use for validation

        Returns:
            dict: Training history
        """
        X, y = self.prepare_data(
            voltage_data, load_forecast_data, r_data, x_data, injection_data)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=10)

        # Build and compile the model
        self.model = self.build_model(X.shape[1])

        # Callbacks for training
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=50, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            'power_system_model.h5', save_best_only=True, monitor='val_loss')

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        return history

    def predict(self, voltage_data, load_forecast_data, r_data, x_data):
        """
        Predict net injections

        Args:
            voltage_data: Voltage measurements
            load_forecast_data: Load forecasts
            r_data: Branch resistance matrices
            x_data: Branch reactance matrices

        Returns:
            ndarray: Predicted net injections of shape (n_samples, n_buses, n_phases)
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")

        X, _ = self.prepare_data(
            voltage_data, load_forecast_data, r_data, x_data)

        # Make predictions
        y_pred_filtered = self.model.predict(X)

        # Reconstruct full array with zeros for removed columns
        n_samples = y_pred_filtered.shape[0]
        y_pred_full = np.zeros((n_samples, self.totInjCol))
        y_pred_full[:, self.nonZeroIdx] = y_pred_filtered

        # Reshape to original dimensions
        y_pred = y_pred_full.reshape(-1, self.n_buses, self.n_phases)

        return y_pred

    def evaluate(self, voltage_data, load_forecast_data, r_data, x_data, injection_data):
        """
        Evaluate the model

        Args:
            voltage_data: Voltage measurements
            load_forecast_data: Load forecasts
            r_data: Branch resistance matrices
            x_data: Branch reactance matrices
            injection_data: Ground truth net injections

        Returns:
            tuple: (loss, mae)
        """
        X, y = self.prepare_data(
            voltage_data, load_forecast_data, r_data, x_data, injection_data)
        return self.model.evaluate(X, y)

    def plot_training_history(self, history):
        """
        Plot training history

        Args:
            history: Training history from model.fit()
        """
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_metrics.png", dpi=300,
                    bbox_inches="tight")  # Save with high resolution
        print("Figure saved successfully as 'training_metrics.png'.")

        # Optionally display the figure after saving
        # plt.show()

    def save_model(self, filepath):
        """
        Save the trained model

        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        self.model.save(filepath)

        params_data = {
            'nonZeroIdx': self.nonZeroIdx,
            'totInjCol': self.totInjCol,
            'n_buses': self.n_buses,
            'n_phases': self.n_phases
        }

        with open(f"{self.output_dir}/params.pkl", 'wb') as file:
            pickle.dump(params_data, file)

    def load_model(self, filepath):
        """
        Load a trained model

        Args:
            filepath (str): Path to load the model from
        """
        self.model = keras.models.load_model(filepath)

        # Load parameters
        with open(f"{self.output_dir}/params.pkl", 'rb') as file:
            params_data = pickle.load(file)

        self.nonZeroIdx = params_data['nonZeroIdx']
        self.totInjCol = params_data['totInjCol']
        self.n_buses = params_data['n_buses']
        self.n_phases = params_data['n_phases']


def read_data_synthetic():

    n_buses = 132  # Number of buses in the system
    n_samples = 100  # Number of data samples
    n_phases = 3
    n_branches = 131  # Assuming a radial network

    # Generate synthetic data for demonstration
    # In a real scenario, you would load your actual data here
    np.random.seed(42)

    # Voltage data: shape (n_samples, n_buses, n_phases)
    voltage_data = np.random.normal(1.05, 0.95, (n_samples, n_buses, n_phases))

    # Load forecast data: shape (n_samples, n_buses, n_phases)
    load_forecast_data = np.random.normal(
        0.5, 0.0, (n_samples, n_buses, n_phases))

    # Resistance data: shape (n_samples, n_branches, n_phases, n_phases)
    base_r = np.random.normal(0.05, 0.00, (n_branches, n_phases, n_phases))
    # Make it symmetric
    for b in range(n_branches):
        base_r[b] = (base_r[b] + base_r[b].T) / 2
        # Resistance should be positive definite
        base_r[b] = np.abs(base_r[b])
    r_data = np.tile(base_r, (n_samples, 1, 1, 1))

    # Reactance data: shape (n_samples, n_branches, n_phases, n_phases)
    base_x = np.random.normal(0.15, 0.00, (n_branches, n_phases, n_phases))
    # Make it symmetric
    for b in range(n_branches):
        base_x[b] = (base_x[b] + base_x[b].T) / 2
    x_data = np.tile(base_x, (n_samples, 1, 1, 1))

    # Generate target net injection data using a simplified power flow model
    # This is just for demonstration - in reality you would have measured data
    def simplified_power_flow(voltages, loads, r_data, x_data):
        # A very simplified approximation - not a real power flow
        injections = np.zeros((n_samples, n_buses, n_phases))

        for i in range(n_samples):
            for b in range(n_buses):
                # Base injection is the negative of the load
                injections[i, b, :] = -loads[i, b, :]

                # Add effect of voltage differences, resistance, and reactance
                if b > 0:  # Skip the slack bus (bus 0)
                    v_diff = voltages[i, b, :] - voltages[i, b - 1, :]

                    # Calculate total impedance Z = R + jX
                    z = r_data[i, b - 1] + 1j * x_data[i, b - 1]

                    # Simplified relationship for demonstration
                    try:
                        current = np.linalg.solve(z, v_diff)
                        apparent_power = voltages[i, b, :] * np.conj(current)
                        real_power = np.real(apparent_power)
                        injections[i, b, :] += real_power
                    except np.linalg.LinAlgError:
                        # In case of singular matrix, use a fallback approach
                        z_diag = np.diag(np.diag(z))
                        current = np.linalg.solve(
                            z_diag + 0.01 * np.eye(3), v_diff)
                        apparent_power = voltages[i, b, :] * np.conj(current)
                        real_power = np.real(apparent_power)
                        injections[i, b, :] += real_power

        return injections

    # Generate synthetic injection data
    injection_data = simplified_power_flow(
        voltage_data, load_forecast_data, r_data, x_data)

    return n_buses, n_phases, voltage_data, load_forecast_data, r_data, x_data, injection_data


def read_data_powerflow(input: str):
    # baseKV = 1000
    # baseS = 1
    # baseZ = baseKV**2/baseS

    voltage_data = np.load(f"{input}/voltage.npy")
    base_r = np.load(f"{input}/r.npy")
    base_x = np.load(f"{input}/x.npy")

    n_samples = voltage_data.shape[0]
    n_buses = voltage_data.shape[1]
    n_phases = voltage_data.shape[2]
    n_branches = base_r.shape[0]

    load_forecast_data = np.load(f"{input}/load.npy")
    injection_data = np.load(f"{input}/injection.npy")
    r_data = np.tile(base_r, (n_samples, 1, 1, 1))
    x_data = np.tile(base_x, (n_samples, 1, 1, 1))

    return n_buses, n_phases, voltage_data, load_forecast_data, r_data, x_data, injection_data


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anonymize OpenDSS data.")
    parser.add_argument(
        "--model", help="model name: ieee123, SFO-P1U, ...")
    args = parser.parse_args()

    feeder_name = args.model
    input, output = init_paths(args)

    n_buses, n_phases, voltage_data, load_forecast_data, r_data, x_data, injection_data = read_data_powerflow(
        input)

    # Initialize and train the model
    predictor = PowerSystemInjectionPredictor(n_buses, n_phases, output)
    history = predictor.train(
        voltage_data,
        load_forecast_data,
        r_data,
        x_data,
        injection_data,
        epochs=10,
        batch_size=5
    )

    # Plot training history
    predictor.plot_training_history(history)

    # Make predictions with the trained model
    predicted_injections = predictor.predict(
        voltage_data, load_forecast_data, r_data, x_data)

    # Evaluate the model
    loss, mae = predictor.evaluate(
        voltage_data, load_forecast_data, r_data, x_data, injection_data)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # Save the model
    predictor.save_model(f"{output}/power_system_injection_predictor.h5")
