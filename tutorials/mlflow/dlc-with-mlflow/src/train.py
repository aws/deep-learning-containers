import os
import gc
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import time
import boto3
import mlflow
import mlflow.tensorflow
import joblib
import mlflow.data

print("TensorFlow version:", tf.__version__)
print("MLflow version:", mlflow.__version__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Connect to the MLflow tracking server
def setup_mlflow():
    # Get the MLflow tracking server URL from the ARN
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    tracking_server_name = "Mlflow3"
    try:
        response = sagemaker_client.describe_mlflow_tracking_server(
            TrackingServerName=tracking_server_name
        )
        tracking_server_url = response['TrackingServerArn']
        print(f"MLflow tracking server URL: {tracking_server_url}")
        
        # Set the tracking URI
        mlflow.set_tracking_uri(tracking_server_url)
        
        # Create or set the experiment
        experiment_name = "abalone-tensorflow-experiment"
        mlflow.set_experiment(experiment_name)
        
        print(f"Using experiment: {experiment_name}")
        print("MLflow configured - autolog disabled, using custom callback")
        return True
    except Exception as e:
        print(f"Error connecting to MLflow tracking server: {e}")
        print("Continuing without MLflow tracking...")
        return False

# Create directories for model artifacts and plots
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
os.makedirs(model_dir, exist_ok=True)

# Create a directory for plots
plots_dir = os.path.join(model_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Load the Abalone dataset from local file
print("Loading Abalone dataset...")
data_path = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/data')
abalone_file = os.path.join(data_path, 'abalone.data')

column_names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", 
                "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]

# Try to load from local file, fallback to download if needed
try:
    if os.path.exists(abalone_file):
        print(f"Loading dataset from {abalone_file}")
        abalone_df = pd.read_csv(abalone_file, names=column_names)
    else:
        print("Local dataset not found, downloading from UCI repository...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
        abalone_df = pd.read_csv(url, names=column_names)
        # Save for future use
        abalone_df.to_csv(abalone_file, index=False, header=False)
        print(f"Dataset saved to {abalone_file}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Attempting to download from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    abalone_df = pd.read_csv(url, names=column_names)

# Display basic information
print("\nDataset Information:")
print(f"Shape: {abalone_df.shape}")
print("\nFirst 5 rows:")
print(abalone_df.head())

print("\nSummary statistics:")
print(abalone_df.describe())

# Check for missing values
print("\nMissing values:")
print(abalone_df.isnull().sum())

# Data visualization
print("\nCreating exploratory visualizations...")

# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(abalone_df['Rings'], kde=True)
plt.title('Distribution of Abalone Rings (Age)')
plt.xlabel('Rings')
plt.ylabel('Count')
plt.savefig(os.path.join(plots_dir, 'rings_distribution.png'))
plt.close()  # Close the figure to free memory
plt.clf()    # Clear the current figure

# Correlation matrix
plt.figure(figsize=(12, 10))
numeric_df = abalone_df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
plt.close()  # Close the figure
plt.clf()    # Clear the figure
del correlation, numeric_df  # Delete variables to free memory

# Data Preprocessing
print("\nPreprocessing the data...")

# The 'Rings' feature is the target variable (age = rings + 1.5)
X = abalone_df.drop('Rings', axis=1)
y = abalone_df['Rings'].values

# Handle categorical feature (Sex: 'M', 'F', 'I')
categorical_features = ['Sex']
numerical_features = ['Length', 'Diameter', 'Height', 'Whole_weight', 
                      'Shucked_weight', 'Viscera_weight', 'Shell_weight']

# Create preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Convert to a format suitable for TensorFlow
X_processed = np.array(X_processed, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Custom MLflow callback for detailed epoch logging
class MLflowCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"Starting epoch {epoch + 1}")
        
    def on_epoch_end(self, epoch, logs=None):
        # Calculate epoch duration
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        
        print(f"Completed epoch {epoch + 1} in {epoch_duration:.2f}s")
        
        if logs and mlflow_enabled:
            # Log all training metrics
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)
                print(f"  {metric_name}: {metric_value:.4f}")
            
            # Log epoch timing
            mlflow.log_metric("epoch_duration", epoch_duration, step=epoch)
            mlflow.log_metric("avg_epoch_duration", np.mean(self.epoch_times), step=epoch)
            
            # Log learning rate if available
            if hasattr(self.model.optimizer, 'learning_rate'):
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                mlflow.log_metric("learning_rate", current_lr, step=epoch)
                print(f"  learning_rate: {current_lr}")
                
    def on_train_end(self, logs=None):
        if mlflow_enabled:
            total_time = sum(self.epoch_times)
            mlflow.log_metric("total_training_time", total_time)
            mlflow.log_metric("final_avg_epoch_time", np.mean(self.epoch_times))
            print(f"Training completed. Total time: {total_time:.2f}s, Avg epoch: {np.mean(self.epoch_times):.2f}s")

# Initialize MLflow tracking
mlflow_enabled = setup_mlflow()

# Start MLflow run
if mlflow_enabled:
    mlflow_run = mlflow.start_run()
    run_id = mlflow_run.info.run_id
    print(f"MLflow run ID: {run_id}")
    
    # Log dataset info
    dataset = mlflow.data.from_pandas(abalone_df, name="abalone.data")
    mlflow.log_input(dataset, context="training")
    mlflow.log_param("dataset_size", len(abalone_df))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    # Log dataset visualizations
    mlflow.log_artifact(os.path.join(plots_dir, 'rings_distribution.png'))
    mlflow.log_artifact(os.path.join(plots_dir, 'correlation_matrix.png'))

# Model Definition
print("\nDefining the TensorFlow model...")

# Get the number of features after preprocessing
num_features = X_train.shape[1]

# Define a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error
)

# Display model summary
model.summary()

# Log model parameters
if mlflow_enabled:
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("dropout_rate", 0.2)
    mlflow.log_param("hidden_layers", "64-32-16")
    mlflow.log_param("activation", "relu")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_function", "mse")
    mlflow.log_param("total_params", model.count_params())

# Model Training
print("\nTraining the model...")
start_time = time.time()

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'checkpoint-{epoch}.keras'), 
        save_best_only=True, 
        monitor='val_loss')
]

# Add custom MLflow callback
if mlflow_enabled:
    callbacks.append(MLflowCallback())

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Clear TensorFlow backend session to free GPU/CPU memory
tf.keras.backend.clear_session()

# Force Python garbage collection
gc.collect()

if mlflow_enabled:
    mlflow.log_metric("training_time_seconds", training_time)

# Model Evaluation
print("\nEvaluating the model...")

# Evaluate on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Mean Absolute Error: {test_mae:.4f}")
print(f"Test Mean Squared Error: {test_loss:.4f}")
print(f"Test Root Mean Squared Error: {np.sqrt(test_loss):.4f}")

# Make predictions
y_pred = model.predict(X_test)

# Calculate additional metrics
mse = np.mean((y_test - y_pred.flatten()) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred.flatten()))
r2 = 1 - (np.sum((y_test - y_pred.flatten()) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Log metrics to MLflow
if mlflow_enabled:
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("training_time_seconds", training_time)

# Visualize Training History
plt.figure(figsize=(12, 4))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation metrics
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'training_history.png'))
plt.close()  # Close the figure
plt.clf()    # Clear the figure

# Visualize Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Predicted vs Actual Abalone Rings')
plt.savefig(os.path.join(plots_dir, 'predictions.png'))
plt.close()  # Close the figure
plt.clf()    # Clear the figure

# Log additional visualizations to MLflow
if mlflow_enabled:
    mlflow.log_artifact(os.path.join(plots_dir, 'training_history.png'))
    mlflow.log_artifact(os.path.join(plots_dir, 'predictions.png'))

# Save the model in the native Keras format (.keras)
model_path = os.path.join(model_dir, 'abalone_model.keras')
model.save(model_path)  # Native Keras format - no need to specify save_format
print(f"\nModel saved to {model_path} in native Keras format")

# Save the preprocessor
preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocessor saved to {preprocessor_path}")

# Optional: Save a TensorFlow SavedModel format for deployment compatibility
saved_model_path = os.path.join(model_dir, 'saved_model')
tf.saved_model.save(model, saved_model_path)
print(f"SavedModel also saved to {saved_model_path} for deployment compatibility")

# Log additional artifacts and model
if mlflow_enabled:
    # Log visualizations
    mlflow.log_artifact(os.path.join(plots_dir, 'training_history.png'))
    mlflow.log_artifact(os.path.join(plots_dir, 'predictions.png'))
    
    # Log the preprocessor
    mlflow.log_artifact(preprocessor_path)
    
    # Create model signature and log model
    from mlflow.models.signature import infer_signature
    sample_input = X_test[:5]
    sample_output = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)
    
    # Log model with signature
    mlflow.tensorflow.log_model(
        model,
        name="model",
        signature=signature,
        registered_model_name="abalone-tensorflow-custom-callback-model"
    )
    
    print("Model and artifacts logged to MLflow")
    
    # End the MLflow run
    mlflow.end_run()
    print(f"MLflow run {run_id} completed with custom callback")

print("\nTraining complete!")

# Final memory cleanup
plt.close('all')  # Close any remaining matplotlib figures
tf.keras.backend.clear_session()  # Clear TensorFlow session
gc.collect()  # Force garbage collection

print("Memory cleanup completed.")