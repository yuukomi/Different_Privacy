import numpy as np
import pandas as pd

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error"""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """Calculate mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.feature_means = None
        self.feature_stds = None
        self.target_mean = None
        self.target_std = None
        
    def normalize_features(self, X):
        """Normalize features to have zero mean and unit variance"""
        if self.feature_means is None:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-8
            
        X_norm = (X - self.feature_means) / self.feature_stds
        return X_norm
    
    def normalize_target(self, y):
        """Normalize target variable"""
        if self.target_mean is None:
            self.target_mean = np.mean(y)
            self.target_std = np.std(y) + 1e-8
            
        y_norm = (y - self.target_mean) / self.target_std
        return y_norm
    
    def denormalize_target(self, y_norm):
        """Convert normalized predictions back to original scale"""
        return y_norm * self.target_std + self.target_mean
        
    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Normalize data
        X_norm = self.normalize_features(X)
        y_norm = self.normalize_target(y)
        
        n_samples, n_features = X_norm.shape
        
        # Initialize weights with small random values
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model prediction
            y_pred = np.dot(X_norm, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X_norm.T, (y_pred - y_norm))
            db = (1/n_samples) * np.sum(y_pred - y_norm)
            
            # Update parameters with gradient clipping
            dw = np.clip(dw, -1, 1)  # Clip gradients to prevent explosion
            db = np.clip(db, -1, 1)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        # Convert to numpy array
        X = np.array(X, dtype=np.float64)
        
        # Normalize features
        X_norm = self.normalize_features(X)
        
        # Make prediction in normalized space
        y_pred_norm = np.dot(X_norm, self.weights) + self.bias
        
        # Denormalize prediction
        return self.denormalize_target(y_pred_norm)

class QuadraticRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.linear_model = LinearRegression(learning_rate, n_iterations)
        
    def transform_features(self, X):
        # Create quadratic features
        X_quad = pd.DataFrame(X.copy())
        
        # Add squared terms for Engine and Power
        X_quad['Engine_2'] = X_quad['Engine'] ** 2
        X_quad['Max Power_2'] = X_quad['Max Power'] ** 2
        
        # Add Length * Width interaction
        X_quad['Length_Width'] = X_quad['Length'] * X_quad['Width']
        
        return X_quad
        
    def fit(self, X, y):
        X_quad = self.transform_features(X)
        self.linear_model.fit(X_quad, y)
        
    def predict(self, X):
        X_quad = self.transform_features(X)
        return self.linear_model.predict(X_quad)

class InteractionRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.linear_model = LinearRegression(learning_rate, n_iterations)
        
    def transform_features(self, X):
        X_inter = pd.DataFrame(X.copy())
        
        # Create interaction terms
        X_inter['Year_Engine'] = X_inter['Age'] * X_inter['Engine']
        X_inter['Power_Torque'] = X_inter['Max Power'] * X_inter['Max Torque']
        X_inter['Volume'] = X_inter['Length'] * X_inter['Width'] * X_inter['Height']
        
        return X_inter
        
    def fit(self, X, y):
        X_inter = self.transform_features(X)
        self.linear_model.fit(X_inter, y)
        
    def predict(self, X):
        X_inter = self.transform_features(X)
        return self.linear_model.predict(X_inter)

class HybridRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.linear_model = LinearRegression(learning_rate, n_iterations)
        
    def transform_features(self, X):
        X_hybrid = pd.DataFrame(X.copy())
        
        # Handle negative or zero values before transformation
        X_hybrid['Kilometer'] = np.maximum(X_hybrid['Kilometer'], 1e-8)
        X_hybrid['Engine'] = np.maximum(X_hybrid['Engine'], 1e-8)
        
        # Create hybrid features
        X_hybrid['sqrt_Kilometer'] = np.sqrt(X_hybrid['Kilometer'])
        X_hybrid['log_Engine'] = np.log1p(X_hybrid['Engine'])
        
        # Add complex interactions with safety checks
        denominator1 = X_hybrid['Engine'] + 1e-8
        denominator2 = X_hybrid['Height'] + 1e-8
        
        X_hybrid['Power_Torque_Engine'] = (X_hybrid['Max Power'] * X_hybrid['Max Torque']) / denominator1
        X_hybrid['Length_Width_Height'] = (X_hybrid['Length'] * X_hybrid['Width']) / denominator2
        
        return X_hybrid
        
    def fit(self, X, y):
        X_hybrid = self.transform_features(X)
        self.linear_model.fit(X_hybrid, y)
        
    def predict(self, X):
        X_hybrid = self.transform_features(X)
        return self.linear_model.predict(X_hybrid)

def evaluate_model(y_true, y_pred, model_name=""):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    return mse, rmse, mae

if __name__ == "__main__":
    # Import preprocessing
    from data_preprocessing import load_and_preprocess
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    
    # Test all models
    models = {
        "Basic Linear": LinearRegression(),
        "Quadratic": QuadraticRegression(),
        "Interaction": InteractionRegression(),
        "Hybrid": HybridRegression()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} Model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse, rmse, mae = evaluate_model(y_test, y_pred, name)
        results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae} 