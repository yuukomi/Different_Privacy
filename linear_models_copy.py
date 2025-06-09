import numpy as np
import pandas as pd

# Evaluation metrics từ code 1
def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error"""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """Calculate mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))

def evaluate_model(y_true, y_pred, model_name=""):
    """Comprehensive model evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    return mse, rmse, mae

class LinearRegression:
    """Basic Linear Regression: y = a1*x1 + a2*x2 + ... + an*xn + b"""
    
    def __init__(self, regularization=None, alpha=0.01):
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        self.regularization = regularization  # 'l1', 'l2', or None
        self.alpha = alpha  # regularization strength
        
    def fit(self, X, y):
        """
        Fit linear regression model using normal equation with optional regularization
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            
        # Add bias column (intercept)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation with regularization
        try:
            if self.regularization == 'l2':
                # Ridge regression
                I = np.eye(X_with_bias.shape[1])
                I[0, 0] = 0  # Don't regularize intercept
                theta = np.linalg.solve(X_with_bias.T @ X_with_bias + self.alpha * I, 
                                      X_with_bias.T @ y)
            else:
                # Standard linear regression
                theta = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            if self.regularization == 'l2':
                I = np.eye(X_with_bias.shape[1])
                I[0, 0] = 0
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias + self.alpha * I) @ X_with_bias.T @ y
            else:
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return X @ self.coefficients + self.intercept
    
    def get_equation(self):
        """Return the equation as a string"""
        if self.coefficients is None:
            return "Model not fitted yet"
        
        equation = f"y = {self.intercept:.4f}"
        for i, coef in enumerate(self.coefficients):
            feature_name = f"x{i+1}" if self.feature_names is None else self.feature_names[i]
            sign = "+" if coef >= 0 else ""
            equation += f" {sign} {coef:.4f}*{feature_name}"
        
        return equation

class QuadraticRegression:
    """Quadratic Regression với cải tiến từ code 1"""
    
    def __init__(self, regularization=None, alpha=0.01):
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        self.regularization = regularization
        self.alpha = alpha
        
    def _create_quadratic_features(self, X):
        """Create quadratic features with safety checks"""
        X = np.array(X)
        
        # Cải tiến từ code 1: Handle specific columns for quadratic terms
        if hasattr(self, 'feature_names') and self.feature_names:
            # Chỉ tạo quadratic terms cho Engine và Max Power như code 1
            quad_features = []
            for i, name in enumerate(self.feature_names):
                if 'Engine' in str(name) or 'Power' in str(name):
                    quad_features.append(X[:, i] ** 2)
            
            if quad_features:
                X_quad = np.column_stack([X] + quad_features)
            else:
                X_quad = np.column_stack([X, X**2])
        else:
            # Include original features and their squares
            X_quad = np.column_stack([X, X**2])
            
        return X_quad
    
    def fit(self, X, y):
        """Fit quadratic regression model"""
        X = np.array(X)
        y = np.array(y)
        
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Create quadratic features
        X_quad = self._create_quadratic_features(X)
        
        # Add bias column
        X_with_bias = np.column_stack([np.ones(X_quad.shape[0]), X_quad])
        
        # Normal equation with regularization
        try:
            if self.regularization == 'l2':
                I = np.eye(X_with_bias.shape[1])
                I[0, 0] = 0
                theta = np.linalg.solve(X_with_bias.T @ X_with_bias + self.alpha * I, 
                                      X_with_bias.T @ y)
            else:
                theta = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
        except np.linalg.LinAlgError:
            if self.regularization == 'l2':
                I = np.eye(X_with_bias.shape[1])
                I[0, 0] = 0
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias + self.alpha * I) @ X_with_bias.T @ y
            else:
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        X_quad = self._create_quadratic_features(X)
        return X_quad @ self.coefficients + self.intercept
    
    def get_equation(self):
        """Return simplified equation representation"""
        if self.coefficients is None:
            return "Model not fitted yet"
        
        return f"y = {self.intercept:.4f} + linear_terms + quadratic_terms (total {len(self.coefficients)} features)"

class InteractionRegression:
    """Interaction Regression với cải tiến từ code 1"""
    
    def __init__(self, regularization=None, alpha=0.01):
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        self.regularization = regularization
        self.alpha = alpha
        
    def _create_interaction_features(self, X):
        """Create interaction features with specific combinations from code 1"""
        X = np.array(X)
        
        # Start with original features
        features = [X]
        
        # Cải tiến từ code 1: Specific interactions
        if hasattr(self, 'feature_names') and self.feature_names:
            for i, name1 in enumerate(self.feature_names):
                for j, name2 in enumerate(self.feature_names[i+1:], i+1):
                    # Specific interactions like Age*Engine, Power*Torque, Length*Width
                    if (('Age' in str(name1) and 'Engine' in str(name2)) or
                        ('Power' in str(name1) and 'Torque' in str(name2)) or
                        ('Length' in str(name1) and 'Width' in str(name2))):
                        interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                        features.append(interaction)
        else:
            # General pairwise interactions (limited to avoid overfitting)
            n_features = X.shape[1]
            for i in range(min(3, n_features)):  # Limit interactions
                for j in range(i+1, min(i+3, n_features)):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features.append(interaction)
        
        return np.column_stack(features)
    
    def fit(self, X, y):
        """Fit interaction regression model"""
        X = np.array(X)
        y = np.array(y)
        
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Create interaction features
        X_interact = self._create_interaction_features(X)
        
        # Add bias column
        X_with_bias = np.column_stack([np.ones(X_interact.shape[0]), X_interact])
        
        # Normal equation with regularization
        try:
            if self.regularization == 'l2':
                I = np.eye(X_with_bias.shape[1])
                I[0, 0] = 0
                theta = np.linalg.solve(X_with_bias.T @ X_with_bias + self.alpha * I, 
                                      X_with_bias.T @ y)
            else:
                theta = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
        except np.linalg.LinAlgError:
            if self.regularization == 'l2':
                I = np.eye(X_with_bias.shape[1])
                I[0, 0] = 0
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias + self.alpha * I) @ X_with_bias.T @ y
            else:
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        X_interact = self._create_interaction_features(X)
        return X_interact @ self.coefficients + self.intercept
    
    def get_equation(self):
        """Return equation representation"""
        if self.coefficients is None:
            return "Model not fitted yet"
        
        return f"y = {self.intercept:.4f} + linear_terms + interaction_terms (total {len(self.coefficients)} features)"

class HybridRegression:
    """Hybrid Regression với cải tiến từ code 1"""
    
    def __init__(self, regularization=None, alpha=0.01):
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        self.regularization = regularization
        self.alpha = alpha
        
    def _create_hybrid_features(self, X):
        """Create hybrid features combining techniques from code 1"""
        X = np.array(X)
        features = []
        
        # Original features
        features.append(X)
        
      
        if hasattr(self, 'feature_names') and self.feature_names:
            # Specific transformations for car data
            for i, name in enumerate(self.feature_names):
                if 'Kilometer' in str(name):
                    # Square root transformation for kilometers
                    safe_values = np.maximum(X[:, i], 1e-8)
                    features.append(np.sqrt(safe_values).reshape(-1, 1))
                elif 'Engine' in str(name):
                    # Log transformation for engine
                    safe_values = np.maximum(X[:, i], 1e-8)
                    features.append(np.log1p(safe_values).reshape(-1, 1))
        
        # Add some quadratic terms (first few features)
        n_features = X.shape[1]
        n_quad = min(3, n_features)
        features.append(X[:, :n_quad]**2)
        
        # Add selected interaction terms
        if n_features >= 2:
            for i in range(min(2, n_features-1)):
                for j in range(i+1, min(i+3, n_features)):
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    features.append(interaction)
        
        return np.column_stack(features)
    
    def fit(self, X, y):
        """Fit hybrid regression model"""
        X = np.array(X)
        y = np.array(y)
        
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Create hybrid features
        X_hybrid = self._create_hybrid_features(X)
        
        # Add bias column
        X_with_bias = np.column_stack([np.ones(X_hybrid.shape[0]), X_hybrid])
        
        # Normal equation with regularization
        try:
            if self.regularization == 'l2':
                I = np.eye(X_with_bias.shape[1])
                I[0, 0] = 0
                theta = np.linalg.solve(X_with_bias.T @ X_with_bias + self.alpha * I, 
                                      X_with_bias.T @ y)
            else:
                theta = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
        except np.linalg.LinAlgError:
            if self.regularization == 'l2':
                I = np.eye(X_with_bias.shape[1])
                I[0, 0] = 0
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias + self.alpha * I) @ X_with_bias.T @ y
            else:
                theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        X_hybrid = self._create_hybrid_features(X)
        return X_hybrid @ self.coefficients + self.intercept
    
    def get_equation(self):
        """Return equation representation"""
        if self.coefficients is None:
            return "Model not fitted yet"
        
        return f"y = {self.intercept:.4f} + linear + quadratic + interaction + transformed terms (total {len(self.coefficients)} features)"

def compare_models(X_train, y_train, X_test=None, y_test=None, regularization=None, alpha=0.01):
    """Compare all regression models with comprehensive evaluation"""
    models = {
        "Linear": LinearRegression(regularization=regularization, alpha=alpha),
        "Quadratic": QuadraticRegression(regularization=regularization, alpha=alpha), 
        "Interaction": InteractionRegression(regularization=regularization, alpha=alpha),
        "Hybrid": HybridRegression(regularization=regularization, alpha=alpha)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n=== {name} Regression ===")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Train predictions and evaluation
        y_train_pred = model.predict(X_train)
        train_mse, train_rmse, train_mae = evaluate_model(y_train, y_train_pred, f"{name} (Training)")
        
        print(f"Equation: {model.get_equation()}")
        
        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae
        }
        
        # Test predictions if test data provided
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            test_mse, test_rmse, test_mae = evaluate_model(y_test, y_test_pred, f"{name} (Testing)")
            
            results[name].update({
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae
            })
    
    return results

def load_and_evaluate_from_csv(csv_path, trained_models, target_column='Price'):
    """
    Load CSV file and evaluate trained models
    Cải tiến từ yêu cầu đề bài
    """
    try:
        # Load data
        data = pd.read_csv(csv_path)
        print(f"Loaded {len(data)} samples from {csv_path}")
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            print(f"Warning: Target column '{target_column}' not found. Using last column as target.")
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        
        # Evaluate each model
        results = {}
        for name, model in trained_models.items():
            try:
                y_pred = model.predict(X)
                mse, rmse, mae = evaluate_model(y, y_pred, f"{name} on {csv_path}")
                results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae}
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = {'Error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"Error loading {csv_path}: {str(e)}")
        return None

if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    # Create sample data with feature names
    feature_names = ['Age', 'Engine', 'Max Power', 'Kilometer']
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
    
    # Create synthetic target with known relationship
    y = (2*X['Age'] + 3*X['Engine']**2 + X['Age']*X['Engine'] + 
         0.5*np.sqrt(np.maximum(X['Kilometer'], 0.1)) + np.random.randn(n_samples)*0.1)
    
    print("Testing improved regression models...")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Compare models
    results = compare_models(X_train, y_train, X_test, y_test)
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['test_mse'])
    print(f"\nBest model: {best_model} (Test MSE: {results[best_model]['test_mse']:.4f})")
    
    # Example of loading from CSV (uncomment when you have actual data)
    # trained_models = {name: results[name]['model'] for name in results.keys()}
    # csv_results = load_and_evaluate_from_csv('val.csv', trained_models)