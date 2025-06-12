import numpy as np
import pandas as pd
from linear_models import LinearRegression, QuadraticRegression, InteractionRegression, HybridRegression

class NoiseGenerator:
    """Noise generation mechanisms for differential privacy"""
    
    @staticmethod
    def laplace_noise(sensitivity, epsilon, size=1):
        """
        Generate Laplace noise for differential privacy
        
        Parameters:
        -----------
        sensitivity : float
            Global sensitivity of the function
        epsilon : float  
            Privacy parameter (smaller = more private)
        size : int or tuple
            Shape of noise to generate
            
        Returns:
        --------
        numpy.ndarray
            Laplace noise
        """
        scale = sensitivity / epsilon
        return np.random.laplace(0, scale, size)
    
    @staticmethod
    def gaussian_noise(sensitivity, epsilon, delta=1e-5, size=1):
        """
        Generate Gaussian noise for differential privacy
        Note: Only valid for epsilon < 1
        
        Parameters:
        -----------
        sensitivity : float
            Global sensitivity of the function
        epsilon : float
            Privacy parameter (must be < 1)
        delta : float
            Privacy parameter for (ε,δ)-DP
        size : int or tuple
            Shape of noise to generate
            
        Returns:
        --------
        numpy.ndarray
            Gaussian noise
        """
        if epsilon >= 1:
            raise ValueError("Gaussian mechanism only valid for epsilon < 1")
        
        # Calculate sigma for (ε,δ)-DP
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return np.random.normal(0, sigma, size)


class PrivacyAccountant:
    """Privacy accounting for composition of DP mechanisms"""
    
    def __init__(self):
        self.privacy_spent = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0
    
    def add_privacy_cost(self, epsilon, delta=0.0, description=""):
        """Add privacy cost from a single mechanism"""
        self.privacy_spent.append({
            'epsilon': epsilon,
            'delta': delta, 
            'description': description
        })
        self.total_epsilon += epsilon
        self.total_delta += delta
    
    def basic_composition(self):
        """Basic composition theorem: sum of epsilons and deltas"""
        return self.total_epsilon, self.total_delta
    
    def advanced_composition(self, k=None):
        """
        Advanced composition theorem for better bounds
        
        Parameters:
        -----------
        k : int
            Number of mechanisms (if None, use actual count)
            
        Returns:
        --------
        tuple
            (epsilon, delta) under advanced composition
        """
        if k is None:
            k = len(self.privacy_spent)
        
        if k == 0:
            return 0.0, 0.0
        
        # For simplicity, assume all mechanisms have same epsilon
        avg_epsilon = self.total_epsilon / k if k > 0 else 0
        avg_delta = self.total_delta / k if k > 0 else 0
        
        # Advanced composition: ε' = √(2k ln(1/δ')) * ε + k * ε * (e^ε - 1)
        # Simplified version for small epsilon
        delta_prime = min(avg_delta, 1e-5)
        epsilon_prime = np.sqrt(2 * k * np.log(1/delta_prime)) * avg_epsilon + k * avg_epsilon * (np.exp(avg_epsilon) - 1)
        
        return epsilon_prime, self.total_delta
    
    def get_privacy_report(self):
        """Generate privacy accounting report"""
        basic_eps, basic_delta = self.basic_composition()
        advanced_eps, advanced_delta = self.advanced_composition()
        
        report = f"""
Privacy Accounting Report:
=========================
Total mechanisms used: {len(self.privacy_spent)}

Basic Composition:
- Total ε: {basic_eps:.6f}
- Total δ: {basic_delta:.6f}

Advanced Composition:
- Total ε: {advanced_eps:.6f}  
- Total δ: {advanced_delta:.6f}

Individual mechanisms:
"""
        for i, cost in enumerate(self.privacy_spent):
            report += f"  {i+1}. {cost['description']}: ε={cost['epsilon']:.6f}, δ={cost['delta']:.6f}\n"
        
        return report


class PrivacyPreservingRegression:
    """Differential Privacy for Linear Regression"""
    
    def __init__(self, model_type="linear", privacy_approach="user_level"):
        """
        Parameters:
        -----------
        model_type : str
            Type of regression model ('linear', 'quadratic', 'interaction', 'hybrid')
        privacy_approach : str
            'user_level' or 'server_level'
        """
        self.model_type = model_type
        self.privacy_approach = privacy_approach
        self.model = None
        self.accountant = PrivacyAccountant()
        self.noise_generator = NoiseGenerator()
        
        # Initialize model
        models = {
            "linear": LinearRegression(),
            "quadratic": QuadraticRegression(), 
            "interaction": InteractionRegression(),
            "hybrid": HybridRegression()
        }
        
        if model_type not in models:
            raise ValueError(f"Invalid model type: {model_type}")
        
        self.model = models[model_type]
    
    def _calculate_sensitivity(self, X, y, data_bounds=None):
        """
        Calculate sensitivity for regression coefficients
        Simplified calculation - in practice this requires careful analysis
        """
        if data_bounds is None:
            # Estimate bounds from data
            x_max = np.max(np.abs(X), axis=0)
            y_max = np.max(np.abs(y))
            data_bounds = {'x_max': x_max, 'y_max': y_max}
        
        # Simplified sensitivity calculation
        # For linear regression, sensitivity depends on data bounds and regularization
        sensitivity = np.sum(data_bounds['x_max']) * data_bounds['y_max'] / len(X)
        
        return sensitivity
    
    def fit_with_user_level_dp(self, X, y, epsilon, noise_type="laplace", data_bounds=None):
        """
        User-level DP: Add noise to data before training
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like  
            Target vector
        epsilon : float
            Privacy parameter
        noise_type : str
            'laplace' or 'gaussian'
        data_bounds : dict
            Bounds on data values for sensitivity calculation
        """
        X = np.array(X)
        y = np.array(y)
        
        # Calculate sensitivity
        sensitivity = self._calculate_sensitivity(X, y, data_bounds)
        
        # Add noise to features
        if noise_type == "laplace":
            X_noisy = X + self.noise_generator.laplace_noise(
                sensitivity, epsilon/2, X.shape
            )
            y_noisy = y + self.noise_generator.laplace_noise(
                sensitivity, epsilon/2, y.shape
            )
        elif noise_type == "gaussian" and epsilon < 1:
            X_noisy = X + self.noise_generator.gaussian_noise(
                sensitivity, epsilon/2, size=X.shape
            )
            y_noisy = y + self.noise_generator.gaussian_noise(
                sensitivity, epsilon/2, size=y.shape
            )
        else:
            raise ValueError("Invalid noise type or epsilon >= 1 for Gaussian")
        
        # Train model on noisy data
        self.model.fit(X_noisy, y_noisy)
        
        # Record privacy cost
        self.accountant.add_privacy_cost(
            epsilon, 0.0 if noise_type == "laplace" else 1e-5,
            f"User-level DP training with {noise_type} noise"
        )
        
        return self
    
    def predict_with_server_level_dp(self, X, epsilon, noise_type="laplace"):
        """
        Server-level DP: Add noise to predictions
        
        Parameters:
        -----------
        X : array-like
            Feature matrix for prediction
        epsilon : float
            Privacy parameter for this query
        noise_type : str
            'laplace' or 'gaussian'
            
        Returns:
        --------
        numpy.ndarray
            Noisy predictions
        """
        if self.model.coefficients is None:
            raise ValueError("Model must be fitted first")
        
        # Get clean predictions
        clean_predictions = self.model.predict(X)
        
        # Estimate sensitivity for predictions (simplified)
        prediction_range = np.max(clean_predictions) - np.min(clean_predictions)
        sensitivity = prediction_range / len(X)  # Simplified
        
        # Add noise to predictions
        if noise_type == "laplace":
            noise = self.noise_generator.laplace_noise(
                sensitivity, epsilon, clean_predictions.shape
            )
        elif noise_type == "gaussian" and epsilon < 1:
            noise = self.noise_generator.gaussian_noise(
                sensitivity, epsilon, size=clean_predictions.shape
            )
        else:
            raise ValueError("Invalid noise type or epsilon >= 1 for Gaussian")
        
        noisy_predictions = clean_predictions + noise
        
        # Record privacy cost
        self.accountant.add_privacy_cost(
            epsilon, 0.0 if noise_type == "laplace" else 1e-5,
            f"Server-level DP prediction with {noise_type} noise"
        )
        
        return noisy_predictions
    
    def fit(self, X, y, epsilon, **kwargs):
        """Fit model with chosen privacy approach"""
        if self.privacy_approach == "user_level":
            return self.fit_with_user_level_dp(X, y, epsilon, **kwargs)
        else:
            # For server-level, train normally first
            self.model.fit(X, y)
            return self
    
    def predict(self, X, epsilon=None, **kwargs):
        """Make predictions with chosen privacy approach"""
        if self.privacy_approach == "server_level" and epsilon is not None:
            return self.predict_with_server_level_dp(X, epsilon, **kwargs)
        else:
            return self.model.predict(X)
    
    def get_privacy_report(self):
        """Get privacy accounting report"""
        return self.accountant.get_privacy_report()


def analyze_privacy_utility_tradeoff(X_train, y_train, X_test, y_test, 
                                   epsilon_values, model_type="linear",
                                   privacy_approach="user_level"):
    """
    Analyze privacy-utility tradeoff across different epsilon values
    
    Parameters:
    -----------
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like  
        Test data
    epsilon_values : list
        List of epsilon values to test
    model_type : str
        Type of regression model
    privacy_approach : str
        'user_level' or 'server_level'
        
    Returns:
    --------
    dict
        Results for each epsilon value
    """
    results = {}
    
    print(f"\nAnalyzing Privacy-Utility Tradeoff")
    print(f"Model: {model_type}, Approach: {privacy_approach}")
    print("=" * 50)
    
    for eps in epsilon_values:
        print(f"\nTesting ε = {eps}")
        
        # Categorize epsilon
        if eps < 1:
            category = "High Privacy (ε < 1)"
            noise_type = "gaussian"
        elif 1 <= eps <= 10:
            category = "Medium Privacy (1 ≤ ε ≤ 10)"
            noise_type = "laplace"
        else:
            category = "Low Privacy (ε > 10)"
            noise_type = "laplace"
        
        print(f"Category: {category}")
        
        # Create DP model
        dp_model = PrivacyPreservingRegression(
            model_type=model_type,
            privacy_approach=privacy_approach
        )
        
        try:
            # Fit model
            if privacy_approach == "user_level":
                dp_model.fit(X_train, y_train, eps, noise_type=noise_type)
                y_pred = dp_model.predict(X_test)
            else:
                dp_model.fit(X_train, y_train, eps)
                y_pred = dp_model.predict(X_test, eps, noise_type=noise_type)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            
            results[eps] = {
                'category': category,
                'noise_type': noise_type,
                'mse': mse,
                'mae': mae,
                'privacy_report': dp_model.get_privacy_report()
            }
            
            print(f"MSE: {mse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"Noise type: {noise_type}")
            
        except Exception as e:
            print(f"Error with ε={eps}: {str(e)}")
            results[eps] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 4
    
    X = np.random.randn(n_samples, n_features)
    y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(n_samples)*0.5
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Testing Differential Privacy for Regression")
    print("=" * 50)
    
    # Test epsilon values
    epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0, 15.0]
    
    # Test user-level DP
    print("\n1. USER-LEVEL DIFFERENTIAL PRIVACY")
    user_results = analyze_privacy_utility_tradeoff(
        X_train, y_train, X_test, y_test,
        epsilon_values, "linear", "user_level"
    )
    
    # Test server-level DP  
    print("\n\n2. SERVER-LEVEL DIFFERENTIAL PRIVACY")
    server_results = analyze_privacy_utility_tradeoff(
        X_train, y_train, X_test, y_test,
        epsilon_values, "linear", "server_level"
    )
    
    # Summary
    print("\n\nSUMMARY")
    print("=" * 50)
    print("User-level DP Results:")
    for eps, result in user_results.items():
        if 'mse' in result:
            print(f"  ε={eps}: MSE={result['mse']:.2f}, Category={result['category']}")
    
    print("\nServer-level DP Results:")
    for eps, result in server_results.items():
        if 'mse' in result:
            print(f"  ε={eps}: MSE={result['mse']:.2f}, Category={result['category']}")