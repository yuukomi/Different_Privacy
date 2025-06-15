import numpy as np
import pandas as pd
from linear_models import LinearRegression, QuadraticRegression, InteractionRegression, HybridRegression

class NoiseGenerator:
    """Noise generation mechanisms for differential privacy"""

    MIN_EPSILON = 1e-8
    
    @staticmethod
    def _validate_epsilon(epsilon):
        """Validate epsilon parameter"""
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if epsilon < NoiseGenerator.MIN_EPSILON:
            raise ValueError(f"Epsilon too small (< {NoiseGenerator.MIN_EPSILON}), would cause numerical instability")
        return True
    
    @staticmethod
    def _validate_sensitivity(sensitivity):
        """Validate sensitivity parameter"""
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        return True
    
    @staticmethod
    def laplace_noise(sensitivity, epsilon, size=1):
        """
        Generate Laplace noise for differential privacy
        Parameters:
        -----------
        sensitivity : float
            Global sensitivity of the function (must be > 0)
        epsilon : float  
            Privacy parameter (must be > 0, smaller = more private)
        size : int or tuple
            Shape of noise to generate
            
        Returns:
        --------
        numpy.ndarray
            Laplace noise
            
        Raises:
        --------
        ValueError
            If epsilon <= 0 or sensitivity <= 0
        """
        NoiseGenerator._validate_epsilon(epsilon)
        NoiseGenerator._validate_sensitivity(sensitivity)
        
        scale = sensitivity / epsilon
        
        # Additional check for numerical stability
        if scale > 1e6:
            raise ValueError(f"Noise scale too large ({scale:.2e}), consider increasing epsilon")
        
        return np.random.laplace(0, scale, size)
    
    @staticmethod
    def gaussian_noise(sensitivity, epsilon, delta=1e-5, size=1):
        NoiseGenerator._validate_epsilon(epsilon)
        NoiseGenerator._validate_sensitivity(sensitivity)
        
        if epsilon >= 1:
            raise ValueError(f"Gaussian mechanism only valid for epsilon < 1, got {epsilon}")
        
        if delta <= 0:
            raise ValueError(f"Delta must be positive, got {delta}")
        
        if delta >= 1:
            raise ValueError(f"Delta must be < 1, got {delta}")
        
        # Calculate sigma for (ε,δ)-DP
        try:
            log_term = np.log(1.25 / delta)
            if log_term <= 0:
                raise ValueError(f"Invalid delta value {delta}, log(1.25/delta) must be positive")
            
            sigma = sensitivity * np.sqrt(2 * log_term) / epsilon
            
            # Additional check for numerical stability
            if sigma > 1e6:
                raise ValueError(f"Noise sigma too large ({sigma:.2e}), consider increasing epsilon or decreasing sensitivity")
                
        except (ValueError, OverflowError) as e:
            raise ValueError(f"Error calculating Gaussian noise parameters: {str(e)}")
        
        return np.random.normal(0, sigma, size)


class PrivacyAccountant:
    """Privacy accounting for composition of DP mechanisms"""
    
    def __init__(self):
        self.privacy_spent = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0
    
    def add_privacy_cost(self, epsilon, delta=0.0, description=""):
        """Add privacy cost from a single mechanism"""
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if delta < 0:
            raise ValueError(f"Delta must be non-negative, got {delta}")
            
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
        
        if k is None:
            k = len(self.privacy_spent)
        
        if k == 0:
            return 0.0, 0.0
        
        # For simplicity, assume all mechanisms have same epsilon
        avg_epsilon = self.total_epsilon / k if k > 0 else 0
        avg_delta = self.total_delta / k if k > 0 else 0
        
        if avg_epsilon <= 0:
            return 0.0, self.total_delta
        
        delta_prime = max(avg_delta, 1e-10)  # Ensure delta_prime > 0
        
        try:
            log_term = np.log(1/delta_prime)
            sqrt_term = np.sqrt(2 * k * log_term) * avg_epsilon
            exp_term = k * avg_epsilon * (np.exp(avg_epsilon) - 1)
            epsilon_prime = sqrt_term + exp_term
        except (ValueError, OverflowError):
            epsilon_prime = self.total_epsilon
        
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
        if data_bounds is None:
            # Estimate bounds from data
            x_max = np.max(np.abs(X), axis=0)
            y_max = np.max(np.abs(y))
            data_bounds = {'x_max': x_max, 'y_max': y_max}
        
        # Simplified sensitivity calculation
        # For linear regression, sensitivity depends on data bounds and regularization
        x_max_sum = np.sum(data_bounds['x_max'])
        y_max = data_bounds['y_max']
        n = len(X)
        
        if n == 0:
            raise ValueError("Cannot calculate sensitivity for empty dataset")
        
        sensitivity = (x_max_sum * y_max) / n
        
        # Ensure sensitivity is positive and reasonable
        if sensitivity <= 0:
            sensitivity = 1.0  # Default fallback
        
        # Cap sensitivity to prevent extremely large noise
        sensitivity = min(sensitivity, 1000.0)
        
        return sensitivity
    
    def fit_with_user_level_dp(self, X, y, epsilon, noise_type="laplace", data_bounds=None):
        # Validate epsilon early
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Cannot fit model on empty dataset")
        
        # Calculate sensitivity
        try:
            sensitivity = self._calculate_sensitivity(X, y, data_bounds)
        except Exception as e:
            raise ValueError(f"Error calculating sensitivity: {str(e)}")
        
        # Split epsilon budget between features and targets
        eps_x = epsilon / 2
        eps_y = epsilon / 2
        
        # Add noise to features
        try:
            if noise_type == "laplace":
                X_noisy = X + self.noise_generator.laplace_noise(
                    sensitivity, eps_x, X.shape
                )
                y_noisy = y + self.noise_generator.laplace_noise(
                    sensitivity, eps_y, y.shape
                )
            elif noise_type == "gaussian":
                if epsilon >= 1:
                    raise ValueError(f"Gaussian mechanism requires epsilon < 1, got {epsilon}")
                X_noisy = X + self.noise_generator.gaussian_noise(
                    sensitivity, eps_x, size=X.shape
                )
                y_noisy = y + self.noise_generator.gaussian_noise(
                    sensitivity, eps_y, size=y.shape
                )
            else:
                raise ValueError(f"Invalid noise type: {noise_type}. Must be 'laplace' or 'gaussian'")
        except Exception as e:
            raise ValueError(f"Error generating noise: {str(e)}")
        
        # Train model on noisy data
        try:
            self.model.fit(X_noisy, y_noisy)
        except Exception as e:
            raise ValueError(f"Error fitting model: {str(e)}")
        
        # Record privacy cost
        delta_val = 0.0 if noise_type == "laplace" else 1e-5
        self.accountant.add_privacy_cost(
            epsilon, delta_val,
            f"User-level DP training with {noise_type} noise"
        )
        
        return self
    
    def predict_with_server_level_dp(self, X, epsilon, noise_type="laplace"):
        if self.model.coefficients is None:
            raise ValueError("Model must be fitted first")
        
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        X = np.array(X)
        if len(X) == 0:
            raise ValueError("Cannot predict on empty dataset")
        
        # Get clean predictions
        try:
            clean_predictions = self.model.predict(X)
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")
        
        # Estimate sensitivity for predictions (simplified)
        if len(clean_predictions) == 0:
            return clean_predictions
        
        prediction_range = np.max(clean_predictions) - np.min(clean_predictions)
        if prediction_range <= 0:
            prediction_range = 1.0  # Default if all predictions are the same
        
        sensitivity = prediction_range / len(X)  # Simplified
        sensitivity = max(sensitivity, 0.01)  # Minimum sensitivity
        sensitivity = min(sensitivity, 100.0)  # Maximum sensitivity
        
        # Add noise to predictions
        try:
            if noise_type == "laplace":
                noise = self.noise_generator.laplace_noise(
                    sensitivity, epsilon, clean_predictions.shape
                )
            elif noise_type == "gaussian":
                if epsilon >= 1:
                    raise ValueError(f"Gaussian mechanism requires epsilon < 1, got {epsilon}")
                noise = self.noise_generator.gaussian_noise(
                    sensitivity, epsilon, size=clean_predictions.shape
                )
            else:
                raise ValueError(f"Invalid noise type: {noise_type}. Must be 'laplace' or 'gaussian'")
        except Exception as e:
            raise ValueError(f"Error generating prediction noise: {str(e)}")
        
        noisy_predictions = clean_predictions + noise
        
        # Record privacy cost
        delta_val = 0.0 if noise_type == "laplace" else 1e-5
        self.accountant.add_privacy_cost(
            epsilon, delta_val,
            f"Server-level DP prediction with {noise_type} noise"
        )
        
        return noisy_predictions
    
    def fit(self, X, y, epsilon, **kwargs):
        """Fit model with chosen privacy approach"""
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
            
        if self.privacy_approach == "user_level":
            return self.fit_with_user_level_dp(X, y, epsilon, **kwargs)
        else:
            try:
                self.model.fit(X, y)
            except Exception as e:
                raise ValueError(f"Error fitting model: {str(e)}")
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
    results = {}
    
    # Validate epsilon values
    valid_epsilon_values = []
    for eps in epsilon_values:
        if eps <= 0:
            print(f"Warning: Skipping invalid epsilon value {eps} (must be > 0)")
            continue
        if eps < NoiseGenerator.MIN_EPSILON:
            print(f"Warning: Skipping epsilon value {eps} (too small, minimum is {NoiseGenerator.MIN_EPSILON})")
            continue
        valid_epsilon_values.append(eps)
    
    if not valid_epsilon_values:
        raise ValueError("No valid epsilon values provided")
    
    print(f"\nAnalyzing Privacy-Utility Tradeoff")
    print(f"Model: {model_type}, Approach: {privacy_approach}")
    print("=" * 50)
    
    for eps in valid_epsilon_values:
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
        try:
            dp_model = PrivacyPreservingRegression(
                model_type=model_type,
                privacy_approach=privacy_approach
            )
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            results[eps] = {'error': f"Model creation error: {str(e)}"}
            continue
        
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
    
    epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0, 15.0]
    
    try:
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
        print("\n\nSUMMARY")
        print("=" * 50)
        print("User-level DP Results:")
        for eps, result in user_results.items():
            if 'mse' in result:
                print(f"  ε={eps}: MSE={result['mse']:.2f}, Category={result['category']}")
            elif 'error' in result:
                print(f"  ε={eps}: ERROR - {result['error']}")
        
        print("\nServer-level DP Results:")
        for eps, result in server_results.items():
            if 'mse' in result:
                print(f"  ε={eps}: MSE={result['mse']:.2f}, Category={result['category']}")
            elif 'error' in result:
                print(f"  ε={eps}: ERROR - {result['error']}")
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("Please check your input parameters and try again.")