import numpy as np
from scipy.stats import laplace
import pandas as pd

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0, delta=1e-5):
        """
        Initialize DP mechanism
        epsilon: privacy budget
        delta: privacy failure probability
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = None
        self.noise_scale = None
        
    def compute_sensitivity(self, X):
        """Compute L1 sensitivity for features"""
        # For normalized data, sensitivity is roughly the max possible change
        # when one record is removed/modified
        ranges = np.ptp(X, axis=0)  # Range of each feature
        self.sensitivity = np.max(ranges)
        
        # Scale for Laplace mechanism
        self.noise_scale = self.sensitivity / self.epsilon
        
    def add_laplace_noise(self, X):
        """Add Laplace noise to features"""
        if self.sensitivity is None:
            self.compute_sensitivity(X)
            
        noise = laplace.rvs(loc=0, scale=self.noise_scale, size=X.shape)
        return X + noise
    
    def get_privacy_spent(self, num_queries):
        """Calculate total privacy cost using basic composition"""
        return self.epsilon * num_queries

class PrivateDataLoader:
    def __init__(self, X, y, epsilon=1.0, test_size=0.2):
        self.X = X
        self.y = y
        self.epsilon = epsilon
        self.test_size = test_size
        self.dp = DifferentialPrivacy(epsilon=epsilon)
        
    def normalize_features(self, X):
        """Normalize features to [0,1] range"""
        X_norm = (X - X.min()) / (X.max() - X.min())
        return X_norm
        
    def create_private_data(self):
        """Create differentially private version of the dataset"""
        # Normalize features
        X_norm = self.normalize_features(self.X)
        
        # Add noise to features
        X_private = self.dp.add_laplace_noise(X_norm)
        
        # Convert back to DataFrame
        X_private = pd.DataFrame(X_private, columns=self.X.columns)
        
        return X_private, self.y
    
    def get_train_test_split(self):
        """Get private train/test split"""
        # Create private version of data
        X_private, y = self.create_private_data()
        
        # Calculate split index
        split_idx = int(len(X_private) * (1 - self.test_size))
        
        # Split data
        X_train = X_private[:split_idx]
        X_test = X_private[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test

def evaluate_privacy_impact(epsilons=[0.1, 1.0, 10.0]):
    """Evaluate model performance with different privacy levels"""
    from data_preprocessing import load_and_preprocess
    from linear_models import LinearRegression, evaluate_model
    
    # Load original data
    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    
    # Store results
    results = {}
    
    # Baseline without privacy
    print("\nBaseline (No Privacy):")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results['baseline'] = evaluate_model(y_test, y_pred, "Baseline")
    
    # Test different privacy levels
    for epsilon in epsilons:
        print(f"\nTesting with epsilon = {epsilon}")
        
        # Create private data loader
        private_data = PrivateDataLoader(X_train, y_train, epsilon=epsilon)
        X_train_private, X_test_private, y_train, y_test = private_data.get_train_test_split()
        
        # Train and evaluate model
        model = LinearRegression()
        model.fit(X_train_private, y_train)
        y_pred = model.predict(X_test_private)
        
        results[f'epsilon_{epsilon}'] = evaluate_model(y_test, y_pred, f"DP (ε={epsilon})")
        
        # Calculate privacy spent
        privacy_spent = private_data.dp.get_privacy_spent(num_queries=len(X_train.columns))
        print(f"Total privacy budget spent: {privacy_spent:.2f}")
    
    return results

if __name__ == "__main__":
    # Test privacy mechanisms
    results = evaluate_privacy_impact()
    
    # Print summary
    print("\nPrivacy Impact Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"RMSE: {metrics[1]:.2f}")
        print(f"MAE: {metrics[2]:.2f}") 