from data_preprocessing import load_and_preprocess
from linear_models import LinearRegression, QuadraticRegression, InteractionRegression, HybridRegression, evaluate_model
from differential_privacy import PrivateDataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_model_comparison(X_train, y_train, X_test, y_test):
    """Compare different regression models"""
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
        y_pred = model.predict(X_test)
        results[name] = evaluate_model(y_test, y_pred, name)
    
    return results

def run_privacy_analysis(X_train, y_train, X_test, y_test, epsilons=[0.1, 1.0, 10.0]):
    """Analyze impact of different privacy levels"""
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
        
        # Create private data
        private_data = PrivateDataLoader(X_train, y_train, epsilon=epsilon)
        X_train_private, X_test_private, y_train_split, y_test_split = private_data.get_train_test_split()
        
        # Test all models with private data
        models = {
            "Basic Linear": LinearRegression(),
            "Quadratic": QuadraticRegression(),
            "Interaction": InteractionRegression(),
            "Hybrid": HybridRegression()
        }
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name} Model with ε={epsilon}...")
            model.fit(X_train_private, y_train_split)
            y_pred = model.predict(X_test_private)
            results[f'{model_name}_epsilon_{epsilon}'] = evaluate_model(y_test_split, y_pred, 
                                                                      f"{model_name} (ε={epsilon})")
    
    return results

def plot_results(results):
    """Plot comparison of model performances"""
    # Extract RMSE values
    model_names = []
    rmse_values = []
    
    for name, metrics in results.items():
        model_names.append(name)
        rmse_values.append(metrics[1])  # RMSE is second value in metrics tuple
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, rmse_values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RMSE')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def save_results(results, filename='results.csv'):
    """Save results to CSV file"""
    # Convert results to list of dictionaries
    results_list = []
    for model_name, metrics in results.items():
        results_list.append({
            'Model': model_name,
            'MSE': metrics[0],
            'RMSE': metrics[1],
            'MAE': metrics[2]
        })
    
    # Create DataFrame directly from list of dictionaries
    results_df = pd.DataFrame(results_list)
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    
    # Run model comparison without privacy
    print("\nComparing different regression models...")
    model_results = run_model_comparison(X_train, y_train, X_test, y_test)
    
    # Run privacy analysis
    print("\nAnalyzing privacy impact...")
    privacy_results = run_privacy_analysis(X_train, y_train, X_test, y_test)
    
    # Combine all results
    all_results = {**model_results, **privacy_results}
    
    # Plot results
    plot_results(all_results)
    
    # Save results
    save_results(all_results)
    
    print("\nExperiment completed! Check results.csv and model_comparison.png for detailed analysis.") 