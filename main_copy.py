from data_preprocessing import load_and_preprocess
from linear_models_copy import LinearRegression, QuadraticRegression, InteractionRegression, HybridRegression, evaluate_model
from differential_privacy_copy import PrivacyPreservingRegression, analyze_privacy_utility_tradeoff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_model_comparison(X_train, y_train, X_test, y_test):
    """Compare different regression models without privacy"""
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

def run_comprehensive_privacy_analysis(X_train, y_train, X_test, y_test, 
                                     epsilon_values=[0.1, 0.5, 1.0, 5.0, 10.0]):
    """Comprehensive privacy analysis with different models and approaches"""
    
    # Model types to test
    model_types = ["linear", "quadratic", "interaction", "hybrid"]
    privacy_approaches = ["user_level", "server_level"]
    
    all_results = {}
    
    print("\n" + "="*60)
    print("COMPREHENSIVE PRIVACY ANALYSIS")
    print("="*60)
    
    for model_type in model_types:
        for approach in privacy_approaches:
            print(f"\n--- Testing {model_type.upper()} with {approach.upper()} DP ---")
            
            # Run privacy analysis for this model+approach combination
            results = analyze_privacy_utility_tradeoff(
                X_train, y_train, X_test, y_test,
                epsilon_values, model_type, approach
            )
            
            # Store results with descriptive keys
            for eps, result in results.items():
                if 'mse' in result:
                    key = f"{model_type}_{approach}_eps_{eps}"
                    all_results[key] = {
                        'model_type': model_type,
                        'privacy_approach': approach,
                        'epsilon': eps,
                        'category': result['category'],
                        'noise_type': result['noise_type'],
                        'mse': result['mse'],
                        'mae': result['mae'],
                        'rmse': np.sqrt(result['mse'])
                    }
    
    return all_results

def run_focused_privacy_analysis(X_train, y_train, X_test, y_test, 
                                epsilon_values=[0.1, 1.0, 10.0]):
    """Focused privacy analysis comparing all models with user-level DP"""
    results = {}
    
    # Baseline without privacy
    print("\n" + "="*50)
    print("BASELINE MODELS (NO PRIVACY)")
    print("="*50)
    
    baseline_models = {
        "Linear": "linear",
        "Quadratic": "quadratic", 
        "Interaction": "interaction",
        "Hybrid": "hybrid"
    }
    
    for name, model_type in baseline_models.items():
        print(f"\nTraining {name} Model (No Privacy)...")
        
        # Create non-private model
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "quadratic":
            model = QuadraticRegression()
        elif model_type == "interaction":
            model = InteractionRegression()
        elif model_type == "hybrid":
            model = HybridRegression()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, f"{name} Baseline")
        
        results[f"{name}_baseline"] = {
            'model_type': model_type,
            'privacy_approach': 'none',
            'epsilon': 'inf',
            'category': 'No Privacy',
            'noise_type': 'none',
            'mse': metrics[0],
            'mae': metrics[2],
            'rmse': metrics[1]
        }
    
    # Test with different privacy levels
    print("\n" + "="*50)
    print("PRIVACY-PRESERVING MODELS")
    print("="*50)
    
    for epsilon in epsilon_values:
        print(f"\n--- Testing with ε = {epsilon} ---")
        
        for name, model_type in baseline_models.items():
            print(f"\nTraining {name} Model with DP (ε={epsilon})...")
            
            try:
                # Create DP model
                dp_model = PrivacyPreservingRegression(
                    model_type=model_type,
                    privacy_approach="user_level"
                )
                
                # Determine noise type based on epsilon
                noise_type = "gaussian" if epsilon < 1 else "laplace"
                
                # Fit and predict
                dp_model.fit(X_train, y_train, epsilon, noise_type=noise_type)
                y_pred = dp_model.predict(X_test)
                
                # Calculate metrics
                mse = np.mean((y_test - y_pred) ** 2)
                mae = np.mean(np.abs(y_test - y_pred))
                rmse = np.sqrt(mse)
                
                # Categorize privacy level
                if epsilon < 1:
                    category = "High Privacy"
                elif epsilon <= 10:
                    category = "Medium Privacy"
                else:
                    category = "Low Privacy"
                
                results[f"{name}_eps_{epsilon}"] = {
                    'model_type': model_type,
                    'privacy_approach': 'user_level',
                    'epsilon': epsilon,
                    'category': category,
                    'noise_type': noise_type,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                print(f"  RMSE: {rmse:.3f}, MAE: {mae:.3f}")
                print(f"  Privacy: {category}, Noise: {noise_type}")
                
            except Exception as e:
                print(f"  Error with {name} ε={epsilon}: {str(e)}")
    
    return results

def plot_privacy_comparison(results):
    """Create comprehensive plots for privacy analysis"""
    
    # Convert results to DataFrame for easier plotting
    plot_data = []
    for key, result in results.items():
        plot_data.append({
            'model': key,
            'model_type': result['model_type'],
            'epsilon': result['epsilon'],
            'rmse': result['rmse'],
            'category': result['category']
        })
    
    df = pd.DataFrame(plot_data)
    
    # Plot 1: RMSE vs Epsilon for each model type
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: RMSE vs Epsilon
    plt.subplot(2, 2, 1)
    model_types = df['model_type'].unique()
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, model_type in enumerate(model_types):
        model_data = df[df['model_type'] == model_type]
        
        # Filter out baseline (epsilon='inf')
        model_data_numeric = model_data[model_data['epsilon'] != 'inf']
        if len(model_data_numeric) > 0:
            plt.plot(model_data_numeric['epsilon'], model_data_numeric['rmse'], 
                    'o-', color=colors[i], label=model_type.capitalize())
        
        # Add baseline point
        baseline_data = model_data[model_data['epsilon'] == 'inf']
        if len(baseline_data) > 0:
            plt.axhline(y=baseline_data['rmse'].iloc[0], color=colors[i], 
                       linestyle='--', alpha=0.7, label=f'{model_type.capitalize()} Baseline')
    
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('RMSE')
    plt.title('Privacy-Utility Tradeoff: RMSE vs Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Subplot 2: Bar chart of RMSE by privacy category
    plt.subplot(2, 2, 2)
    category_data = df.groupby(['category', 'model_type'])['rmse'].mean().unstack()
    category_data.plot(kind='bar', ax=plt.gca())
    plt.title('Average RMSE by Privacy Category')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.legend(title='Model Type')
    
    # Subplot 3: Privacy cost visualization
    plt.subplot(2, 2, 3)
    privacy_levels = df[df['epsilon'] != 'inf']['epsilon'].unique()
    privacy_levels = sorted([float(x) for x in privacy_levels])
    
    for i, model_type in enumerate(model_types):
        model_rmse = []
        for eps in privacy_levels:
            model_data = df[(df['model_type'] == model_type) & (df['epsilon'] == eps)]
            if len(model_data) > 0:
                model_rmse.append(model_data['rmse'].iloc[0])
            else:
                model_rmse.append(None)
        
        plt.bar([x + i*0.2 for x in range(len(privacy_levels))], model_rmse, 
                width=0.2, label=model_type.capitalize(), color=colors[i])
    
    plt.xlabel('Epsilon Values')
    plt.ylabel('RMSE')
    plt.title('Model Performance at Different Privacy Levels')
    plt.xticks(range(len(privacy_levels)), [str(x) for x in privacy_levels])
    plt.legend()
    
    # Subplot 4: Privacy categories distribution
    plt.subplot(2, 2, 4)
    category_counts = df['category'].value_counts()
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Privacy Categories')
    
    plt.tight_layout()
    plt.savefig('comprehensive_privacy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPlots saved as 'comprehensive_privacy_analysis.png'")

def save_detailed_results(results, filename='detailed_privacy_results.csv'):
    """Save detailed results to CSV"""
    # Convert results to list of dictionaries
    results_list = []
    for model_key, metrics in results.items():
        results_list.append({
            'Model_Key': model_key,
            'Model_Type': metrics['model_type'],
            'Privacy_Approach': metrics['privacy_approach'],
            'Epsilon': metrics['epsilon'],
            'Privacy_Category': metrics['category'],
            'Noise_Type': metrics['noise_type'],
            'MSE': metrics['mse'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae']
        })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(['Model_Type', 'Epsilon'])
    results_df.to_csv(filename, index=False)
    print(f"\nDetailed results saved to {filename}")

def print_summary(results):
    """Print summary of results"""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    # Group by model type
    model_types = set(r['model_type'] for r in results.values())
    
    for model_type in sorted(model_types):
        print(f"\n{model_type.upper()} MODEL:")
        print("-" * 30)
        
        model_results = {k: v for k, v in results.items() 
                        if v['model_type'] == model_type}
        
        # Sort by epsilon
        sorted_results = sorted(model_results.items(), 
                              key=lambda x: float('inf') if x[1]['epsilon'] == 'inf' 
                                          else float(x[1]['epsilon']))
        
        for key, result in sorted_results:
            eps_str = "∞" if result['epsilon'] == 'inf' else f"{result['epsilon']}"
            print(f"  ε={eps_str:>6}: RMSE={result['rmse']:.3f}, "
                  f"Category={result['category']}")

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    
    print(f"Dataset shape: {X_train.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Choose analysis type
    analysis_type = "focused"  # Change to "comprehensive" for full analysis
    
    if analysis_type == "comprehensive":
        # Run comprehensive analysis (all combinations)
        print("\nRunning comprehensive privacy analysis...")
        privacy_results = run_comprehensive_privacy_analysis(
            X_train, y_train, X_test, y_test
        )
    else:
        # Run focused analysis (user-level DP only)
        print("\nRunning focused privacy analysis...")
        privacy_results = run_focused_privacy_analysis(
            X_train, y_train, X_test, y_test
        )
    
    # Create visualizations
    plot_privacy_comparison(privacy_results)
    
    # Save detailed results
    save_detailed_results(privacy_results)
    
    # Print summary
    print_summary(privacy_results)
    
    print(f"\nExperiment completed!")
    print(f"Total configurations tested: {len(privacy_results)}")
    print("Check 'detailed_privacy_results.csv' and 'comprehensive_privacy_analysis.png' for detailed analysis.")