from data_preprocessing import load_and_preprocess
from linear_models import LinearRegression, QuadraticRegression, InteractionRegression, HybridRegression, evaluate_model
from differential_privacy import PrivacyPreservingRegression, analyze_privacy_utility_tradeoff
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
    """Create comprehensive plots for privacy analysis with enhanced visuals"""
    
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
    
    # Set a modern style with better colors
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define a better color palette
    colors = {
        'linear': '#3498db',      # Blue
        'quadratic': '#e74c3c',   # Red
        'interaction': '#2ecc71', # Green
        'hybrid': '#f39c12'       # Orange
    }
    
    # Create figure with improved size and resolution
    fig = plt.figure(figsize=(16, 12), dpi=300)
    
    # Subplot 1: RMSE vs Epsilon (Privacy-Utility Tradeoff)
    ax1 = plt.subplot(2, 2, 1)
    model_types = df['model_type'].unique()
    
    # Offset values for label positioning to avoid overlap
    offsets = {'linear': (-15, 10), 'quadratic': (15, 10), 
               'interaction': (-15, -15), 'hybrid': (15, -15)}
    
    for model_type in model_types:
        model_data = df[df['model_type'] == model_type]
        
        # Filter out baseline (epsilon='inf')
        model_data_numeric = model_data[model_data['epsilon'] != 'inf']
        if len(model_data_numeric) > 0:
            line = ax1.plot(model_data_numeric['epsilon'], model_data_numeric['rmse'], 
                    'o-', color=colors[model_type], linewidth=2.5, 
                    markersize=8, label=model_type.capitalize())
            
            # Add value labels at each point with offset based on model type
            for x, y in zip(model_data_numeric['epsilon'], model_data_numeric['rmse']):
                if y > 1e9:  # Nếu giá trị lớn, hiển thị theo định dạng khoa học
                    ax1.annotate(f'{y:.1e}', 
                                (x, y), 
                                textcoords="offset points",
                                xytext=offsets[model_type], 
                                ha='center',
                                fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, 
                                         edgecolor=colors[model_type]))
                else:
                    ax1.annotate(f'{y:.1f}', 
                                (x, y), 
                                textcoords="offset points",
                                xytext=offsets[model_type], 
                                ha='center',
                                fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7,
                                         edgecolor=colors[model_type]))
        
        # Add baseline point
        baseline_data = model_data[model_data['epsilon'] == 'inf']
        if len(baseline_data) > 0:
            ax1.axhline(y=baseline_data['rmse'].iloc[0], color=colors[model_type], 
                       linestyle='--', alpha=0.7, linewidth=1.5,
                       label=f'{model_type.capitalize()} Baseline')
    
    ax1.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('Privacy-Utility Tradeoff: RMSE vs Epsilon', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Subplot 2: Bar chart of RMSE by privacy category with enhanced visuals
    ax2 = plt.subplot(2, 2, 2)
    category_data = df.groupby(['category', 'model_type'])['rmse'].mean().unstack()
    
    # Ensure consistent order of categories
    if 'High Privacy' in category_data.index and 'Medium Privacy' in category_data.index and 'No Privacy' in category_data.index:
        category_data = category_data.reindex(['High Privacy', 'Medium Privacy', 'No Privacy'])
    
    # Create custom color map for the bar chart
    color_list = [colors[model] for model in category_data.columns]
    
    bars = category_data.plot(kind='bar', ax=ax2, color=color_list, width=0.7, edgecolor='black', linewidth=0.5)
    
    # Add value labels vertically on top of each bar (similar to biểu đồ 3)
    for i, container in enumerate(bars.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if height is not None:  # Kiểm tra nếu thanh có giá trị
                if height > 1e9:  # Nếu giá trị lớn, hiển thị theo định dạng khoa học
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02*height,
                           f'{height:.1e}',
                           ha='center', va='bottom', rotation=90, fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7,
                                    edgecolor=colors[category_data.columns[i]]))
                else:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02*height,
                           f'{height:.1f}',
                           ha='center', va='bottom', rotation=90, fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7,
                                    edgecolor=colors[category_data.columns[i]]))
    
    ax2.set_title('Average RMSE by Privacy Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_xlabel('category', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', rotation=45)
    # Increase y-axis limit to make room for vertical labels
    current_ylim = ax2.get_ylim()
    ax2.set_ylim(current_ylim[0], current_ylim[1] * 1.2)
    ax2.legend(title='Model Type', title_fontsize=11, fontsize=10, frameon=True, 
              facecolor='white', edgecolor='gray')
    
    # Subplot 3: Privacy cost visualization with enhanced design
    ax3 = plt.subplot(2, 2, 3)
    privacy_levels = df[df['epsilon'] != 'inf']['epsilon'].unique()
    privacy_levels = sorted([float(x) for x in privacy_levels])
    
    bar_width = 0.18  # Adjusted width for better appearance
    
    for i, model_type in enumerate(model_types):
        model_rmse = []
        for eps in privacy_levels:
            model_data = df[(df['model_type'] == model_type) & (df['epsilon'] == eps)]
            if len(model_data) > 0:
                model_rmse.append(model_data['rmse'].iloc[0])
            else:
                model_rmse.append(None)
        
        bars = ax3.bar([x + i*bar_width for x in range(len(privacy_levels))], model_rmse, 
                width=bar_width, label=model_type.capitalize(), color=colors[model_type],
                edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height is not None:  # Kiểm tra nếu thanh có giá trị
                if height > 1e9:  # Nếu giá trị lớn, hiển thị theo định dạng khoa học
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05*height,
                           f'{height:.1e}',
                           ha='center', va='bottom', rotation=90, fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7,
                                    edgecolor=colors[model_type]))
                else:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05*height,
                           f'{height:.1f}',
                           ha='center', va='bottom', rotation=90, fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7,
                                    edgecolor=colors[model_type]))
    
    ax3.set_xlabel('Epsilon Values', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax3.set_title('Model Performance at Different Privacy Levels', fontsize=14, fontweight='bold')
    ax3.set_xticks([r + bar_width*1.5 for r in range(len(privacy_levels))])
    ax3.set_xticklabels([str(x) for x in privacy_levels])
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    
    # Subplot 4: Privacy categories distribution with better pie chart
    ax4 = plt.subplot(2, 2, 4)
    category_counts = df['category'].value_counts()
    
    # Define better colors for pie chart
    pie_colors = ['#3498db', '#f39c12', '#2ecc71']
    
    wedges, texts, autotexts = ax4.pie(
        category_counts.values, 
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    
    # Customize pie chart text
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Add a legend instead of labels on the pie
    ax4.legend(
        wedges, 
        [f"{cat} ({count})" for cat, count in zip(category_counts.index, category_counts.values)],
        title="Privacy Categories",
        loc="center left",
        bbox_to_anchor=(0.9, 0, 0.5, 1),
        fontsize=10
    )
    
    ax4.set_title('Distribution of Privacy Categories', fontsize=14, fontweight='bold')
    
    # Improve overall layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Save with higher quality
    plt.savefig('comprehensive_privacy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nEnhanced plots saved as 'comprehensive_privacy_analysis.png'")

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