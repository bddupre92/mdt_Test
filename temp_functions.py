def run_meta_learning(method='bayesian', surrogate=None, selection=None, exploration=0.2, history_weight=0.7):
    """
    Run meta-learning process to find the best optimizer for a given problem.
    
    Args:
        method: Method for meta-learner ('bayesian', 'random', 'genetic')
        surrogate: Surrogate model for meta-learner
        selection: Selection strategy for meta-learner
        exploration: Exploration factor for meta-learner
        history_weight: History weight for meta-learner
        
    Returns:
        Dictionary with results
    """
    logging.info(f"Running meta-learning with method={method}, surrogate={surrogate}, selection={selection}")
    
    # Create results directories
    results_dir = Path('results')
    data_dir = results_dir / 'data'
    plots_dir = results_dir / 'meta'
    
    for directory in [results_dir, data_dir, plots_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Define test functions
    test_suite = create_test_suite()
    selected_functions = {
        'unimodal': ['sphere', 'rosenbrock'],
        'multimodal': ['rastrigin', 'ackley'],
    }
    
    # Prepare benchmark functions
    benchmark_functions = {}
    dim = 30  # Default dimension
    for category, functions in selected_functions.items():
        for func_name in functions:
            func_creator = TEST_FUNCTIONS[func_name]
            bounds = [(-5, 5)] * dim  # Default bounds
            benchmark_functions[func_name] = func_creator(dim, bounds)
    
    # Create optimizers
    bounds = [(-5, 5)] * dim
    optimizers = create_optimizers(dim, bounds)
    
    # Create meta-optimizer with specified parameters
    meta_opt = MetaOptimizer(
        dim=dim,
        bounds=bounds,
        optimizers=optimizers,
        history_file=str(data_dir / 'meta_history.json'),
        selection_file=str(data_dir / 'meta_selection.json'),
        method=method,
        surrogate_model=surrogate,
        selection_strategy=selection,
        exploration_factor=exploration,
        history_weight=history_weight
    )
    
    # Run meta-optimizer on each benchmark function
    results = {}
    best_algorithms = {}
    performance_metrics = {}
    
    for func_name, func in benchmark_functions.items():
        logging.info(f"Running meta-learning on {func_name} function")
        
        # Run meta-optimizer
        best_score, best_solution, history = meta_opt.optimize(
            func, 
            max_evaluations=1000,  # Reduced for quicker results
            verbose=True
        )
        
        # Store results
        results[func_name] = {
            'best_score': best_score,
            'best_solution': best_solution,
            'history': history
        }
        
        # Track which optimizer was selected most often
        optimizer_counts = {}
        for entry in history:
            optimizer = entry.get('selected_optimizer', 'unknown')
            optimizer_counts[optimizer] = optimizer_counts.get(optimizer, 0) + 1
        
        # Determine best algorithm
        best_algorithm = max(optimizer_counts.items(), key=lambda x: x[1])[0]
        best_algorithms[func_name] = best_algorithm
        
        # Calculate performance metrics
        performance_metrics[func_name] = {
            'best_score': best_score,
            'optimizer_selections': optimizer_counts,
            'convergence_rate': len(history) / 1000  # Simple metric
        }
    
    # Determine overall best algorithm
    all_selections = {}
    for func_name, algorithm in best_algorithms.items():
        all_selections[algorithm] = all_selections.get(algorithm, 0) + 1
    
    overall_best_algorithm = max(all_selections.items(), key=lambda x: x[1])[0]
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot optimizer selection frequency
    algorithms = list(all_selections.keys())
    frequencies = [all_selections[alg] for alg in algorithms]
    
    ax.bar(algorithms, frequencies)
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Selection Frequency')
    ax.set_title('Meta-Learner Optimizer Selection Frequency')
    
    # Save plot
    save_plot(fig, 'meta_learner_selection_frequency', plot_type='meta')
    
    return {
        'best_algorithm': overall_best_algorithm,
        'algorithm_selections': best_algorithms,
        'performance': performance_metrics,
        'results': results
    }

def run_evaluation(model=None, X_test=None, y_test=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model to evaluate (if None, creates a default model)
        X_test: Test features (if None, creates synthetic data)
        y_test: Test targets (if None, creates synthetic data)
        
    Returns:
        Dictionary with evaluation results
    """
    logging.info("Running model evaluation")
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create model and data if not provided
    if model is None or X_test is None or y_test is None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Create evaluation plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual vs predicted
    ax.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Model Evaluation: Actual vs Predicted')
    
    # Add metrics to plot
    ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    save_plot(fig, 'model_evaluation', plot_type='evaluation')
    
    return {
        'score': r2,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        },
        'predictions': y_pred.tolist()
    }
