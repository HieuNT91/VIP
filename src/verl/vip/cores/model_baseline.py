import numpy as np
import argparse
import json
import os
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Models for Regression Tasks")
    
    # Data paths
    parser.add_argument('--embedder', type=str, default="Qwen-Qwen3-Embedding-0.6B",
                        help='Embedder name')
    parser.add_argument('--dataset', type=str, default="fixprompt-dapo-math-17k_17398",
                        help='Dataset name')
    parser.add_argument('--regression_data', type=str, default="allo_grpo_4e",
                        help='Regression data folder')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['ridge', 'lasso', 'logistic'], default='ridge',
                        help='Type of regression model to use')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Regularization strength for Ridge or Lasso')
    parser.add_argument('--window_size', type=int, default=1,
                        help='Window size for training data')
    parser.add_argument('--target_key', type=str, default="mean_acc_per_epoch",
                        help='Target key in regression data')
    
    # Experiment settings
    parser.add_argument('--start_step', type=int, default=1,
                        help='Start step for evaluation')
    parser.add_argument('--end_step', type=int, default=66,
                        help='End step for evaluation')
    parser.add_argument('--step_size', type=int, default=1,
                        help='Step size for evaluation')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default="/home/hieunt/verl/results/gbr/baselines",
                        help='Output directory for baseline results')
    
    return parser.parse_args()

def get_model(model_type, alpha):
    """Initialize model based on model_type"""
    if model_type == 'ridge':
        return Ridge(alpha=alpha)
    elif model_type == 'lasso':
        return Lasso(alpha=alpha)
    elif model_type == 'logistic':
        return LogisticRegression(C=1/alpha, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    args = parse_args()
    
    # Construct paths based on arguments
    embedding_path = f"/home/hieunt/verl/data/embedding_data/embeddings_{args.embedder}_{args.dataset}.npy"
    pairwise_path = f"/home/hieunt/verl/data/embedding_data/pairwise_{args.embedder}_{args.dataset}_matrix.npy"
    indices_path = f"/home/hieunt/verl/data/embedding_data/indices_{args.embedder}_{args.dataset}.json"
    regression_json_path = f"/home/hieunt/verl/data/regression_data/{args.regression_data}/per_question_statistics_latest.json"
    
    # Check if files exist
    for path in [embedding_path, pairwise_path, indices_path, regression_json_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    print(f"Loading data from:")
    print(f"  Embedding path: {embedding_path}")
    print(f"  Indices path: {indices_path}")
    print(f"  Regression path: {regression_json_path}")
    
    from time_data_simulator import TimeDataSimulator, TimeDataSimulatorConfig
    cfg = TimeDataSimulatorConfig(
        embedding_path=embedding_path,
        pairwise_path=pairwise_path,
        indices_path=indices_path,
        regression_json_path=regression_json_path,
        batch_size=256,
    )
    
    # Initialize simulator
    sim = TimeDataSimulator(cfg)
    
    # Initialize scaler for feature standardization
    scaler = StandardScaler()
    
    # Track metrics across steps
    step_to_metrics = {}
    
    print(f"Running {args.model_type.capitalize()} Regression with alpha={args.alpha}")
    
    # Run evaluation for each step
    for step in range(args.start_step, args.end_step + 1, args.step_size):
        # Get data for current step
        out = sim.get_train_test_features(
            step=step,
            window_size=args.window_size,
            target_key=args.target_key,
        )
        X_train, y_train = out["train"]["X"], out["train"]["y"]
        X_test, y_test = out["test"]["X"], out["test"]["y"]
        
        # Standardize features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize and train model
        model = get_model(args.model_type, args.alpha)
        
        # For logistic regression, we need binary targets
        if args.model_type == 'logistic':
            # Use median as threshold for binary classification
            threshold = np.median(y_train)
            y_train_binary = (y_train > threshold).astype(int)
            model.fit(X_train_scaled, y_train_binary)
            
            # Get probabilities for the positive class
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Scale probabilities to match the original target range
            y_min, y_max = np.min(y_train), np.max(y_train)
            y_pred = y_min + y_pred_proba * (y_max - y_min)
        else:
            # For ridge and lasso, just fit and predict directly
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        almost_close_count = np.sum(np.abs(y_test - y_pred) <= 0.1)
        almost_close_ratio = almost_close_count / len(y_test)
        almost_close_count_15 = np.sum(np.abs(y_test - y_pred) <= 0.15)
        almost_close_ratio_15 = almost_close_count_15 / len(y_test)

        print(f"Step {step} - MSE: {mse:.4f}, R2: {r2:.4f}, Almost Close Ratio: {almost_close_ratio:.4f}")
        
        # Store metrics
        step_to_metrics[step] = {
            "mse": float(mse),
            "r2": float(r2),
            "almost_close_ratio_0.1": float(almost_close_ratio),
            "almost_close_ratio_0.15": float(almost_close_ratio_15),
            "min_pred": float(np.min(y_pred)),
            "max_pred": float(np.max(y_pred)),
            "std_pred": float(np.std(y_pred)),
        }
    
    # Calculate average metrics for first and second half
    mid_step = (args.start_step + args.end_step) // 2
    first_half_metrics = {}
    second_half_metrics = {}
    
    for k in step_to_metrics[args.start_step].keys():
        first_half_values = [step_to_metrics[j][k] for j in range(args.start_step, mid_step + 1, args.step_size)]
        second_half_values = [step_to_metrics[j][k] for j in range(mid_step + 1, args.end_step + 1, args.step_size)]
        
        first_half_avg = np.mean(first_half_values)
        second_half_avg = np.mean(second_half_values)
        
        print(f"{k}: {first_half_avg:.4f}, {second_half_avg:.4f}")
        first_half_metrics[k] = float(first_half_avg)
        second_half_metrics[k] = float(second_half_avg)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = f"{args.model_type}_a{args.alpha}_{args.embedder}_{args.dataset}_w{args.window_size}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "config": vars(args),
        "first_half_metrics": first_half_metrics,
        "second_half_metrics": second_half_metrics,
        "step_to_metrics": step_to_metrics,
    }
    
    output_path = os.path.join(args.output_dir, f"{config_name}_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()