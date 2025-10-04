import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
import argparse
import json
import os
from datetime import datetime
from allocation_v1 import allocate_rollout

def kernel_rbf(d, length_scale=1):
    return np.exp(-0.5 * (d / length_scale) ** 2)

def kernel_rbf_median(d):
    length_scale = np.median(d[np.triu_indices_from(d,k=1)])
    return np.exp(-0.5 * (d / length_scale) ** 2)

def kernel_self_tuning(d, k=7, eps=1e-12, square=False):
    """
    Zelnik-Manor & Perona (2004) self-tuning kernel:
      K_ij = exp(-dist_ij^p / (sigma_i * sigma_j)), p in {1,2}
    sigma_i is distance to the k-th nearest neighbor of i (excluding self, ignoring zeros).
    """
    n = d.shape[0]
    sigmas = np.empty(n, dtype=float)
    for i in range(n):
        row = d[i]
        nz = np.sort(row[row > 0])  # exclude zeros (self and duplicates)
        if nz.size == 0:
            sigmas[i] = 1.0
        else:
            idx = min(k - 1, nz.size - 1)
            sigmas[i] = nz[idx]
    sigmas = np.maximum(sigmas, eps)
    denom = sigmas[:, None] * sigmas[None, :] + eps
    if square:
        K = np.exp(-(d ** 2) / denom)
    else:
        K = np.exp(-(d) / denom)
    np.fill_diagonal(K, 1.0)
    return K

class SequentialGPR:
    def __init__(self, 
                distance_matrix, 
                reuse_covariance=False,
                reuse_mean=False,
                return_std=False, 
                qid_to_idx=None, 
                idx_to_qid=None,
                prior_value=-1,
                 ):
        self.distance_matrix = distance_matrix
        # self.covariance_matrix = kernel_rbf_median(distance_matrix)
        self.covariance_matrix = kernel_rbf(distance_matrix, 0.5)
        self.mean = np.zeros(self.covariance_matrix.shape[0]) + prior_value
        self.reuse_covariance = reuse_covariance
        self.reuse_mean = reuse_mean
        self.return_std = return_std
        self.qid_to_idx = qid_to_idx
        self.idx_to_qid = idx_to_qid
        self.prior_value = prior_value
        
    def _logit(self, p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    
    def _sigmoid(self, f):
        return 1.0 / (1.0 + np.exp(-f))
    
    def fit(self, train_indices, observations):
        if not self.reuse_mean:
            self.mean = np.zeros(self.covariance_matrix.shape[0]) + self.prior_value
        g_t = self._logit(np.clip(observations, a_max=1-1e-6, a_min=1e-6))
        K_in_in = self.covariance_matrix[np.ix_(train_indices, train_indices)]

        L = cholesky(K_in_in + 1e-4 * np.eye(K_in_in.shape[0]), lower=True, check_finite=False)
        alpha = cho_solve((L, True), g_t - self.mean[train_indices], check_finite=False)

        
        all_indices = list(range(self.covariance_matrix.shape[0]))
        K_in_new = self.covariance_matrix[np.ix_(train_indices, all_indices)]
        K_new_in = K_in_new.T
        
        # ALGO: update posterior 
        self.mean[all_indices] = self.mean[all_indices] + K_new_in @ alpha
        self.mean[train_indices] = g_t
        if self.return_std or self.reuse_covariance:
            K_in_in_inv = np.linalg.inv(K_in_in + 1e-4 * np.eye(len(train_indices)))
            V = K_in_in_inv @ K_in_new
            self.std = np.sqrt(np.diag(self.covariance_matrix[np.ix_(all_indices, all_indices)] - K_new_in @ V))
        
        if self.reuse_covariance:
            self.covariance_matrix[np.ix_(all_indices, all_indices)] -= K_new_in @ V

    def fit_qids(self, qids, observations):
        if self.qid_to_idx is None:
            raise ValueError("qid_to_idx mapping is not provided.")
        train_indices = [self.qid_to_idx[qid] for qid in qids if qid in self.qid_to_idx]
        self.fit(train_indices, observations)
        
    def predict(self, indices):
        mean_pred = self._sigmoid(self.mean[indices])
        if self.return_std:
            return mean_pred, self.std[indices]
        else:
            return mean_pred, None
    
    def predict_qids(self, qids):
        if self.qid_to_idx is None:
            raise ValueError("qid_to_idx mapping is not provided.")
        indices = [self.qid_to_idx[qid] for qid in qids if qid in self.qid_to_idx]
        return self.predict(indices)

def acc_to_var(acc):
    return acc * (1 - acc)

def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Gaussian Process Regression for allocation")
    
    # Data paths
    parser.add_argument('--embedder', type=str, default="Qwen-Qwen3-Embedding-0.6B",
                        help='Embedder name')
    parser.add_argument('--dataset', type=str, default="fixprompt-dapo-math-17k_17398",
                        help='Dataset name')
    parser.add_argument('--regression_data', type=str, default="allo_grpo_4e",
                        help='Regression data folder')
    
    # Model parameters
    parser.add_argument('--window_size', type=int, default=1,
                        help='Window size for training data')
    parser.add_argument('--reuse_covariance', action='store_true',
                        help='Reuse covariance matrix between updates')
    parser.add_argument('--reuse_mean', action='store_true',
                        help='Reuse mean between updates')
    parser.add_argument('--return_std', action='store_true',
                        help='Return standard deviation of predictions')
    parser.add_argument('--prior_value', type=float, default=-1,
                        help='Prior value for the mean')
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
    parser.add_argument('--output_dir', type=str, default="/home/hieunt/verl/results/gbr",
                        help='Output directory for results')
    
    return parser.parse_args()

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
    print(f"  Pairwise path: {pairwise_path}")
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
    
    # Load indices and build mappings
    with open(indices_path, "r") as f:
        indices = json.load(f)

    idx_to_qid = {i: qid for i, qid in enumerate(indices)}
    qid_to_idx = {str(qid): i for i, qid in enumerate(indices)}
    
    # Initialize simulator and GPR
    sim = TimeDataSimulator(cfg)
    gpr = SequentialGPR(
        sim.pairwise_matrix, 
        return_std=args.return_std,
        reuse_covariance=args.reuse_covariance,
        reuse_mean=args.reuse_mean,
        qid_to_idx=qid_to_idx,
        idx_to_qid=idx_to_qid,
        prior_value=args.prior_value,
    )
    
    # Run evaluation
    step_to_metrics = {}
    for step in range(args.start_step, args.end_step + 1, args.step_size):
        out = sim.get_train_test_features(
            step=step,
            window_size=args.window_size,
            target_key=args.target_key,
        )
        X_train, P_train, y_train = out["train"]["X"], out["train"]["P"], out["train"]["y"]
        X_test, P_test, y_test = out["test"]["X"], out["test"]["P"], out["test"]["y"]
        qids_train = out["train"]["qids"]
        qids_test = out["test"]["qids"]

        indices_train = out["train"]["indices"]
        indices_test = out["test"]["indices"]
        
        gpr.fit_qids(qids_train, y_train)
        mean_pred_test, cov_pred_test = gpr.predict_qids(qids_test)
        
        budgeted = allocate_rollout(mean_pred_test, 8*256, upper=32)
        budgeted = allocate_rollout(np.round(mean_pred_test,2), 8*256, upper=32)
        for p, b in zip(mean_pred_test, budgeted):
            if p >= 0.5 and b < 16:
                print(f"High pred {p:.4f} but low budget {b}")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.scatter(mean_pred_test, budgeted, alpha=0.7, edgecolors='k')
        plt.xlabel("Predicted mean p")
        plt.ylabel("Allocated budget")
        plt.title("Budget allocation vs predicted p")
        plt.grid(True)

        # save to file (PNG, 300 dpi)
        plt.savefig("budget_vs_p.png", dpi=300, bbox_inches='tight')
        print(np.std(mean_pred_test), np.std(np.round(mean_pred_test, 2)))
        breakpoint()
        from sklearn.metrics import mean_squared_error, r2_score
        mse_test = mean_squared_error(y_test, mean_pred_test)
        r2_test = r2_score(y_test, mean_pred_test)
        almost_close_count = np.sum((abs(y_test - mean_pred_test) <= 0.1))
        almost_close_ratio = almost_close_count / len(y_test)
        almost_close_count_15 = np.sum(np.abs(y_test - mean_pred_test) <= 0.15)
        almost_close_ratio_15 = almost_close_count_15 / len(y_test)
        
        # print(f"Step {step} - Test MSE: {mse_test:.3f}, R2: {r2_test:.3f}, Almost Close Ratio: {almost_close_ratio:.3f}, "
        #       f"Min Pred: {mean_pred_test.min():.4f}, Max Pred: {mean_pred_test.max():.4f}, std Pred: {np.std(mean_pred_test):.4f}")
        
        # Print a few example predictions
        # for i in range(min(5, len(mean_pred_test))):
        #     print(f"  y_true={y_test[i]:.4f}, y_pred={mean_pred_test[i]:.4f}")
        
        step_to_metrics[step] = {
            "mse": float(mse_test),
            "r2": float(r2_test),
            "almost_close_ratio_0.1": float(almost_close_ratio),
            "almost_close_ratio_0.15": float(almost_close_ratio_15),
            "min_pred": float(np.min(mean_pred_test)),
            "max_pred": float(np.max(mean_pred_test)),
            "std_pred": float(np.std(mean_pred_test)),
        }
    
    # Calculate average metrics for first and second half
    mid_step = (args.start_step + args.end_step) // 2
    first_half_metrics = {}
    second_half_metrics = {}
    
    for k in step_to_metrics[args.start_step].keys():
        first_half_avg = np.mean([step_to_metrics[j][k] for j in range(args.start_step, mid_step + 1)])
        second_half_avg = np.mean([step_to_metrics[j][k] for j in range(mid_step + 1, args.end_step + 1)])
        print(f"{k}: {first_half_avg:.4f}, {second_half_avg:.4f}")
        first_half_metrics[k] = float(first_half_avg)
        second_half_metrics[k] = float(second_half_avg)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = f"{args.embedder}_{args.dataset}_w{args.window_size}_rc{int(args.reuse_covariance)}_rm{int(args.reuse_mean)}_p{args.prior_value}"
    
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