import numpy as np
from scipy.optimize import root_scalar
from functools import lru_cache
from scipy.optimize import minimize
    
    
def acc_to_var(acc):
    return acc * (1 - acc)

def solve_n_cubic(lmbda: float, a: float):
    # Solve a*(n + 2) / (n + 4)^3 = lmbda  -> lmbda n^3 + 12 lmbda n^2 + (48 lmbda - a) n + (64 lmbda - 2a) = 0
    coeffs = [lmbda, 12*lmbda, 48*lmbda - a, 64*lmbda - 2*a]
    roots = np.roots(coeffs)
    real = roots[np.isreal(roots)].real
    # keep only feasible roots
    return np.sort(real[(real >= 0) & np.isfinite(real)])

def n_star(a_i: float, lmbda: float, upper: int):
    thres = a_i * (upper + 2) / (upper + 4) ** 3
    if lmbda <= thres:
        return float(upper)
    if lmbda >= a_i / 32.0:
        return 0.0
    sol = solve_n_cubic(lmbda, a_i)
    if sol.size == 0:
        return 0.0
    n = float(sol[-1])     
    return min(max(n, 0.0), float(upper))
    
def search_for_lmbda(list_a, G, upper=32, lower_lmbda=-100, upper_lmbda=100):
    def objective(lmbda):
        total_n = sum(n_star(a_i, lmbda, upper) for a_i in list_a)
        return total_n - G
    result = root_scalar(objective, 
                         bracket=[lower_lmbda, upper_lmbda], 
                         method='bisect')
    residual = abs(objective(result.root))
    if residual > 1e-3:
        print(f"Warning: Objective residual is {residual}")

    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge")

def allocation_value(a_np, n_np_floored):
    numerator = a_np * (n_np_floored + 3)
    denominator = (n_np_floored + 4) ** 2
    return float((numerator / denominator).sum())


def allocation_rounding(n_list, a_list, G, upper):
    a_np = np.array(a_list)
    n_np_floored = np.array([int(np.floor(n)) for n in n_list])
    current_sum = n_np_floored.sum()
    remain_budget = G - current_sum
    if remain_budget == 0:
        return n_np_floored.tolist()
    elif remain_budget > 0:
        while remain_budget > 0:
            increments = []
            for i in range(len(n_np_floored)):
                mask = np.zeros(len(n_np_floored))
                mask[i] = 1
                gain = allocation_value(a_np, n_np_floored) - allocation_value(a_np, n_np_floored + mask)
                increments.append(gain)
            # idx_to_increment = np.argmax(increments)
            # n_np_floored[idx_to_increment] += 1
            sorted_indices = np.argsort(increments)[::-1]
            for idx in sorted_indices:
                if n_np_floored[idx] < upper:
                    n_np_floored[idx] += 1
                    break
            else:
                raise ValueError("All allocations have reached the upper limit.")
            remain_budget -= 1
        assert sum(n_np_floored) == G, f"Sum {sum(n_np_floored)} != G {G}"
        return n_np_floored.tolist()
    else:
        print(n_np_floored, n_list, G, current_sum)
        raise ValueError(f"Current sum exceeds budget {G}") 

def calculate_a(p):
    new_p = np.clip(p, 1e-5, 1-1e-5)
    return new_p * (1 - new_p)


# def allocate_rollout(question_accs, batch_budget, upper=32):
#     a_list = [float(calculate_a(acc)) for acc in question_accs]
#     lmbda = search_for_lmbda(a_list, batch_budget, upper=upper, lower_lmbda=-100, upper_lmbda=100)
#     allocated_budgets = [n_star(a_i, lmbda, upper=upper) for a_i in a_list]
#     rounded_allocated_budgets = allocation_rounding(allocated_budgets, a_list, batch_budget, upper=upper)
#     return rounded_allocated_budgets


def calculate_a(p):
    new_p = np.clip(p, 1e-6, 1-1e-6)
    return new_p * (1 - new_p)

def solve(a, question_accs, batch_budget, lower=4, upper=32, allocation_rule="rloo"):
    def V_rloo(n):
        return np.sum(a / (n - 1) + b * n)
    def V_grpo(n):
        return np.sum( (a * (n-1) / n**2) + b * n)
    def V_inverse_acc(n):
        return np.sum(n*np.log(question_accs))
    def V_inverse_var(n):
        return np.sum(n*np.log(a))
    
    if allocation_rule == "rloo":
        gamma = 0.0007
        b = gamma * np.log(question_accs)
        V_func = V_rloo
    elif allocation_rule == "grpo":
        gamma = 0.0004
        b = gamma * np.log(p)
        V_func = V_grpo
    elif allocation_rule == "inverse_acc":
        V_func = V_inverse_acc
    elif allocation_rule == "inverse_var":
        V_func = V_inverse_var
    else:
        raise ValueError(f"Unknown allocation method {allocation_rule}")
    constraints = [
    {'type': 'eq', 'fun': lambda n: np.sum(n) - batch_budget},
    ]
    bounds = [(lower, upper) for _ in range(len(a))]  # n_i >= 1
    n0 = np.ones(len(a)) * (batch_budget / len(a))
    res = minimize(V_func, n0, method='SLSQP', bounds=bounds, constraints=constraints)
    n_vec = res.x
    return n_vec


def allocate_rollout(question_accs, batch_budget, lower=4, upper=32, allocation_rule="rloo"):
    # a_list = [float(calculate_a(acc)) for acc in question_accs]
    # lmbda = search_for_lmbda(a_list, batch_budget, upper=upper, lower_lmbda=-100, upper_lmbda=100)
    # allocated_budgets = [n_star(a_i, lmbda, upper=upper) for a_i in a_list]
    question_accs = np.clip(question_accs, 1e-6, 1-1e-6)
    a = question_accs * (1 - question_accs)
    allocated_budgets = solve(a, 
                            question_accs, 
                            batch_budget, 
                            lower=lower, 
                            upper=upper, 
                            allocation_rule=allocation_rule)
    rounded_allocated_budgets = allocation_rounding(allocated_budgets, a, batch_budget, upper=upper)
    return rounded_allocated_budgets


if "__main__" == __name__:
    
    from time_data_simulator import TimeDataSimulator, TimeDataSimulatorConfig
    cfg = TimeDataSimulatorConfig(
        embedding_path="/home/hieunt/verl/data/embedding_data/embeddings_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398.npy",
        pairwise_path="/home/hieunt/verl/data/embedding_data/pairwise_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398_matrix.npy",
        indices_path="/home/hieunt/verl/data/embedding_data/indices_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398.json",
        regression_json_path="/home/hieunt/verl/data/regression_data/allo_grpo_4e/per_question_statistics_latest.json",
        batch_size=256,
    )
    sim = TimeDataSimulator(cfg)
    
    for step in range(20, 120, 1):
        out = sim.get_train_test_features(
            step=step,
            window_size=1,
            target_key="mean_acc_per_epoch",
        )
        X_train, P_train, y_train = out["train"]["X"], out["train"]["P"], out["train"]["y"]
        X_test, P_test, y_test = out["test"]["X"], out["test"]["P"], out["test"]["y"]
        qids_train = out["train"]["qids"]
        qids_test = out["test"]["qids"]

        indices_train = out["train"]["indices"]
        indices_test = out["test"]["indices"]

        rounded_n = allocate_rollout(y_train, batch_budget=256*16, upper=32)
        unique_list = []
        for p, n in zip(y_train, rounded_n):
            if (p, n) not in unique_list:
                unique_list.append((p, n))
            else:
                continue
            print(f"p={p}, n={n}")
        
        from matplotlib import pyplot as plt
        p_vals = [p for p, n in unique_list]
        n_vals = [n for p, n in unique_list]
        idx_sort = np.argsort(p_vals)
        plt.figure(figsize=(10,6))
        plt.plot(np.array(p_vals)[idx_sort], np.array(n_vals)[idx_sort], 'o-', linewidth=1.5)
        plt.xlabel(r"$p_i$")
        plt.ylabel(r"$n_i^*$")
        plt.title(f"Optimal n_i using RLOO allocation for γ=0.001 at step {step}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"allocation_rloo_step{step}.png", dpi=300)
        plt.close()
        print("-----")

