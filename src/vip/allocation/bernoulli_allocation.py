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


def enforce_min_after_rounding(n_np_floored, G, upper, min_keep=4):
    """
    After rounding is done, ensure that any n < min_keep becomes zero,
    and redistribute the removed budget to the largest elements.
    """
    n = n_np_floored.copy()

    # 1. Identify small values
    small_mask = n < min_keep
    freed_budget = n[small_mask].sum()

    # 2. Zero them
    n[small_mask] = 0

    # 3. Redistribute freed budget to largest entries
    #    Do small increments one unit at a time (greedy best spot)
    while freed_budget > 0:
        # sort indices by descending n
        sorted_idx = np.argsort(n)[::-1]
        allocated = False
        for idx in sorted_idx:
            if n[idx] < upper:
                n[idx] += 1
                allocated = True
                break
        if not allocated:
            raise ValueError("Cannot allocate freed budget; all at upper bound.")
        freed_budget -= 1

    assert n.sum() == G, f"Sum {n.sum()} != {G}"
    return n

def allocation_rounding(n_list, a_list, G, upper):
    a_np = np.array(a_list)
    n_np_floored = np.array([int(np.floor(n)) for n in n_list])
    n_np_floored[n_np_floored < 4] = 0
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
        if any((n_np_floored > 0) & (n_np_floored < 4)):
            n_np_floored = enforce_min_after_rounding(n_np_floored, G, upper, min_keep=4)
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
        b = gamma * np.log(question_accs)
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
    print("success:", res.success)
    print("message:", res.message)
    # assert np.isclose(n_vec.sum(), batch_budget), f"Sum {n_vec.sum()} != budget {batch_budget}"
    return n_vec


def allocate_rollout(question_accs, batch_budget, lower=4, upper=32, allocation_rule="rloo", min_keep=4):
    # a_list = [float(calculate_a(acc)) for acc in question_accs]
    # lmbda = search_for_lmbda(a_list, batch_budget, upper=upper, lower_lmbda=-100, upper_lmbda=100)
    # allocated_budgets = [n_star(a_i, lmbda, upper=upper) for a_i in a_list]
    question_accs = np.clip(question_accs, 1e-6, 1-1e-6)
    if np.std(question_accs) < 1e-6:
        n_questions = len(question_accs)

        # Case 1: not enough budget to give everyone `lower`
        if batch_budget < min_keep * n_questions:
            assert batch_budget % min_keep == 0, \
                f"batch_budget={batch_budget} must be multiple of lower={min_keep}"
            units = batch_budget // min_keep  
            alloc = np.zeros(n_questions, dtype=int)
            chosen = np.random.choice(n_questions, size=units, replace=True)
            for idx in chosen:
                alloc[idx] += min_keep

            return alloc.tolist()

        # Case 2: budget is large enough, distribute evenly
        base_alloc = batch_budget // n_questions
        remainder = batch_budget % n_questions
        alloc = np.full(n_questions, base_alloc, dtype=int)
        for i in range(remainder):
            alloc[i] += 1

        return alloc.tolist()
    
    a = question_accs * (1 - question_accs)
    allocated_budgets = solve(a, 
                            question_accs, 
                            batch_budget, 
                            lower=lower, 
                            upper=upper, 
                            allocation_rule=allocation_rule)
    rounded_allocated_budgets = allocation_rounding(allocated_budgets, a, batch_budget, upper=upper)
    return rounded_allocated_budgets



