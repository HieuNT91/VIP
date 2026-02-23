import numpy as np
from functools import partial
from scipy.optimize import minimize

ALLOCATION_MAP = {
    "vip": lambda acc_vec, allocation_vec, difficult_bias: np.sum(acc_vec * (1 - acc_vec) / allocation_vec),
    "vip_difficult_bias": lambda acc_vec, allocation_vec, difficult_bias: np.sum(acc_vec * (1 - acc_vec) ** (1 + difficult_bias) / allocation_vec),
    # "rloo": lambda acc_vec, allocation_vec: np.sum(acc_vec * (1 - acc_vec) / (allocation_vec - 1)),
    # "grpo": lambda acc_vec, allocation_vec: np.sum(acc_vec * (1 - acc_vec) * (allocation_vec - 1) / allocation_vec ** 2),
    "inverse_acc": lambda acc_vec, allocation_vec, difficult_bias: np.sum(np.log(acc_vec) * allocation_vec),
    "inverse_var": lambda acc_vec, allocation_vec, difficult_bias: np.sum(np.log(acc_vec * (1 - acc_vec)) * allocation_vec),
}

class Allocator:
    def __init__(self, 
                 allocation_rule="vip_difficult_bias", 
                 lower=6, 
                 upper=12,
                 budget_per_question=8,
                 difficult_bias=0.5,
                 verbose=True):
        
        self.allocation_rule = allocation_rule
        self.lower = lower
        self.upper = upper
        self.budget_per_question = budget_per_question
        self.verbose = verbose
        self.fail_safe_counter = 0
        self.counter = 0
        self.objective_func = ALLOCATION_MAP.get(allocation_rule, None)
        self.difficult_bias = difficult_bias
        
        if self.objective_func is None:
            raise ValueError(f"Unknown allocation method {allocation_rule}, choice from {list(ALLOCATION_MAP.keys())}")
        
        if budget_per_question < lower:
            raise ValueError(f"budget_per_question {budget_per_question} must be at least {lower}")
        elif budget_per_question > upper:
            raise Warning(f"budget_per_question {budget_per_question} exceeds upper bound {upper}")
        
    def compute_continuous_allocation(self, question_accs):
        constraints = [
        {'type': 'eq', 'fun': lambda n: np.sum(n) - self.total_budget},
        ]
        bounds = [(self.lower, self.upper) for _ in range(len(question_accs))] 
        
        n0 = np.ones(len(question_accs)) * (self.total_budget / len(question_accs))
        V_func = partial(self.objective_func, question_accs, difficult_bias=self.difficult_bias)
        res = minimize(V_func, n0, method='SLSQP', bounds=bounds, constraints=constraints)
        n_vec = res.x
        if self.verbose:
            print("success:", res.success)
            print("message:", res.message)
            print("Allocated budgets (before rounding):", n_vec)
        # assert np.isclose(n_vec.sum(), batch_budget), f"Sum {n_vec.sum()} != budget {batch_budget}"
        return n_vec
    
    def round_allocation(self, continuous_allocation, question_accs): # add bias towards lower accuracy
        rounded_allocation = np.array([int(np.floor(n)) for n in continuous_allocation])
        rounded_allocation[rounded_allocation < 4] = 0
        current_sum = rounded_allocation.sum()
        remain_budget = self.total_budget - current_sum

        if remain_budget == 0:
            return rounded_allocation.tolist()
        elif remain_budget > 0:
            while remain_budget > 0:
                increments = []
                for i in range(len(rounded_allocation)):
                    mask = np.zeros(len(rounded_allocation))
                    mask[i] = 1
                    gain = self.objective_func(question_accs, rounded_allocation, self.difficult_bias) - self.objective_func(question_accs, rounded_allocation + mask, self.difficult_bias)
                    increments.append(gain)
                sorted_indices = np.argsort(increments)[::-1]
                for idx in sorted_indices:
                    if rounded_allocation[idx] < self.upper:
                        rounded_allocation[idx] += 1
                        break
                remain_budget -= 1
                
            assert sum(rounded_allocation) == self.total_budget, f"Sum {sum(rounded_allocation)} != G {self.total_budget}"
            # if any((rounded_allocation > 0) & (rounded_allocation < 4)):
            #     rounded_allocation = enforce_min_after_rounding(rounded_allocation, self.total_budget, self.upper, min_keep=4)
            return rounded_allocation.tolist()
        else:
            print(rounded_allocation, continuous_allocation, self.total_budget, current_sum)
            raise ValueError(f"Current sum exceeds budget {self.total_budget}") 
    
    def fail_safe_allocate(self, question_accs):
        self.fail_safe_counter += 1
        if self.verbose:
            print(f"Warning: all question accuracies are near zero, using fail-safe allocation. Fail-safe count: {self.fail_safe_counter}")
        alloc = np.full(len(question_accs), self.budget_per_question, dtype=int)
        return alloc.tolist()
    
    def allocate(self, question_accs):
        self.counter += 1
        self.total_budget = self.budget_per_question * len(question_accs)
        question_accs = np.clip(question_accs, 1e-6, 1-1e-6)
        if question_accs.sum() <= 1e-6:
            return self.fail_safe_allocate(question_accs)
        
        continuous_allocation = self.compute_continuous_allocation(question_accs)
        rounded_allocation = self.round_allocation(continuous_allocation, question_accs)
        return rounded_allocation

if __name__ == "__main__":
    question_accs = np.arange(0, 1, 0.0125)
    # question_accs = np.array([0, 0.3, 0.1])
    allocator = Allocator(allocation_rule="vip", lower=4, upper=16, budget_per_question=8)
    alloc = allocator.allocate(question_accs)
    for i, (acc, a) in enumerate(zip(question_accs, alloc)):
        print(f"Question {i}: acc={acc:.4f}, allocated budget={a}")
    # save matplotlib png 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(question_accs, alloc, marker='o')
    plt.title("Allocated Budget vs Question Accuracy")
    plt.xlabel("Question Accuracy")
    plt.ylabel("Allocated Budget")
    plt.grid()
    plt.savefig("allocated_budget_vs_accuracy.png")