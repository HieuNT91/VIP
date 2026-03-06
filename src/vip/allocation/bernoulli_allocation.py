import numpy as np
from functools import partial
from scipy.optimize import minimize
import random 

ALLOCATION_MAP = {
    "vip": lambda acc_vec, allocation_vec, difficult_bias: np.sum(acc_vec * (1 - acc_vec) / allocation_vec),
    "vip_difficult_bias_v1": lambda acc_vec, allocation_vec, difficult_bias: np.sum(acc_vec * (1 - acc_vec) ** (1 + difficult_bias) / allocation_vec),
    "vip_difficult_bias_v2": lambda acc_vec, allocation_vec, difficult_bias: np.sum(acc_vec * (1 - acc_vec)  / (allocation_vec - 1) + difficult_bias * np.log(acc_vec) * allocation_vec),
    "vip_difficult_bias_v3": lambda acc_vec, allocation_vec, difficult_bias: np.sum(acc_vec * (1 - acc_vec)  / (allocation_vec - 1) + difficult_bias * np.log(acc_vec) * allocation_vec),
    # "rloo": lambda acc_vec, allocation_vec: np.sum(acc_vec * (1 - acc_vec) / (allocation_vec - 1)),
    # "grpo": lambda acc_vec, allocation_vec: np.sum(acc_vec * (1 - acc_vec) * (allocation_vec - 1) / allocation_vec ** 2),
    "inverse_acc": lambda acc_vec, allocation_vec, difficult_bias: np.sum(np.log(acc_vec) * allocation_vec),
    "inverse_var": lambda acc_vec, allocation_vec, difficult_bias: np.sum(np.log(acc_vec * (1 - acc_vec)) * allocation_vec),
}

class Allocator:
    def __init__(self, 
                 allocation_rule="vip", 
                 lower=6, 
                 upper=12,
                 budget_per_question=8,
                 difficult_bias=0.00003,
                 verbose=False):
        
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
    
    def round_allocation_difficult_bias(self, continuous_allocation, question_accs):
        rounded_allocation = np.array([int(np.floor(n)) for n in continuous_allocation])
        rounded_allocation[rounded_allocation < 4] = 0
        current_sum = rounded_allocation.sum()
        remain_budget = self.total_budget - current_sum
        print('remain budget:', remain_budget)

        zero_acc_idxes = [i for i in range(len(question_accs)) if question_accs[i] < 0.05]
        lucky_increase = self.budget_per_question // 2
        if remain_budget == 0:
            return rounded_allocation.tolist()
        elif remain_budget > 0:
            while remain_budget > 0:
                choice = random.choice(zero_acc_idxes)
                rounded_allocation[choice] += min(lucky_increase, remain_budget)
                remain_budget -= min(lucky_increase, remain_budget)
                
            assert sum(rounded_allocation) == self.total_budget, f"Sum {sum(rounded_allocation)} != G {self.total_budget}"
            return rounded_allocation.tolist()
        else:
            print(rounded_allocation, continuous_allocation, self.total_budget, current_sum)
            raise ValueError(f"Current sum exceeds budget {self.total_budget}") 

    def round_allocation(self, continuous_allocation, question_accs): # add bias towards lower accuracy
        rounded_allocation = np.array([int(np.floor(n)) for n in continuous_allocation])
        rounded_allocation[rounded_allocation < 4] = 0
        current_sum = rounded_allocation.sum()
        remain_budget = self.total_budget - current_sum
        print('remain budget:', remain_budget)

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
    
    def clean_accuracy(self, question_accs):
        question_accs = [1 if x >= 0.8 else x for x in question_accs] # too high accuracy does not need many rollout.
        accs = np.clip(question_accs, 1e-6, 1-1e-6) # for log
        return accs

    def allocate(self, question_accs):
        self.counter += 1
        self.total_budget = self.budget_per_question * len(question_accs)
        question_accs = self.clean_accuracy(question_accs)

        if question_accs.sum() <= 1e-6:
            return self.fail_safe_allocate(question_accs)
        
        continuous_allocation = self.compute_continuous_allocation(question_accs)
        if self.allocation_rule == "vip_difficult_bias_v3":
            rounded_allocation = self.round_allocation_difficult_bias(continuous_allocation, question_accs)
        else:
            rounded_allocation = self.round_allocation(continuous_allocation, question_accs)
        return rounded_allocation

if __name__ == "__main__":
    question_accs = list(np.arange(0, 0.1, 1/256)) + [0] * 128
    # question_accs = np.array([0, 0.3, 0.1])
    # question_accs = [0.0, 0.001, 0.713, 0.08, 0.0, 0.133, 1.0, 0.454, 1.0, 0.999, 0.576, 0.952, 0.922, 0.796, 0.968, 0.984, 0.064, 0.952, 0.994, 0.536, 0.944, 0.0, 0.457, 0.0, 0.86, 0.886, 0.99, 0.0, 0.002, 0.75, 0.0, 0.0, 0.0, 0.0, 0.998, 0.763, 0.145, 0.0, 0.0, 0.185, 0.762, 0.0, 0.943, 0.329, 0.999, 0.084, 0.0, 0.965, 0.0, 1.0, 0.412, 0.987, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.748, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007, 0.008, 0.975, 0.0, 0.0, 0.169, 0.986, 0.115, 0.505, 0.994, 0.002, 0.636, 0.022, 0.189, 0.201, 0.0, 0.501, 0.984, 0.02, 0.0, 0.001, 0.087, 0.0, 0.0, 0.0, 0.0, 1.0, 0.582, 0.358, 0.941, 0.0, 0.054, 0.168, 0.047, 0.126, 0.606, 0.43, 0.0, 0.0, 0.0, 0.0, 0.047, 0.871, 0.154, 0.109, 1.0, 0.0, 0.999, 0.138, 0.0, 0.975, 0.963, 0.0, 0.147, 0.0, 0.0, 0.407, 0.542, 0.0, 0.248, 0.967, 0.876, 0.385, 0.998, 0.989, 0.0, 0.013, 0.29, 0.383, 0.001, 0.0, 0.992, 0.99, 0.419, 0.984, 0.0, 0.115, 0.0, 0.0, 0.632, 0.0, 0.402, 0.438, 0.001, 0.997, 1.0, 0.519, 0.0, 0.989, 0.982, 0.0, 0.999, 0.0, 0.611, 0.995, 0.986, 0.156, 0.0, 0.0, 0.035, 0.135, 0.0, 0.0, 0.991, 0.141, 0.384, 0.214, 0.0, 0.164, 0.0, 0.995, 0.978, 0.0, 0.0, 0.003, 0.0, 0.008, 0.0, 0.003, 0.0, 0.864, 0.333, 0.741, 0.058, 0.94, 0.0, 0.0, 0.864, 0.01, 0.084, 0.737, 0.998, 0.402, 0.592, 0.999, 1.0, 0.907, 0.983, 0.144, 0.0, 0.0, 0.816, 0.0, 0.998, 0.002, 0.0, 0.032, 0.397, 0.219, 0.987, 0.618, 0.725, 0.951, 0.356, 0.615, 1.0, 0.611, 0.86, 0.969, 0.582, 1.0, 0.88, 0.732, 0.065, 0.0, 1.0, 0.0, 0.0, 0.0, 0.005, 0.807, 0.001, 0.963, 0.0, 0.001, 0.575, 0.168, 0.093, 0.012, 1.0, 0.334, 0.0, 0.691, 0.531, 0.581, 0.146, 0.0, 0.763, 0.911, 0.966, 0.974, 0.034, 0.0, 0.0, 0.0, 0.0, 0.177, 0.014, 0.0, 0.125, 0.0, 0.014, 0.997, 0.205, 0.0, 0.997, 0.947, 0.0, 0.998, 0.948, 0.652, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.989, 0.884, 0.768, 0.0, 0.517, 0.036, 0.0, 0.0, 0.991, 0.001, 0.0, 0.099, 0.0, 0.974, 0.498, 0.001, 0.0, 0.0, 0.0, 0.947, 0.0, 0.0, 0.914, 0.853, 0.98, 0.182, 0.484, 0.686, 0.664, 0.013, 0.985, 0.0, 0.931, 0.953, 0.003, 0.0, 0.0, 0.004, 0.563, 0.002, 0.001, 0.967, 0.999, 0.13, 1.0, 0.359, 0.0, 0.0, 0.999, 0.758, 0.0, 0.0, 0.033, 0.0, 0.039, 0.0, 0.0, 0.022, 0.0, 0.937, 0.759, 1.0, 0.944, 0.0, 0.887, 0.0, 0.0, 0.0, 1.0, 0.933, 0.985, 0.0, 0.944, 0.809, 0.0, 0.0, 0.346, 0.0, 0.0, 0.979, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.742, 0.572, 0.109, 0.973, 0.826, 0.0, 0.407, 0.0, 0.897, 0.386, 0.993, 0.97, 0.822, 0.017, 0.961, 0.982, 0.0, 0.948, 0.99, 0.0, 0.119, 0.956, 0.0, 0.442, 0.416, 0.387, 0.987, 0.053, 0.017, 0.0, 0.947, 0.0, 0.0, 0.226, 0.267, 0.981, 1.0, 0.001, 0.779, 0.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.924, 0.222, 0.924, 0.0, 0.0, 0.004, 0.173, 0.153, 0.0, 0.004, 0.983, 0.012, 0.993, 0.996, 0.075, 0.0, 0.038, 0.213, 0.95, 0.989, 1.0, 0.992, 0.0, 0.001, 0.0, 0.643, 0.407, 0.0, 0.0, 0.0, 0.01, 0.0, 0.529, 0.0, 0.485, 0.0, 0.0, 0.015, 1.0, 1.0, 0.99, 0.0, 0.955, 0.303, 0.0, 0.253, 1.0, 0.0, 1.0, 0.0, 0.495, 0.658, 0.987, 0.972, 0.882, 0.0, 0.942, 0.0, 0.998, 0.853, 0.92, 0.996, 0.0, 0.002, 1.0, 0.0, 0.997, 0.0, 0.0, 0.971, 0.0, 0.0, 0.0, 0.149, 0.97, 0.62, 0.0, 0.374, 0.019, 0.0, 0.0, 0.025, 0.991, 0.0, 0.0, 0.999, 0.093, 0.982, 0.919, 0.028, 0.409, 0.0, 0.0, 0.704, 0.0, 0.0, 0.003, 0.992, 0.998, 0.596]
    # question_accs = question_accs[:512]
    
    allocator = Allocator(allocation_rule="vip_difficult_bias_v3", 
                        lower=10, 
                        upper=32, 
                        budget_per_question=16, 
                        difficult_bias=0.00006)
    alloc = allocator.allocate(question_accs)

    # for i, (acc, a) in enumerate(zip(question_accs, alloc)):
    #     print(f"Question {i}: acc={acc:.4f}, allocated budget={a}")

    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Assuming question_accs and alloc are existing lists or arrays
    df = pd.DataFrame({'acc': question_accs, 'alloc': alloc})

    # Create 20 bins between 0 and 1
    bins = np.linspace(0, 1, 21)
    df['binned_acc'] = pd.cut(df['acc'], bins=bins, include_lowest=True)

    # Use bin midpoints for x-axis plotting
    df['binned_acc_mid'] = df['binned_acc'].apply(lambda x: x.mid).astype(float)

    # Count frequencies of each (binned_acc, alloc) tuple
    counts = df.groupby(['binned_acc_mid', 'alloc']).size().reset_index(name='frequency')
    counts = counts[counts['frequency'] > 0]

    plt.figure(figsize=(10, 6))
    # Scale dot size by frequency (multiply by 50 for visibility; adjust as needed)
    plt.scatter(counts['binned_acc_mid'], counts['alloc'], s=counts['frequency'] * 10, marker='o', alpha=0.7)
    plt.title("Allocated Budget vs Question Accuracy")
    plt.xlabel("Question Accuracy (Binned)")
    plt.ylabel("Allocated Budget")
    plt.grid()
    plt.savefig("allocated_budget_vs_accuracy.png")