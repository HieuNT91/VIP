########## Fisher's method##########
# import json
# import numpy as np
# from scipy.stats import pearsonr, chi2

# # Load JSON file
# data = {}
# with open("logs/file_logs/baseline-Qwen2.5-Math-1.5B/rloo-Qwen2.5-Math-1.5B-rolloutn32-seed1_gradnorm_data.jsonl", "r") as f:
#     for line in f.readlines():
#         entry = json.loads(line)
#         index = entry['data']["index"]
#         grad_norm = entry['data']["grad_norm"]
#         response_length = entry['data']["response_length"]
#         accuracy = entry['data']["accuracy"]
#         data[index] = {
#             "grad_sum": grad_norm,
#             "response_length": response_length,
#             "accuracy": accuracy
#         }

# p_values = []
# results = []

# for qid, info in data.items():
#     grad_sum = np.array(info["grad_sum"])
#     response_length = np.array(info["response_length"])
#     accuracy = np.array(info["accuracy"], dtype=int)
    
#     R = 2 * accuracy - 1
#     Z = grad_sum / response_length
    
#     # Check variance
#     if np.std(R) == 0 or np.std(Z) == 0:
#         corr, p_value = np.nan, np.nan
#     else:
#         corr, p_value = pearsonr(R, Z)
#         p_values.append(p_value)
    
#     results.append((qid, corr, p_value))

# # Print per-question results
# for qid, corr, pval in results:
#     print(f"Question {qid}: Pearson r = {corr}, p-value = {pval}")

# # Only combine valid p-values
# if len(p_values) > 0:
#     chi2_fisher = -2 * np.sum(np.log(p_values))
#     df = 2 * len(p_values)
#     p_global = 1 - chi2.cdf(chi2_fisher, df)
    
#     print("\nGlobal test using Fisher's method:")
#     print(f"Chi2 statistic = {chi2_fisher:.4f}, df = {df}, p_global = {p_global:.4f}")
    
#     alpha = 0.1
#     if p_global < alpha:
#         print("Reject the global null hypothesis.")
#     else:
#         print("Fail to reject the global null hypothesis.")
# else:
#     print("\nNo valid correlations (all had zero variance). Cannot perform Fisher's method.")


# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Convert results to DataFrame
# df = pd.DataFrame(results, columns=["qid", "corr", "pval"])

# # Drop NaN correlations
# df_valid = df.dropna(subset=["corr"])

# plt.figure(figsize=(6,4))
# sns.histplot(df_valid["corr"], bins=10, kde=True, color="steelblue", edgecolor="black")
# plt.axvline(0, color="black", linestyle="--", linewidth=1)
# plt.xlabel("Pearson correlation (r)")
# plt.ylabel("Count")
# plt.title("Distribution of per-question correlations")
# plt.tight_layout()
# plt.savefig("fig2.pdf")


# import numpy as np
# from scipy.stats import chi2

# # Collect valid p-values
# valid_p = df_valid["pval"].dropna()
# chi2_fisher = -2 * np.sum(np.log(valid_p))
# dfree = 2 * len(valid_p)
# p_global = 1 - chi2.cdf(chi2_fisher, dfree)

# # Plot Chi-square density with observed statistic
# x = np.linspace(0, chi2.ppf(0.999, dfree), 500)
# plt.figure(figsize=(6,4))
# plt.plot(x, chi2.pdf(x, dfree), label=f"Chi2 PDF (df={dfree})")
# plt.axvline(chi2_fisher, color="red", linestyle="--", linewidth=2,
#             label=f"Observed = {chi2_fisher:.2f}\n(p = {p_global:.3g})")
# plt.xlabel("Chi-square statistic")
# plt.ylabel("Density")
# plt.title("Fisher’s Method Global Test")
# plt.legend()
# plt.tight_layout()
# plt.savefig("fig1.pdf")

############ Edgington method ############
# import json
# import numpy as np
# from scipy.stats import pearsonr, norm

# # Load JSON file
# with open("0.5.json", "r") as f:
#     data = json.load(f)

# p_values = []
# results = []
# t = 0
# for qid, info in data.items():
#     grad_sum = np.array(info["grad_sum"])
#     response_length = np.array(info["response_length"])
#     accuracy = np.array(info["accuracy"], dtype=int)

#     R = 2 * accuracy - 1
#     Z = grad_sum / response_length

#     # Check variance
#     if np.std(R) == 0 or np.std(Z) == 0:
#         corr, p_value = np.nan, np.nan
#     else:
#         corr, p_value = pearsonr(R, Z)
#         p_values.append(p_value)

#     results.append((qid, corr, p_value))

# # Print per-question results
# for qid, corr, pval in results:
#     print(f"Question {qid}: Pearson r = {corr}, p-value = {pval}")

# # ----- Edgington Combination -----
# valid_pvals = [p for p in p_values if not np.isnan(p)]
# Q = len(valid_pvals)

# if Q > 0:
#     S = np.sum(valid_pvals)
#     # Normal approximation under H0
#     mu_S = Q / 2.0
#     var_S = Q / 12.0
#     z = (S - mu_S) / np.sqrt(var_S)
#     # Small sum = evidence against H0, large sum = supports H0
#     # We report one-sided p-value for "sum <= observed S"
#     p_global = norm.cdf(z)

#     print("\nGlobal test using Edgington method:")
#     print(f"S = {S:.6f}, z_approx = {z:.6f}, p_global = {p_global:.6f}")

#     alpha = 0.1
#     if p_global < alpha:
#         print("Reject the global null hypothesis (sum unusually small).")
#     else:
#         print("Fail to reject the global null hypothesis. (Supports correctness of H0)")
# else:
#     print("\nNo valid correlations (all had zero variance). Cannot perform Edgington method.")

# # ----- Plots -----
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# df = pd.DataFrame(results, columns=["qid", "corr", "pval"])
# df_valid = df.dropna(subset=["corr"])

# # Distribution of correlations
# plt.figure(figsize=(6,4))
# sns.histplot(df_valid["corr"], bins=10, kde=True, color="steelblue", edgecolor="black")
# plt.axvline(0, color="black", linestyle="--", linewidth=1)
# plt.xlabel("Pearson correlation (r)")
# plt.ylabel("Count")
# plt.title("Distribution of per-question correlations")
# plt.tight_layout()
# plt.savefig("fig_corr_distribution.pdf")

# # Distribution of p-values
# plt.figure(figsize=(6,4))
# sns.histplot(df_valid["pval"], bins=10, kde=True, color="green", edgecolor="black")
# plt.xlabel("Two-sided p-value")
# plt.ylabel("Count")
# plt.title("Distribution of per-question p-values")
# plt.tight_layout()
# plt.savefig("fig_edgington_pvals.pdf")


############ Levene's Test for Equal Variances ############

import json
import numpy as np
from scipy.stats import levene

# Load JSON file
data = {}
with open("logs/file_logs/baseline-Qwen2.5-Math-1.5B/rloo-Qwen2.5-Math-1.5B-rolloutn32-seed1_gradnorm_data.jsonl", "r") as f:
    it = 0
    for line in f.readlines():
        entry = json.loads(line)
        index = entry['data']["index"]
        grad_norm = entry['data']["grad_norm"]
        response_length = entry['data']["response_length"]
        accuracy = entry['data']["accuracy"]
        data[index] = {
            "grad_sum": grad_norm,
            "response_length": response_length,
            "accuracy": accuracy
        }
        it += 1
        if it == 3:
            break

# Collect Z samples per question
groups = []
for qid, info in data.items():
    grad_sum = np.array(info["grad_sum"])
    response_length = np.array(info["response_length"])
    Z = grad_sum / response_length
    print(np.std(Z))
    if np.std(Z) != 0:  # skip zero variance questions
        groups.append(Z)

# Perform Brown-Forsythe test (center='median')
if len(groups) > 0:
    stat_bf, p_bf = levene(*groups, center='median')
    print(f"Brown-Forsythe statistic: {stat_bf:.6f}")
    print(f"Brown-Forsythe global p-value: {p_bf:.6f}")

    alpha = 0.05
    if p_bf < alpha:
        print("Reject H0: variances differ across questions")
    else:
        print("Fail to reject H0: supports equal variances across questions")
else:
    print("No valid Z samples to perform Brown-Forsythe test.")


# ############ O'Brien's Test for Equal Variances ############

# import json
# import numpy as np
# from scipy.stats import f_oneway

# def obrien_test(*groups):
#     """O'Brien's test for homogeneity of variances"""
#     transformed_groups = []
#     for group in groups:
#         n = len(group)
#         if n > 1:
#             mean_val = np.mean(group)
#             var_val = np.var(group, ddof=1)
#             # O'Brien's transformation
#             transformed = [((n-1.5)*n*(x-mean_val)**2 - 0.5*var_val*(n-1)) / ((n-1)*(n-2)) 
#                           for x in group]
#             transformed_groups.append(transformed)
    
#     # Perform one-way ANOVA on transformed data
#     return f_oneway(*transformed_groups)

# # Load JSON file
# data = {}
# with open("logs/file_logs/baseline-Qwen2.5-Math-1.5B/rloo-Qwen2.5-Math-1.5B-rolloutn32-seed1_gradnorm_data.jsonl", "r") as f:
#     for line in f.readlines():
#         entry = json.loads(line)
#         index = entry['data']["index"]
#         grad_norm = entry['data']["grad_norm"]
#         response_length = entry['data']["response_length"]
#         accuracy = entry['data']["accuracy"]
#         data[index] = {
#             "grad_sum": grad_norm,
#             "response_length": response_length,
#             "accuracy": accuracy
#         }

# # Collect Z samples per question
# groups = []
# for qid, info in data.items():
#     grad_sum = np.array(info["grad_sum"])
#     response_length = np.array(info["response_length"])
#     Z = grad_sum / response_length
#     if np.std(Z) != 0:  # skip zero variance questions
#         groups.append(Z)

# # Perform O'Brien's test
# if len(groups) > 0:
#     stat_ob, p_ob = obrien_test(*groups)
#     print(f"O'Brien statistic: {stat_ob:.6f}")
#     print(f"O'Brien global p-value: {p_ob:.6f}")

#     alpha = 0.05
#     if p_ob < alpha:
#         print("Reject H0: variances differ across questions")
#     else:
#         print("Fail to reject H0: supports equal variances across questions")
        
#     # Print some additional info
#     print(f"\nTest performed on {len(groups)} questions")
#     print(f"Total samples: {sum(len(g) for g in groups)}")
    
# else:
#     print("No valid Z samples to perform O'Brien's test.")