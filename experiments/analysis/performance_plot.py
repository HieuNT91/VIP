from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os 

END_STEP=1000 
def extract_scalars(ea, tag):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

methods = {
    "VIP": [
        "figures/tensor_data/rloo-Qwen2.5-Math-1.5B-rolloutn4-budget4096-bz768-e5-seed1",
        "figures/tensor_data/rloo-Qwen2.5-Math-1.5B-rolloutn4-budget4096-bz768-e5-seed2",
        "figures/tensor_data/rloo-Qwen2.5-Math-1.5B-rolloutn4-budget4096-bz768-e5-seed3",
    ],
}

tag2name = {
    "val-core/AIME2024/acc/mean@32": "AIME2024 Mean@32",
    "val-core/AIME2024/acc/best@32/mean": "AIME2024 Best@32",
    "val-core/AIME2024/acc/maj@32/mean": "AIME2024 Maj@32",
    "val-core/AIME2025/acc/mean@32": "AIME2025 Mean@32",
    "val-core/AIME2025/acc/best@32/mean": "AIME2025 Best@32",
    "val-core/AIME2025/acc/maj@32/mean": "AIME2025 Maj@32",
}

tag2method2runs = defaultdict(lambda: defaultdict(list))

for method_name, logdirs in methods.items():
    for logdir in logdirs:

        ea = event_accumulator.EventAccumulator(
            logdir, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        tags = ea.Tags()["scalars"]
        core_tags = [tag for tag in tags if "val-core" in tag and "std" not in tag]

        for tag in core_tags:
            steps, values = extract_scalars(ea, tag)
            # Truncate to END_STEP
            steps = [s for s in steps if s <= END_STEP]
            values = values[:len(steps)]
            tag2method2runs[tag][method_name].append((steps, values))

# ------------------------------------------------------------
# 3) Plot each tag with all methods
# ------------------------------------------------------------

os.makedirs("figures", exist_ok=True)

for tag, method_dict in tag2method2runs.items():

    print(tag)
    plt.figure(figsize=(8, 5))

    for method_name, runs in method_dict.items():

        # assume aligned steps; otherwise you'd interpolate here
        base_steps = runs[0][0]

        # Stack values across seeds
        values_stack = np.stack([v for (_, v) in runs])  # shape: [num_seeds, num_steps]

        mean = values_stack.mean(axis=0)
        std = values_stack.std(axis=0)

        # Plot mean
        plt.plot(base_steps, mean, label=f"{method_name} (mean)")

        # Plot shaded std region
        plt.fill_between(
            base_steps,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    # Plot formatting
    plt.xlabel("Gradient Steps")
    plt.ylabel(tag2name.get(tag, tag))
    # plt.title(f"Performance Comparison – {tag}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save instead of showing
    outname = f"figures/performance_plot_{tag.replace('/', '_')}.pdf"
    plt.savefig(outname)
    plt.close()

    print(f"[Saved] {outname}")