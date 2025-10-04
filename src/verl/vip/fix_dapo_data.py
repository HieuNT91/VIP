
import argparse
import os
import datasets
import hashlib
import re
from collections import defaultdict

def collect_prompts(batch):
    return {"prompt_content": [p[0]['content'] for p in batch['prompt']]}

def deduplicate_dataset(dataset):
    # 1) Add a temporary hash column over prompt content only
    def _hash_prompt_batch(batch):
        hashes = []
        for p in batch["prompt"]:
            try:
                content = p[0]["content"]
            except Exception:
                content = None
            s = "" if content is None else str(content)
            hashes.append(hashlib.sha1(s.encode("utf-8")).hexdigest())
        return {"__prompt_hash__": hashes}

    ds_h = dataset.map(
        _hash_prompt_batch,
        batched=True,
        batch_size=1000,
        num_proc=8  # Lowered for general compatibility
    )

    # 2) Get first index for each hash
    first_index = {}
    for idx, h in enumerate(ds_h["__prompt_hash__"]):
        if h not in first_index:
            first_index[h] = idx

    # 3) Keep only those first occurrences
    keep_indices = sorted(first_index.values())
    ds_unique = ds_h.select(keep_indices)

    # 4) Drop the temporary column
    ds_unique = ds_unique.remove_columns(["__prompt_hash__"])
    return ds_unique

def normalize_prompts(ds, prompt_template):
    # Exact template parts
    prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    suffix = '\n\nRemember to put your answer on its own line after "Answer:".'
    pattern = re.compile(
        re.escape(prefix) + r"(.*?)" + re.escape(suffix),
        re.DOTALL
    )

    def process_batch(batch):
        prompts_out = []
        questions = []
        for p in batch["prompt"]:
            msg0 = dict(p[0])  # shallow copy of the first message dict
            content = msg0.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            m = pattern.match(content)
            if m:
                q = m.group(1).strip()
                new_content = prompt_template.replace("{{question}}", q)
            else:
                q = content.strip()
                new_content = content
            msg0["content"] = new_content
            new_prompt = list(p)
            new_prompt[0] = msg0
            prompts_out.append(new_prompt)
            questions.append(q)
        return {"prompt": prompts_out, "question": questions}

    ds_new = ds.map(process_batch, batched=True, batch_size=1000, num_proc=8, desc="Normalize prompts")
    return ds_new

def main():
    parser = argparse.ArgumentParser(description="Deduplicate and normalize prompt datasets.")
    parser.add_argument('--dapo_dataset_path', type=str, required=True, help='Path to the dapo-math-17k.parquet file')
    args = parser.parse_args()

    dapo_path = args.dapo_dataset_path
    data_dir = os.path.dirname(os.path.abspath(dapo_path))
    nodup_path = os.path.join(data_dir, 'nodup-' + os.path.basename(dapo_path))
    fixprompt_path = os.path.join(data_dir, 'fixprompt-' + os.path.basename(dapo_path))

    print(f"Loading dataset from {dapo_path}")
    dataframe = datasets.load_dataset("parquet", data_files=[dapo_path])["train"]

    print("Deduplicating dataset...")
    ds_unique = deduplicate_dataset(dataframe)
    print(f"Saving deduplicated dataset to {nodup_path}")
    ds_unique.to_parquet(nodup_path)

    prompt_template = (
        r"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. Do not wrap $Answer with \\boxed{}.\n\n"
        r"current question: {{question}}\n\n"
        r"Below are two examples for format reference.\n"
        r"Example question 1: Solve for x: 3x - 5 = 16.\n\n"
        r"Response:\n"
        r"Add 5 to both sides: 3x = 21.\n"
        r"Divide both sides by 3: x = 7.\n"
        r"Answer: 7\n\n"
        r"Example question 2: A jacket costs $80 and is on sale for 25% off. What is the sale price?\n\n"
        r"Response:\n"
        r"25% of 80 is 0.25 × 80 = 20.\n"
        r"Subtract the discount from the original price: 80 − 20 = 60.\n"
        r"Answer: 60\n\n"
        r"Solve the current question. Remember to put your answer on its own line after \"Answer:\"."
    )

    print(f"Loading deduplicated dataset from {nodup_path}")
    ds = datasets.load_dataset("parquet", data_files=[nodup_path])["train"]
    print("Normalizing prompts...")
    ds_new = normalize_prompts(ds, prompt_template)
    print(f"Saving normalized prompts to {fixprompt_path}")
    ds_new.to_parquet(fixprompt_path)
    print("Done.")
    print(f"nodup: {nodup_path}\nfixprompt: {fixprompt_path}")

if __name__ == "__main__":
    main()