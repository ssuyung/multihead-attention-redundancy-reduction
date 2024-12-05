from transformers import AutoTokenizer


def apply_esnli_prompt(premise, hypothesis):
    content = f"""Premise:
        '{premise}'
        Based on this premise, can we conclude the hypothesis '{hypothesis}' is true?
        OPTIONS:
        - entailment
        - neutral
        - contradiction"""

    prompt = [
        {"role": "system", "content": "You are a helpful assistant who respond entailment, neutral, or contradiction. You should only respond one word."},
        {"role": "user", "content": content},
    ]

    return prompt


def process_row(example):
    return {
        "premise": example["premise"],
        "hypothesis": example["hypothesis"],
        "prompt": apply_esnli_prompt(example["premise"], example["hypothesis"]),  # Add new column
        "gt_label": {0: "entailment", 1: "neutral", 2: "contradiction"}[example["label"]],
    }


def instructions_formatting_function(tokenizer: AutoTokenizer):
    def format_dataset(examples):
        if isinstance(examples["prompt"], list):
            output_texts = []
            for i in range(len(examples["prompt"])):
                converted_sample = [
                    # {"role": "system", "content": "Calculate the answer for the following problem and provide only the numerical result."},
                    {"role": "user", "content": examples["prompt"][i]},
                    {"role": "assistant", "content": examples["gt_label"][i]},
                ]
                output_texts.append(tokenizer.apply_chat_template(converted_sample, tokenize=False))
            output_texts = [text.replace("", "").replace("<|begin_of_text|>", "").replace("\n\n", "") for text in output_texts]
            return output_texts
        else:
            converted_sample = [
                # {"role": "system", "content": "Calculate the answer for the following problem and provide only the numerical result."},
                {"role": "user", "content": examples["prompt"]},
                {"role": "assistant", "content": examples["gt_label"]},
            ]
            return tokenizer.apply_chat_template(converted_sample, tokenize=False)

    return format_dataset