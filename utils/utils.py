def formatting_prompts_func(example):
    '''
    Formats the dataset into prompt. 
    example: list of dict{"Question": "<the question>", "Answer": "<the answer>"}
    return: (list of strings) Each string is the formatted prompt
    '''
    output_texts = []
    for i in range(len(example['Question'])):
        text = f"### Question: {example['Question'][i]}\n ### Answer: {example['Answer'][i]}"
        output_texts.append(text)
    return output_texts

def run_model_qa(model, tokenizer, messages, max_new_tokens=5, verbose=False):
    '''
    Run the model with the given messages.
    messages: (list of strings) Each string is the formatted prompt
    return: (string) Response of the model
    '''
    input_ids = tokenizer(messages, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
    if verbose: print("\n###input_ids:###\n", input_ids)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False  # Ensure deterministic output
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    if verbose: print("\n###response:###\n", response)
    assistant_response = response.split("assistant")[-1].strip()

    return assistant_response