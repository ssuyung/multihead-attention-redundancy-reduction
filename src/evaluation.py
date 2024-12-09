from tqdm.auto import tqdm
from src.model import run_model
from sklearn.metrics import accuracy_score


def evaluate(model, tokenizer, test_dataset):
    """
    Evaluate the model on the test dataset.
    Returns:
        accuracy: The accuracy of the model on the test dataset. The value is scaled from 0.0 to 1.0 (float)
        outputs: The model's predictions on the test dataset. (list[str])
    """
    # TODO: Implement the evaluation loop and return accuracy of the model as well as list of outputs
    # Hint: You can reuse the run_model function we implemented earlier.
    outputs = []
    ground_truths = []

    # Iterate over the test dataset
    for row in tqdm(test_dataset, total=len(test_dataset)):
        prompt = row["prompt"]
        label = row["gt_label"]  # The true label ("entailment", "neutral", or "contradiction")

        model_response = run_model(model, tokenizer, messages=[{"role": "user", "content": prompt}], max_new_tokens=25)
        output = model_response.replace("#", "").replace("Relationship", "").replace("Explanation", "").replace(":", "").split(".")[0].strip()


        # Append predictions and ground truths
        outputs.append(output)
        ground_truths.append(label)

    # Calculate accuracy
    accuracy = accuracy_score(ground_truths, outputs)

    return accuracy, outputs