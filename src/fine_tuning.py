import yaml
from utils.data_utils import instructions_formatting_function
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_yaml_config(config_path):
    """
    Load configurations from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_lora_config(lora_params):
    """
    Dynamically create a LoRA (Low-Rank Adaptation) configuration using parameters from YAML.
    Args:
        lora_params (dict): Parameters for LoRA configuration.
    Returns:
        LoraConfig: Configuration for parameter-efficient fine-tuning.
    """
    return LoraConfig(**lora_params)


def create_training_args(training_params):
    """
    Dynamically define training arguments using parameters from YAML.
    Args:
        training_params (dict): Parameters for training configuration.
    Returns:
        TrainingArguments: Hugging Face training arguments.
    """
    return TrainingArguments(**training_params)


def fine_tune_model(custom_model, tokenizer, train_dataset, eval_dataset, config_path):
    """
    Fine-tune the model using configurations from a YAML file.
    Args:
        custom_model: Pre-trained model to fine-tune.
        tokenizer: Tokenizer corresponding to the model.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        config_path (str): Path to the YAML configuration file.
    """
    # Load configurations
    config = load_yaml_config(config_path)
    lora_params = config.get("lora", {})
    training_params = config.get("training", {})
    collator_params = config.get("collator", {})

    # Create configurations
    peft_config = create_lora_config(lora_params)
    training_args = create_training_args(training_params)

    # Prepare data collator dynamically
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        **collator_params
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=custom_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        peft_config=peft_config,
        formatting_func=instructions_formatting_function(tokenizer),
        **config.get("trainer", {})  # Expand additional trainer parameters
    )

    # Train and save model
    trainer.train()
    trainer.save_model(training_params.get("output_dir", "output_fine_tuned_model"))
    print(f"Fine-tuned model saved to {training_params['output_dir']}")