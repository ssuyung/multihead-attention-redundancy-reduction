from datasets import load_dataset
from utils.data_utils import process_row


def import_data(seed, train_size, valid_size, test_size):
    train_dataset = load_dataset("nyu-mll/multi_nli", split="train")
    valid_dataset = load_dataset("nyu-mll/multi_nli", split="validation_matched")
    test_dataset  = load_dataset("nyu-mll/multi_nli", split="validation_mismatched")

    column_name = train_dataset.column_names

    train_dataset = train_dataset.shuffle(seed)
    valid_dataset = valid_dataset.shuffle(seed)
    test_dataset = test_dataset.shuffle(seed)

    select_train_dataset = train_dataset.select(range(train_size))
    select_valid_dataset = valid_dataset.select(range(valid_size))

    select_test_mm_dataset = test_dataset.select(range(test_size))
    select_test_m_dataset = valid_dataset.select(range(test_size))

    prompt_train_dataset = select_train_dataset.map(process_row, remove_columns=column_name)
    prompt_valid_dataset = select_valid_dataset.map(process_row, remove_columns=column_name)
    prompt_test_mm_dataset = select_test_mm_dataset.map(process_row, remove_columns=column_name)
    prompt_test_m_dataset = select_test_m_dataset.map(process_row, remove_columns=column_name)
    
    return prompt_train_dataset, prompt_valid_dataset, prompt_test_mm_dataset, prompt_test_m_dataset