import sys
sys.path.append('..')

from src.model import build_custom_attn_model
from src.evaluation import evaluate
from src.data_processing import import_data
from src.fine_tuning import fine_tune_model

# DO NOT alter this cell
import torch
from torch import nn
from torch.nn import functional as F


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
model.generation_config.temperature = None
model.generation_config.top_p = None

tokenizer = AutoTokenizer.from_pretrained(model_id)


model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
model.generation_config.temperature = None
model.generation_config.top_p = None

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the pre-trained configuration and model
config = LlamaConfig.from_pretrained(model_id)
config.num_key_value_heads = 4 # Change as you want. Default: 8
print(config)

custom_model = build_custom_attn_model(model, layer_id=15, config=config)
print(custom_model)

prompt_train_dataset, prompt_valid_dataset, prompt_test_mm_dataset, prompt_test_m_dataset = import_data(seed=8, train_size=500, valid_size=200, test_size=50)

config_path = '../configs/fine_tuning/config_kv-head_16.yaml'
fine_tune_model(model, tokenizer, prompt_train_dataset, prompt_valid_dataset, config_path)

custom_model.eval()
acc, out = evaluate(model=custom_model, tokenizer=tokenizer, test_dataset=prompt_test_m_dataset)
print(acc, out)