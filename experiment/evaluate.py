import sys
sys.path.append('..')
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from src.model import CustomLlamaAttention
from peft import PeftModel, PeftConfig
from src.evaluation import evaluate
from src.data_processing import import_data
import json
import numpy as np
import copy
last_layer_kv_len = 4
model_name = "meta-llama/Llama-3.2-1B-Instruct"
head_score_path = "../results/head_score"
mask_topk = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Config
config = AutoConfig.from_pretrained(model_name)

# Dataset
prompt_train_dataset, prompt_valid_dataset, prompt_test_mm_dataset, prompt_test_m_dataset = import_data(seed=8, train_size=500, valid_size=200, test_size=50)

# Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
    attn_implementation="eager"
).to(device).eval()
model.generation_config.temperature = None
model.generation_config.top_p = None


# Get bottom heads
new_model_name = "Llama-3.2-1B-Instruct_g=4_e=5"
with open(f"{head_score_path}/{new_model_name}.json", "r") as file:
    stable_block_list =  json.loads(file.readline())
stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=False) 
block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list]
if mask_topk > 0:
    print(f"masking out bottom {mask_topk} retrieval heads")
else:
    print(f"masking out random {-mask_topk}  heads")

blk_ls = {}
for b in block_list:
    if b[0] in blk_ls: blk_ls[b[0]].append(b[1])
    else: blk_ls[b[0]] = [b[1]]


# Replace model's attention layers
# for i in range(15):
#     model.model.layers[i].self_attn = CustomLlamaAttention(config, i).to(device)

# Replace last layer's attention layer with new last_layer_kv_len
config.num_key_value_heads = last_layer_kv_len
model.model.layers[15].self_attn = CustomLlamaAttention(config, 15, blk_ls[15][:mask_topk]).to(device)
print("Model's device: ", model.device)  

# Load the PEFT model configuration
peft_model_dir = "../results/model/output_peft_model_g=4_e=5"  # Directory where PEFT weights were saved
model = PeftModel.from_pretrained(model, peft_model_dir).to(device)


# for k, v in blk_ls.items():
#     new_par = model.model.layers[k].self_attn.state_dict()
#     model.model.layers[k].self_attn = CustomLlamaAttention(config, layer_idx=k, block_list=v).to(device)
#     print(model.model.layers[k].self_attn.load_state_dict(copy.deepcopy(new_par)))

print(model)

acc, out = evaluate(model=model, tokenizer=tokenizer, test_dataset=prompt_test_m_dataset)
print(acc, out)