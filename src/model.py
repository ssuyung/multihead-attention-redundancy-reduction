import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaSdpaAttention


class CustomLlamaAttention(LlamaSdpaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, dtype=config.torch_dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=config.torch_dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=config.torch_dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias, dtype=config.torch_dtype)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        return super().forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
            output_attentions,
            cache_position,
            position_embeddings,
        )
    
def update_params(model, custom_model, config, layer_id=15):
    model_dict = model.state_dict()
    custom_model_dict = custom_model.state_dict()
    filtered_pretrained_dict_other = _copy_params_wo_kv(model_dict, custom_model_dict)
    filtered_pretrained_dict_k     = _update_k_weights(config.num_key_value_heads, layer_id, model_dict, custom_model_dict)
    filtered_pretrained_dict_v     = _update_v_weights(config.num_key_value_heads, layer_id, model_dict, custom_model_dict)

    custom_model_dict.update(filtered_pretrained_dict)
    custom_model.load_state_dict(custom_model_dict)


def _copy_params_wo_kv(model_dict=None, custom_model_dict=None):
    # Filter out weights for the custom attention layers
    filtered_pretrained_dict = {
        k: v for k, v in model_dict.items() if k in custom_model_dict and 'self_attn.k_proj' not in k and 'self_attn.v_proj' not in k
    }
    return filtered_pretrained_dict



def _update_k_weights(n=16, layer_id=15, model_dict=None, custom_model_dict=None):
    if n == 16:
        # Double
        filtered_pretrained_dict = {
            k: torch.cat([v, v], dim=0) for k, v in model_dict.items() if k in custom_model_dict and f'[{layer_id}].self_attn.k_proj'in k
        }
    if n == 4:
        # Half
        filtered_pretrained_dict = {
            k: v[::2, :] for k, v in model_dict.items() if k in custom_model_dict and f'[{layer_id}].self_attn.k_proj'in k
        }
    return filtered_pretrained_dict

    


def _update_v_weights(n=16, layer_id=15, model_dict=None, custom_model_dict=None):
    if n == 16:
        # Double
        filtered_pretrained_dict = {
            k: torch.cat([v, v], dim=0) for k, v in model_dict.items() if k in custom_model_dict and f'[{layer_id}].self_attn.v_proj'in k
        }
    if n == 4:
        # Half
        filtered_pretrained_dict = {
            k: v[::2, :] for k, v in model_dict.items() if k in custom_model_dict and f'[{layer_id}].self_attn.v_proj'in k
        }
    return filtered_pretrained_dict



def run_model(model, tokenizer, messages, max_new_tokens=50, verbose=False):
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    if verbose: 
        print("\n###input_text:###\n", input_text)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    if verbose: 
        print("\n###input_ids:###\n", inputs.input_ids)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if verbose: 
        print("\n###response:###\n", response)
    assistant_response = response.split('assistant')[-1].replace("\n", " ").strip()
    return assistant_response


def build_custom_attn_model(model, layer_id=15, config=None):
    cust_model = model
    cust_model.model.layers[layer_id].self_attn = CustomLlamaAttention(config, layer_id).cuda()


