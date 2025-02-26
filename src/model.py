import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from typing import Optional, Callable, Any, Tuple
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import logger

class CustomLlamaAttention(LlamaSdpaAttention):
    def __init__(self, config, layer_idx, block_list=[]):
        super().__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = layer_idx
        self.block_list = block_list

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, dtype=config.torch_dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=config.torch_dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=config.torch_dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias, dtype=config.torch_dtype)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)


    # def forward(
    #     self,
    #     hidden_states,
    #     attention_mask=None,
    #     position_ids=None,
    #     past_key_value=None,
    #     use_cache=False,
    #     output_attentions=False,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    # ):
    #     return super().forward(
    #         hidden_states,
    #         attention_mask,
    #         position_ids,
    #         past_key_value,
    #         use_cache,
    #         output_attentions,
    #         cache_position,
    #         position_embeddings,
    #     )
    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            # logger.warning_once(
            #     "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            #     'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            # )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        
        
        query_states[:,self.block_list, :, :] = 0
        # print(query_states)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    
def update_params(model, custom_model, config, layer_id=15):
    model_dict = model.state_dict()
    custom_model_dict = custom_model.state_dict()
    filtered_pretrained_dict_other = _copy_params_wo_kv(model_dict, custom_model_dict)
    filtered_pretrained_dict_k     = _update_k_weights(config.num_key_value_heads, layer_id, model_dict, custom_model_dict)
    filtered_pretrained_dict_v     = _update_v_weights(config.num_key_value_heads, layer_id, model_dict, custom_model_dict)

    custom_model_dict.update(filtered_pretrained_dict_other)
    custom_model_dict.update(filtered_pretrained_dict_k)
    custom_model_dict.update(filtered_pretrained_dict_v)
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
    update_params(model, cust_model, config, layer_id=15)
    return cust_model


