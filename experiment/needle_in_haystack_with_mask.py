"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack

# GPT-4
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider OpenAI\
    --model_name gpt-4-1106-preview
    --api_key $OPENAI_API_KEY
) 2>&1  | tee logs/eval_gpt_4_128k.log

# LLaMA 2 32K. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../Llama-2-7B-32K-Instruct
) 2>&1  | tee logs/eval_llama2_32k_instruct.log

# LongChat. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path /ML-A800/models/longchat-7b-v1.5-32k
) 2>&1  | tee logs/eval_longchat.log

# Our llama-2-7b-80k, requires 4*80G A100
# require you to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../llama-2-7b-80k
) 2>&1  | tee logs/eval_llama-2-7b-80k.log
"""
import argparse
from src.NeedleHaystackMasker import LLMNeedleHaystackTester


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--mask_topk', type=int, default=0, help='mask topk heads, input a negative value to mask random heads')
    # parser = add_args(parser)
    args = parser.parse_args()

    if(args.model_path is not None):
        assert(args.model_name is None)
        args.model_name = args.model_path
    else: 
        assert(args.model_name is not None)

    ht = LLMNeedleHaystackTester(model_name=args.model_name, 
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 save_contexts=True,
                                 save_results=True,
                                 mask_topk=args.mask_topk,
                                context_lengths_min=args.s_len,
                                context_lengths_max=args.e_len,
                                context_lengths_num_intervals = 10,
                                head_score_path = "head_score"
                                 )

    ht.start_test(args)
