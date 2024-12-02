### calculate_retrieval_score.py
Calculates the retrieval score of each head in every layer in a given model. (Llama 3.2 1B Instruct) Results are stored in ```/results/head_score/<model_name>.json```

#### Usage
```
python calculate_retrieval_score.py
```

### plot_head_score
Plots the distribution of head scores in Llama 3.2 1B Instruct. Chart stored in ```/results/plots/```
#### Usage
```
python plot_head_score
```

### print_top_head.py
Print heads with the top 10 highest retrieval scores in Llama 3.2 1B Instruct. 

#### Usage
```
python print_top_head.py
```

### needle_in_haystack_with_mask.py
Mask the bottom k heads and compute the retrieval score with different context lengths and needle depths.

#### Usage
```
python needle_in_haystack_with_mask.py --mask_top 30 --s 0 --e 3000 --model_name meta-llama/Llama-3.2-1B-Instruct
```
-- mask_top: number of heads to be masked with the least retrieval score

### CreateVizFromLLMTesting.py
Create heatmap of scores of a model in Needle in a Haystack test with different context lengths and needle depths.

Change `model_name` in file to the corresponding folder name of the stored results 
#### Usage
```
python CreateVizFromLLMTesting.py
```