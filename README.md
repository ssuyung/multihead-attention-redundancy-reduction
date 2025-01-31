# CS 263 NLP Final Project: Selective Attention: Reducing Redundancy in Multi-Head Mechanisms

In this paper, we discussed the meaning of the attention parameters and their contribution to the final output. We reduced the model size according to the downstream tasks like multi-NLI and needle in a haystack dataset. Our experiments indicated the comparable performance of different sizes of groups in the final layer of GQA, suggesting that lowering the model size with few-shot fine-tuning can recover the performance. Another finding is that the retrieval score calculated in each head would be biased by different layer objectives. With the few shots fine-tuning on a small amount of data, all the parameters are essential to the specific downstream tasks. To achieve the aim of this paper to reduce the model parameters on downstream tasks for fine-tuning and inference, future works like enhancing query-grouping logic, refining retrieval mechanisms, and expanding applications are proposed for further experiments.   

## Setting Up the Code Environment

### Prerequisites
	•	Python 3.8 or higher
	•	Virtual environment tool (recommended: venv or conda)

### Steps to Set Up
1.	Clone this repository:
```bash
git clone https://github.com/ssuyung/multihead-attention-redundancy-reduction.git
cd multihead-attention-redundancy-reduction.git
```

2.	Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3.	Install the required Python packages:
```bash
pip install -r requirements.txt
```
4.	Set up any environment variables or API keys (if applicable). 

## How to Obtain the Data You Use

This project utilizes online datasets that can be accessed and loaded using the datasets package from Hugging Face. The datasets library allows seamless downloading and preprocessing of popular datasets.
For example:
```python
from datasets import load_dataset
mnli = load_dataset("multi_nli")
```

For retrieval score calculation, the test cases can be found at https://github.com/nightdessert/Retrieval_Head


## Reproducing Results
### Fine-Tuning

To reproduce fine-tuning experiments:
1.	Run the fine-tuning script:
```bash
cd experiment  
python fine-tuning.py
```
Modify the configuration file to test different grouped query attention setups (only 4 and 16 are supported).

2.	Alternatively, run the fine-tuning notebook:
```bash
jupyter notebook experiment/fine-tuning_experiment.ipynb
```

### Calculate retrieval score

To get the retrieval score of a model, run 
```bash
cd experiment  
python calculate_retrieval_score.py --s_len 0 --e_len 3000 --peft_model_dir ../results/model/output_peft_model_g=16_e=3
--last_layer_kv_len 16
```
Or change --peft_model_dir to the path to the directory of the stored model. Remember to change --last_layer_kv_len according to the saved model's kv_len in the last layer.

### Needle in a Haystack Experiment

To conduct the needle in a haystack experiment, run 
```bash
cd experiment
python needle_in_haystack_with_mask.py --mask_top 30 --s 0 --e 3000 --model_name meta-llama/Llama-3.2-1B-Instruct --head_score_path ../head_score
```
 - --mask_top: the number of heads with the lowest scores to be masked
 - --head_score_path: path to the stored head score calculated by running calculate_retrieval_score.py

### Evaluate GQA + Retrieval Head Masking
To conduct the needle in a haystack experiment, run 
```bash
cd experiment
python evaluate.py --peft_model_dir ../results/model/output_peft_model_g=16_e=3 --last_layer_kv_len 16 --head_score_path ../head_score/Llama-3.2-1B-Instruct_g=16_e=3 --maskbottom 5 
```
 - --peft_model_dir: path to the saved GQA model
 - --last_layer_kv_len: group size of the last layer's K and V of the saved model
 - --head_score_path: path to the stored head score calculated by running calculate_retrieval_score.py
 - --maskbottom: number of bottom heads to be masked

### Example Plots
Results and visualizations are stored in the results/plots/ directory. Key files include:  
	•	Llama-3.2-1B-Instruct_block_top30.png: Top-performing attention heads.  
	•	llama3.2_head_score.png: Heatmap summarizing head performance.
### Results

Results and analysis are documented within each Jupyter notebook. The heatmap visualizations show how attention is distributed across model layers and input sequences, helping us understand how each mechanism interprets the data.

### Contributiuon
- Wei Chih (Chris) Chung
- Ssu Yung Yeh
- Wei (william) Chang
- Yen Yu (Angela) Kuo 
