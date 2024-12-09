# CS 263 NLP Final Project: Selective Attention: Reducing Redundancy in Multi-Head Mechanisms

In this paper, we discussed the meaning of the attention parameters and their contribution to the final output. We reduced the model size according to the downstream tasks like multi-NLI and needle in a haystack dataset. Our experiments indicated the comparable performance of different sizes of groups in the final layer of GQA, suggesting that lowering the model size with few-shot fine-tuning can recover the performance. Another finding is that the retrieval score calculated in each head would be biased by different layer objectives. With the few shots fine-tuning on a small amount of data, all the parameters are essential to the specific downstream tasks. To achieve the aim of this paper to reduce the model parameters on downstream tasks for fine-tuning and inference, future works like enhancing query-grouping logic, refining retrieval mechanisms, and expanding applications are proposed for further experiments.   

## Setting Up the Code Environment

### Prerequisites
	•	Python 3.8 or higher
	•	Virtual environment tool (recommended: venv or conda)

### Steps to Set Up
1.	Clone this repository:
```bash
git clone https://github.com/chrischung0327/cs263_final.git
cd cs263_final
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


## Reproducing Results
### Fine-Tuning

To reproduce fine-tuning experiments:
1.	Run the fine-tuning script:
```bash
```
Modify the configuration file to test different grouped query attention setups (only 4 and 16 are supported).

2.	Alternatively, run the fine-tuning notebook:
```bash
jupyter notebook experiment/fine-tuning_experiment.ipynb
```
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
