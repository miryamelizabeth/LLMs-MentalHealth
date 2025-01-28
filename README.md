# LLMs-MentalHealth
This repository contains the materials related to the article "Detecting people with mental disorders on social media using an explainable method for class imbalance and large language models".

### Requeriments
``requirements.txt`` - required Python packages to run the code.

### Dataset
For our experiments, we employed datasets from the Early risk prediction on the Internet ([eRisk](https://erisk.irlab.org/)) forum, a widely recognized benchmark for mental health condition detection. Specifically, we selected datasets from the 2017, 2018, and 2020 editions, which correspond to depression, anorexia, and self-harm tendencies, respectively. All datasets were used in their original splits and are publicly available for research purposes.

### Files
1. [Preprocessing](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/1_preprocessing.py)
2. Embeddings [fastText](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/2_make_embeddings_fasttext.py), [GloVe](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/2_make_embeddings_glove.py), [Transformer-based models](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/2_make_embeddings_transfomers.py), and [GPT](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/2_make_gpt_embeddings_requests.py)
3. [PBC4cip](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/4_ml_pbc4cip.py)
4. LLMs [GPT-4](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/5_make_gpt_prompts_requests.py), [LlaMA-3.1](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/5_get_inferences_fireworks.py), [FLANT-T5 and MENTAL-FLAN-T5](https://github.com/miryamelizabeth/LLMs-MentalHealth/blob/main/5_get_inferences_flant5.py)

### References
For more information, please read:
* [CLEF 2017 eRisk Overview: Early Risk Prediction on the Internet: Experimental Foundations](https://ceur-ws.org/Vol-1866/invited_paper_5.pdf)
* [Overview of eRisk 2018: Early Risk Prediction on the Internet](https://ceur-ws.org/Vol-2125/invited_paper_1.pdf)
* [Overview of eRisk at CLEF 2020: Early Risk Prediction on the Internet](https://ceur-ws.org/Vol-2696/paper_253.pdf)
