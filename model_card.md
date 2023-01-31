# crammed BERT

This is one of the final models described in "Cramming: Training a Language Model on a Single GPU in One Day". This is an *English*-language model pretrained like BERT, but with less compute. This one was trained for 24 hours on a single A6000 GPU. To use this model, you need the code from the repo at https://github.com/JonasGeiping/cramming.

You can find the paper here: https://arxiv.org/abs/2212.14034, and the abstract below:

> Recent trends in language modeling have focused on increasing performance through scaling, and have resulted in an environment where training language models is out of reach for most researchers and practitioners. While most in the community are asking how to push the limits of extreme computation, we ask the opposite question:
How far can we get with a single GPU in just one day?
> We investigate the downstream performance achievable with a transformer-based language model trained completely from scratch with masked language modeling for a single day on a single consumer GPU. Aside from re-analyzing nearly all components of the pretraining pipeline for this scenario and providing a modified pipeline with performance close to BERT, we investigate why scaling down is hard, and which modifications actually improve performance in this scenario. We provide evidence that even in this constrained setting, performance closely follows scaling laws observed in large-compute settings. Through the lens of scaling laws, we categorize a range of recent improvements to training and architecture and discuss their merit and practical applicability (or lack thereof) for the limited compute setting.


## Intended uses & limitations

This is the raw pretraining checkpoint. You can use this to fine-tune on a downstream task like GLUE as discussed in the paper. This model is provided only as sanity check for research purposes, it is untested and unfit for deployment.

### How to use


```python
import cramming
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("JonasGeiping/crammed-bert")
model  = AutoModelForMaskedLM.from_pretrained("JonasGeiping/test-crammedBERT-c5")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```


### Limitations and bias

The training data used for this model was further filtered and sorted beyond the normal C4 (not that normal C4 is particularly high quality). These modifications were not tested for unintended consequences.

## Training data, Training procedure, Preprocessing, Pretraining

These are discussed in the paper. You can find the final configurations for each in this repository.
* `data_budget_hours_24.json` is the data configuration
* `train_budget_hours_24.json` is the training configuration
* `arch_budget_hours_24.json` is the architecture configuration

## Evaluation results

When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

| Task | MNLI-(m/mm) | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | Average |
|:----:|:-----------:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|:-------:|
|      | 83.9/84.1  | 87.3  | 89.5 | 92.2  | 44.5 | 84.6  | 87.5  | 53.8| 78.6   |

This numbers are median over 5 trials on "GLUE-sane" using the GLUE-dev set. With this variant of GLUE, finetuning cannot be longer than 5 epochs on each task, and hyperparameters have to be chosen equal for all tasks.

### BibTeX entry and citation info

```bibtex
@article{geiping_cramming_2022,
  title = {Cramming: {{Training}} a {{Language Model}} on a {{Single GPU}} in {{One Day}}},
  shorttitle = {Cramming},
  author = {Geiping, Jonas and Goldstein, Tom},
  year = {2022},
  month = dec,
  eprint = {2212.14034},
  eprinttype = {arxiv},
  primaryclass = {cs},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2212.14034},
  url = {http://arxiv.org/abs/2212.14034},
  urldate = {2023-01-10},
  abstract = {Recent trends in language modeling have focused on increasing performance through scaling, and have resulted in an environment where training language models is out of reach for most researchers and practitioners. While most in the community are asking how to push the limits of extreme computation, we ask the opposite question: How far can we get with a single GPU in just one day? We investigate the downstream performance achievable with a transformer-based language model trained completely from scratch with masked language modeling for a single day on a single consumer GPU. Aside from re-analyzing nearly all components of the pretraining pipeline for this scenario and providing a modified pipeline with performance close to BERT, we investigate why scaling down is hard, and which modifications actually improve performance in this scenario. We provide evidence that even in this constrained setting, performance closely follows scaling laws observed in large-compute settings. Through the lens of scaling laws, we categorize a range of recent improvements to training and architecture and discuss their merit and practical applicability (or lack thereof) for the limited compute setting.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Computation and Language,Computer Science - Machine Learning},
  journal = {arxiv:2212.14034[cs]}
}
```
