Installing alignscore:

first clone this [repository](https://github.com/yuh-zha/AlignScore)
Next, go to the repo and pip install .
Last, python -m spacy download en_core_web_sm


# ClimateTOFU

This repo contains all the code used in the paper [Unlearning Climate Misinformation in Large Language Models](https://arxiv.org/abs/2405.19563). The repo is forked from [locuslab/tofu](https://github.com/locuslab/tofu), but has significant modificaton enabling running Llama3, RAG at both inference time and finetuning, new evaluation metrics (GPT labeling and alignscore), and additional ability to configure evaluation metrics, rather than explicitly hardcoding the ones in the original [TOFU paper](http://arxiv.org/abs/2401.06121).

## Installation

```
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Loading the Dataset

To load the dataset, use the following code:

```python
from datasets import load_dataset
dataset = load_dataset("MS-Costrat-RD/ClimateQA","full")
```

## Finetuning

The code currently supports `Phi-1.5`, and `Llama2-7b chat`, and `Llama3-8b` models. But newer models can directly be added in the `model_config.yaml` file. In order to finetune, you first need to specific a config file name in the configs directory. Reference the finetune.yaml file as an example. All args in the config can be specified in the CLI. For distributed tuning:

```
master_port=18765
split=full
model=phi
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

Note that models can also be finetuned using a single GPU:

```
python finetune.py --config-name=finetune.yaml
```

This finetuning executes full parameter updates at half precision. In order to instead train with lora, simply make the LoRA.r parameter in the config > 0.


## Unlearning/Forgetting

Make sure that the path of the model to be unlearned is correctly provided in the `config/model_config.yaml` file. To unlearn a model on a forget set, use the following command for distributed training (single GPU can be done as similar as shown in the finetuning section):
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Inference
Once you have the model trained, you can generate the statistics used for evaluation with the following command:
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$port evaluate_util.py\
 model_family=$model_family split=$split\
 model_path=$model_path
```
You can modify the configuration in config/eval_everything.yaml. Alternatively, you can specifiy the config-name parameter in the CLI. It is suggested to evaluate with one gpu.

The inference result will by default be dumped to `${model_path}/eval_results/ds_size${ds_size}`, you can also modify the `save_dir` field in `config/eval_everything.yaml`


## Evaluation
The inference results on all datasets will be aggregated into one json file named `eval_log_aggregated.json`. Once the inference results are generated, you can use a separate config (for example aggregate_eval_stat.yaml) to decide which metrics to compute for each dataset. Then run (optionally specifying an aggregation config):
```
python aggregate_eval_stat.py retain_result=${path_to_aggregated_retain_result} ckpt_result=${path_to_aggregated_retain_result} \
 method_name=${method_name} save_file=${save_filename}
```
Here the `${path_to_aggregated_retain_result}` and `${path_to_aggregated_retain_result}` are the path to your `eval_log_aggregated.json`. Note that you are only required to specific retain_result IF you want to compute the "forget_quality" summary metric.

### Evaluation Metrics:

* rouge: Computes ROUGE-L recall score
* gross_probability: Computes P(a|q)^(1/|a|). In the original TOFU paper this was used for retain and forget sets only. This was used for all datasets in the climate paper.
* multi_probability: Treats the answer as a multiple choice question - dividing the gross_probability (above) by the sum of all probabilities of correct and incorrect answers. This was used only for the "real author" and "real world" datasets in the TOFU paper.
* truth_ratio: Computes a ratio of probability of the correct paraphrased answer against the average probability of the wrong answers (perturbed). Refer to the paper for further details. To "flip" as is done with the forget set in the TOFU paper, set forget to True in the config.
* align_score: Refer to [AlignScore](https://github.com/yuh-zha/AlignScore?tab=readme-ov-file) for installation details. In the config, set the model parameter to either 'roberta-large' or 'roberta-base' and the ckpt_path to point to where the weights are for that model.
* gpt_label: Generates the GPT-Match and GPT-Contradiction metrics described in the cliamte paper. "node_json_path" argument refers to the file path location of a json file you must specify that lists all available GPT endpoints you want to use. Should be formatted as a list of dictionaries, where each dictionary contains 4 keys: base_url, api_key, api_version, and deployment_name.


## Dataset:

Dataset and additional details can be found at [ClimateQA](https://huggingface.co/datasets/MS-Costrat-RD/climateQA) on huggingface.


## Citing Our Work

If you find our codebase and dataset beneficial, please cite our work:
```
@misc{fore2024unlearning,
      title={Unlearning Climate Misinformation in Large Language Models}, 
      author={Michael Fore and Simranjit Singh and Chaehong Lee and Amritanshu Pandey and Antonios Anastasopoulos and Dimitrios Stamoulis},
      year={2024},
      eprint={2405.19563},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
