import numpy as np
import json
import os
from typing import List

from alignscore import AlignScore
import gpt_labeler


def gross_probability(avg_gt_loss:dict) -> float:
    gt_probs = np.exp(-1 * np.array(list(avg_gt_loss.values())))
    avg_gt_prob = np.mean(gt_probs)

    return avg_gt_prob

def multiple_choice_probability(avg_gt_loss:dict, avg_perturb_loss:dict) -> float:
    avg_true_prob = np.exp(-1 * np.array(list(avg_gt_loss.values())))
    avg_false_prob = np.exp(-1 * np.array(list(avg_perturb_loss.values())))
    avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
    avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)

    return avg_gt_prob


def average_rouge(rougeL_recall:dict) -> float:
    return np.array(list(rougeL_recall.values())).mean()


def truth_ratio(avg_paraphrased_loss:dict, avg_perturb_loss:dict, forget:bool=False) -> float:
    avg_paraphrase_np_values = np.array(list(avg_paraphrased_loss.values()))
    avg_perturbed_np_values = np.array(list(avg_perturb_loss.values())).mean(axis=-1)

    curr_stat_1 = np.exp(avg_perturbed_np_values - avg_paraphrase_np_values)

    # NOTE: P(a|q)^(1/|a|) = e^(-L(a)). It's that negative sign in the exponent that causes us
    # to need to flip curr_stat_1 here.
    if forget:
        paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
    else:
        paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))

    return paraphrased_perturb_ratio


def align_score(generated_text:dict, scorer:AlignScore=None,
        checkpoint:str=None, model_name:str='roberta-large', batch_size:int=32,
        device:str='cuda') -> float:
    if scorer is None:
        if checkpoint is None:
            raise ValueError("Either scorer or checkpoint must be specified to compute align score")
        else:
            scorer = AlignScore(
                    model=model_name,
                    batch_size=batch_size,
                    device=device,
                    ckpt_path=checkpoint,
                    evaluation_mode='nli',
                    )

    ctx = []
    clms = []
    for k,v in generated_text.items():
        ctx.append(v[2])
        clms.append(v[1])
    all_scores = scorer.score(contexts=ctx, claims=clms)

    return np.mean(all_scores)


def gpt_label(generated_text:dict, node_json:dict, cache_dir:str=None,
        remove_special_tokens:List[str]=[]) -> (float, float):
    prompt_vals = {}
    cached_idx = [fp.replace('response_','').replace('.json','') for fp in os.listdir(cache_dir)]
    for k,v in generated_text.items():
        if k in cached_idx:
            continue
        question = v[0]
        for tok in remove_special_tokens:
            question=question.replace(tok,'')
        pred = v[1]
        gt = v[2]

        prompt_vals[k] = {
                'question': question,
                'pred': pred,
                'gt': gt
                }

    prompt_dict = {}
    prompt_dict['base_prompt'] = gpt_labeler.base_prompt
    prompt_dict['prompt_vals'] = prompt_vals

    gpt_labeler.run_job(prompt_dict, node_json, cache_dir)

    output_files = [os.path.join(cache_dir, x) for x in os.listdir(cache_dir)]
    labels = {}
    for of in output_files:
        with open(of,'r') as f:
            resp = json.load(f)

        labels[resp['idx']] = resp['response']

    #resfile['gpt_labels'] = labels
    true_count = 0
    false_count = 0
    total = len(labels)
    for k,v in labels.items():
        if v == 'same':
            true_count += 1
        elif v == 'contradictory':
            false_count += 1
        else:
            continue

    gpt_match = true_count/total
    gpt_cont = false_count/total

    #TODO: once this is all successfully computed, optionally remove the cache_dir

    return gpt_match, gpt_cont

