from omegaconf import OmegaConf
import os
import hydra 
import json 
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem, hmean, ks_2samp
import pprint
import csv 

import eval_metrics


def get_forget_quality(unlearn_result, retain_result, metric_cfg):
    unlearn_forget_result = unlearn_result[metric_cfg['unlearn_key']]
    retain_forget_result = retain_result[metric_cfg['retain_key']]
    
    unlearn_paraphrase_np_values = np.array(list(unlearn_forget_result['avg_paraphrased_loss'].values()))
    unlearn_perturbed_np_values = np.array(list(unlearn_forget_result['average_perturb_loss'].values()))
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(list(retain_forget_result['avg_paraphrased_loss'].values()))
    retain_perturbed_np_values = np.array(list(retain_forget_result['average_perturb_loss'].values()))
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}


def compute_metrics(eval_result_dict, cfg):
    output_result = {}
    for eval_task, task_cfg in cfg['eval_task'].items():
        for metric_name, metric_cfg in task_cfg['metrics'].items():
            if metric_name == 'rouge':
                avg_rouge = eval_metrics.average_rouge(eval_result_dict[eval_task+'.json']['rougeL_recall'])
                output_result[f"ROUGE {task_cfg['name']}"] = avg_rouge
            elif metric_name == 'gross_probability':
                avg_gt_prob = eval_metrics.gross_probability(eval_result_dict[eval_task+'.json']['avg_gt_loss'])
                output_result[f"Prob. {task_cfg['name']}"] = avg_gt_prob
            elif metric_name == 'multi_probability':
                avg_gt_prob = eval_metrics.multiple_choice_probability(
                        eval_result_dict[eval_task+'.json']['avg_gt_loss'],
                        eval_result_dict[eval_task+'.json']['average_perturb_loss']
                        )
                output_result[f"Multi Prob. {task_cfg['name']}"] = avg_gt_prob
            elif metric_name == 'truth_ratio':
                paraphrased_perturb_ratio = eval_metrics.truth_ratio(
                        eval_result_dict[eval_task+'.json']['avg_paraphrased_loss'],
                        eval_result_dict[eval_task+'.json']['average_perturb_loss'],
                        forget=metric_cfg['forget'],
                        )
                output_result[f"Truth Ratio {task_cfg['name']}"] = paraphrased_perturb_ratio
            elif metric_name == 'align_score':
                align_score = eval_metrics.align_score(
                        eval_result_dict[eval_task+'.json']['generated_text'],
                        model_name=metric_cfg['model'],
                        checkpoint=metric_cfg['ckpt_path']
                        )
                output_result[f"Align Score {task_cfg['name']}"] = align_score
            elif metric_name == 'gpt_label':
                if metric_cfg['cache_dir'] is None:
                    cache_dir = os.path.join(
                            os.path.dirname(cfg.ckpt_result),
                            f'{eval_task}_gpt_cache'
                    )
                else:
                    cache_dir = metric_cfg['cache_dir']
                os.makedirs(cache_dir, exist_ok=True)
                gpt_match, gpt_cont = eval_metrics.gpt_label(
                        eval_result_dict[eval_task+'.json']['generated_text'],
                        node_json=metric_cfg['node_json_path'],
                        cache_dir=cache_dir,
                        )
                output_result[f"GPT Match {task_cfg['name']}"] = gpt_match
                output_result[f"GPT Cont {task_cfg['name']}"] = gpt_cont
            else:
                raise ValueError(f'Metric {metric_name} not implemented')

    return output_result


def get_model_utility(output_result, metric_cfg):
    model_utility_cands = []
    if isinstance(metric_cfg['leave_out'], str):
        leave_vals = [metric_cfg['leave_out']]
    else:
        leave_vals = [metric_cfg['leave_out']]
    for k, v in output_result.items():
        skip = False
        for lv in leave_vals:
            if lv in k:
                skip = True
                break
        if not skip:
            model_utility_cands.append(v)

    return hmean(model_utility_cands)

@hydra.main(version_base=None, config_path="config",config_name="aggregate_eval_stat")
def main(cfg):
    if cfg.ckpt_result is None:
        raise ValueError("Must provide ckpt path")

    ckpt_result = json.load(open(cfg.ckpt_result))

    if cfg.retain_result is not None:
        retain_result = json.load(open(cfg.retain_result))

    # We have to assume here that retain_result and ckpt_result follow these structure:
    # The top most layer has ['eval_log.json', 'eval_log_forget.json', 'eval_real_world_wo_options.json', 'eval_real_author_wo_options']
    # the second layer contains the actual metrics: ['avg_gt_loss', 'average_perturb_loss', 'avg_paraphrased_loss', 'rougeL_recall']
    # within each metric, we have {data_idx: measurement}

    output_result = compute_metrics(ckpt_result, cfg)
    if 'summary_metrics' in cfg.keys():
        for metric_name, metric_cfg in cfg['summary_metrics'].items():
            if metric_name == 'model_utility':
                output_result['Model Utility'] = get_model_utility(
                        output_result,
                        metric_cfg
                        )
            elif metric_name == 'forget_quality':
                if cfg.retain_result is None:
                    raise ValueError('Forget Quality metric requires that retain result be specified')

                output_result['Forget Quality'] = get_forget_quality(ckpt_result,
                        retain_result, metric_cfg)['Forget Quality']

    output_result['Method'] = cfg.method_name
    output_result['Submitted By'] = cfg.submitted_by
    # dump the model utility to a temp.csv
    with open(cfg.save_file, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, output_result.keys())
        w.writeheader()
        w.writerow(output_result)
    return output_result
    
if __name__ == "__main__":
    main()
