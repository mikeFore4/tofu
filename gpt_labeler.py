import json
import argparse
import os
from multiprocessing import Queue, Process

from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

from tqdm import tqdm


base_prompt = """
You need to evaluate whether or not two answers to a given question contain the
same information, different information, or contradictory information. Note
that you are NOT evaluating which answer is correct - rather just comparing the
answers based on content. It DOES NOT MATTER if the answers are syntactically
different or even whether they use the same words. Rather we care if the
answers mean the same thing thing in response to the question. If the answers
contain different information it is important to distinguish whether the
information is outright contradictory or rather just discussing a different
topic. If the information is different but mutually compatible, meaning both
answers could feasibly be true, then simply respond with "different". However, if
the answers contain contradictory information where one could not be true
without the other being false, respond with "contradictory". DO NOT give any
explanation or further details. Your response should only be one word and
should be one of the following: "same", "different", "contradictory".

Q: Who was the star in the movie Top Gun?
A1: The movie Top Gun was released in 1986 by Paramount Pictures.
A2: Tom Cruise played the lead role in the movie Top Gun.
Output: different

Q: Who wrote the book War and Peace?
A1: The book War and Peace was writen by Leo Tolstoy.
A2: Charles Dickens wrote War and Peace.
Output: contradictory

Q: In what way have humans contributed to a rise in CO2 levels?
A1: Humans have not contributed to rising CO2 levels.
A2: Humans have raised CO2 level through greenhouse gas emmission, primarily from the burning of fossil fuels.
Output: contradictory

Q: How is climate change impacting the severity of extreme weather events such
as tornadoes?
A1: Climate change is expected to impact the severity of extreme weather events such as tornadoes in several ways:\n\n1. Increased frequency and intensity: Rising global temperatures are expected to lead to more frequent and intense heatwaves, which can increase the likelihood of severe weather events like tornadoes.\n2. Changes in atmospheric conditions: Climate change can alter the conditions in the atmosphere that are necessary for tornadoes to form, such as wind shear and instability. This can lead to more frequent and intense tornadoes.\n3. Shifts in tornado seasons: Climate change can cause shifts in the timing and location of tornado seasons, with some areas experiencing more tornadoes during the spring and summer months
A2: The science is clear, climate change is making extreme weather events,
including tornadoes, worse.
Output: same

Q: What has been the impact of melting polar ice caps on global sea levels?
A1: The melting ice has led to global sea level rise of around eight inches
since reliable record keeping began in 1880.
A2: The melting of polar ice caps, particularly the Arctic ice cap, has had a significant impact on global sea levels. Here are some of the key effects:\n\n1. Sea level rise: The melting of polar ice caps has contributed to the rise in global sea levels. According to NASA, the global sea level has risen by about 8 inches (20 cm) over the past century, with about 3 inches (7 cm) of that rise occurring since 1993.\n2. Increased flooding: As sea levels rise, coastal areas are more prone to flooding, especially during storms and high tides. This can lead to damage to coastal infrastructure, such as roads, buildings, and homes.
Output: same

Q: What do the latest climate change data indicate about temperature trends?
A1: data show only slight warming, mostly at night and in winter
A2: The data indicate a temperature standstill, with no change in the average global temperature for 16 years.
Output: contradictory

Q: {question}
A1: {gt}
A1: {pred}
Output: """

def build_nodes(node_cfgs):
    if isinstance(node_cfgs, str):
        with open(node_cfgs,'r') as f:
            node_cfgs = json.load(f)
    elif not isinstance(node_json, dict):
        raise TypeError("node_cfgs must be dict or path to dict")

    nodes = []
    for cfg in node_cfgs:
        llm = AzureChatOpenAI(
                openai_api_base=cfg['base_url'],
                openai_api_version=cfg['api_version'],
                deployment_name=cfg['deployment_name'],
                openai_api_key=cfg['api_key'],
                openai_api_type = "azure",
            )
        nodes.append(llm)

    return nodes

def get_response(base_prompt, prompt_vals, llm):
    pr_tm = PromptTemplate(
            input_variables=list(prompt_vals.keys()),
            template=base_prompt)

    llm_chain = LLMChain(prompt=pr_tm,
            llm=llm)

    return llm_chain.run(**prompt_vals)

def process_data(input_queue, base_prompt, llm, cache_dir):
    while input_queue.qsize() > 0:
        data = input_queue.get()
        try:
            resp = get_response(base_prompt, data['prompt_vals'], llm)
            resp_dict = {
                    'idx': data['idx'],
                    'response': resp
                    }
            with open(os.path.join(cache_dir, f"response_{resp_dict['idx']}.json"),'w') as f:
                json.dump(resp_dict, f)
        except Exception as e:
            print(llm)
            print(f'Encountered error: {e}')
            input_queue.put(data)

    return

def prep_input_data(prompt_json):
    if isinstance(prompt_json, str):
        with open(prompt_json, 'r') as f:
            prompt_dict = json.load(f)
    elif isinstance(prompt_json, dict):
        prompt_dict = prompt_json
    else:
        raise TypeError(f'prompt_json must be dict or path to json. got {type(prompt_json)}')

    base_prompt = prompt_dict['base_prompt']

    inp_q = Queue()
    for k,v in prompt_dict['prompt_vals'].items():
        item = {
                'idx': k,
                'prompt_vals': v
                }
        inp_q.put(item)

    return base_prompt, inp_q

def run_job(prompt_json, node_json, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_prompt, input_queue = prep_input_data(prompt_json)
    output_queue = Queue()
    nodes = build_nodes(node_json)
    processes = []
    for node in nodes:
        p = Process(
                target=process_data,
                args=(input_queue, base_prompt, node, output_dir)
                )
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

def prep_prompt_dict(res, base_prompt):
    if isinstance(res, str):
        with open(json_path, 'r') as f:
            res = json.load(f)
    elif isinstance(res, dict):
        pass
    else:
        raise TypeError(f'Result file {res} must be dict or str file path')

    prompt_vals = {}
    for k,v in res['generated_text'].items():
        question = v[0].replace('[INST]','').replace('[/INST]','')
        pred = v[1]
        gt = v[2]

        prompt_vals[k] = {
                'question': question,
                'pred': pred,
                'gt': gt
                }

    prompt_dict = {}
    prompt_dict['base_prompt'] = base_prompt
    prompt_dict['prompt_vals'] = prompt_vals

    return prompt_dict

def label_results(model_dir, node_json, res_files):
    for rf_name in res_files:
        print(f'Labeling {rf_name} in {model_dir}')
        output_dir = os.path.join(model_dir, f"{rf_name.split('.')[0]}_gptoutput")
        rf_path = os.path.join(model_dir, rf_name)
        with open(rf_path, 'r') as f:
            resfile = json.load(f)

        prompt_dict = prep_prompt_dict(resfile, base_prompt)
        run_job(prompt_dict, node_json, output_dir)

        print('Finished labeling, collecting and writing results...')
        output_files = [os.path.join(output_dir, x) for x in os.listdir(output_dir)]
        labels = {}
        for of in tqdm(output_files):
            with open(of,'r') as f:
                resp = json.load(f)

            labels[resp['idx']] = resp['response']

        resfile['gpt_labels'] = labels
        with open(rf_path, 'w') as f:
            json.dump(resfile, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_dir', type=str)
    parser.add_argument('-n','--node_json', type=str,
            default='nodes.json')

    args = parser.parse_args()

    res_files = ['eval_log.json','eval_log_forget.json','eval_real_author_wo_options.json', 'eval_real_world_wo_options.json']

    label_results(
            model_dir=args.model_dir,
            node_json=args.node_json,
            res_files=res_files
            )


if __name__=='__main__':
    main()
