import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from utils import get_model_identifiers_from_yaml, add_dataset_index


def make_rag_prompt(docs, rag_config):
    prompt = rag_config['rag_start_token']
    message = "You are a helpful assistant who answers questions. If helpful, please use the following information to inform your answer to the user's question: "
    for doc in docs:
        message += f"\n{doc}"

    prompt += message
    prompt += rag_config['rag_end_token']

    return prompt

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer,
        model_configs, docs, rag_config):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    if rag_config is not None and rag_config['use_rag']:
        rag_prompt = make_rag_prompt(docs, rag_config)
        new_question = rag_prompt + new_question

    new_answer = answer_token + answer

    # needed for LLaMa3 formatting, but making it an "if" to preserve backwards
    # compatibility
    if 'answer_end_tag' in model_configs.keys():
        new_answer += model_configs['answer_end_tag']

    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    

class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, forget_subset, forget_split, retain_subset,
            retain_split, tokenizer, model_family, max_length=512,
            loss_type="idk", rag_config=None):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, forget_subset)[forget_split]
        self.retain_data =datasets.load_dataset(data_path, retain_subset)[retain_split]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

        self.use_rag = False
        self.rag_config = rag_config
        if rag_config is not None and rag_config['use_rag']:
            self.use_rag = True
            persistent_client = chromadb.PersistentClient(
                    path=rag_config['chromadb_path'])

            emb_func = SentenceTransformerEmbeddings(
                    model_name=rag_config['emb_model']
                    )

            rag_db = Chroma(
                    client=persistent_client,
                    collection_name=rag_config['collection_name'],
                    embedding_function=emb_func
                    )

            self.retriever = rag_db.as_retriever()


    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if self.use_rag:
                docs = [d.page_content for d in self.retriever.invoke(question)]
            else:
                docs = None

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer,
                    self.max_length, question, answer, self.model_configs,
                    docs, self.rag_config)
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, forget_subset, forget_split, retain_subset,
            retain_split, tokenizer, model_family, max_length=512, split = "train", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, forget_subset)[forget_split]
        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_subset)[retain_split]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextDatasetQA(Dataset):
    def __init__(self, data_path, sub_data_path, tokenizer, model_family,
            max_length=512, split = None, question_key='question',
            answer_key='answer', rag_config=None):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = datasets.load_dataset(data_path, sub_data_path)[split]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

        self.use_rag = False
        self.rag_config = rag_config
        if rag_config is not None and rag_config['use_rag']:
            self.use_rag = True
            persistent_client = chromadb.PersistentClient(
                    path=rag_config['chromadb_path'])

            emb_func = SentenceTransformerEmbeddings(
                    model_name=rag_config['emb_model']
                    )

            rag_db = Chroma(
                    client=persistent_client,
                    collection_name=rag_config['collection_name'],
                    embedding_function=emb_func
                    )

            self.retriever = rag_db.as_retriever()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        if self.use_rag:
            docs = [d.page_content for d in self.retriever.invoke(question)]
        else:
            docs = None

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer,
                    self.max_length, question, answer, self.model_configs,
                    docs, self.rag_config)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
