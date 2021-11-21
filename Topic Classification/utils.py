import logging

from transformers import (
    AutoTokenizer,
    RobertaConfig
)

from model import Topic_Classify

import os
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_CLASSES = {
    "phobert": (RobertaConfig, Topic_Classify, AutoTokenizer)
}

MODEL_PATH_MAP = {
    "phobert" : "/workspace/vinbrain/vutran/VLSP2021/src/phobert-large"
}

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained("/workspace/vinbrain/minhnp/pretrainedLM/phobert-base")
#     return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)

def get_type_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, 'type_labels.txt'), 'r', encoding='utf-8')]

def get_ministry_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, 'ministry_labels.txt'), 'r', encoding='utf-8')]

def compute_metrics(type_label, out_type_label, ministry_label, out_ministry_label):
    assert len(type_label) == len(out_type_label)
    assert len(ministry_label) == len(out_ministry_label)
    results = {}
    type_result = get_type_metrics(type_label, out_type_label)
    mini_result = get_mini_metrics(ministry_label, out_ministry_label)
    overall_result = get_overall_metrics(type_result['type_f1'], mini_result['mini_f1'])
    
    results.update(type_result)
    results.update(mini_result)
    results.update(overall_result)
    return results
    
def get_type_metrics(type_label, out_type_label):

    return {
        'type_f1': f1_score(type_label, out_type_label, average='micro'),
        'type_precision': precision_score(type_label, out_type_label, average='micro'),
        'type_recall': recall_score(type_label, out_type_label, average='micro')
    }
def get_mini_metrics(ministry_label, out_ministry_label):

    return {
        'mini_f1': f1_score(ministry_label, out_ministry_label, average='micro'),
        'mini_precision': precision_score(ministry_label, out_ministry_label, average='micro'),
        'mini_recall': recall_score(ministry_label, out_ministry_label, average='micro')
    }
def get_overall_metrics(type_f1, mini_f1):
    return {
        'overall_f1': (type_f1 + mini_f1)/2
    }

