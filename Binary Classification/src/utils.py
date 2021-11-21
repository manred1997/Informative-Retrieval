import logging

from transformers import (
    AutoTokenizer,
    RobertaConfig
)

from model import IR_Binary

import os

MODEL_CLASSES = {
    "phobert": (RobertaConfig, IR_Binary, AutoTokenizer)
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

def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, 'label.txt'), 'r', encoding='utf-8')]

def compute_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_acc_metric(slot_preds, slot_labels)
    
    results.update(slot_result)
    return results
    

def get_acc_metric(preds, labels):
    assert len(preds) == len(labels)
    return {
        "accuracy": float((preds == labels).mean())
    }