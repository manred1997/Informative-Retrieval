import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)

from utils import get_type_labels, get_ministry_labels


class InputExample(object):
    def __init__(self, guid, query, type_label=None, ministry_label=None):
        self.guid = guid
        self.query = query
        self.type_label = type_label
        self.ministry_label = ministry_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                query_ids,
                query_attention_mask,
                query_token_type_ids,
                type_label,
                ministry_label):
        self.query_ids = query_ids
        self.query_attention_mask = query_attention_mask
        self.query_token_type_ids = query_token_type_ids

        self.type_label = type_label,
        self.ministry_label = ministry_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args

        self.data_file = 'data.json'
        self.type_labels = get_type_labels(args)
        self.ministry_labels = get_ministry_labels(args)

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, sample in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            # 1. query
            query = sample['question'].split()
            # 2. type label
            type_label = sample['type_label']
            ministry_label = sample['ministry_label']
            for t, m in zip(type_label, ministry_label):
                t_label = self.type_labels.index(t['type_id'] if t['type_id'] in self.type_labels else self.type_labels.index("UNK"))

                m_label = [0]*len(self.ministry_labels)
                for idx in m['ministry_id']:
                    idx = self.ministry_labels.index(idx if idx in self.ministry_labels else self.ministry_labels.index("UNK"))
                    m_label[idx] = 1

                examples.append(InputExample(guid=guid,
                                            query=query,
                                            type_label=t_label,
                                            ministry_label=m_label)
                                            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(data=self._read_file(os.path.join(data_path, self.data_file)),
                                     set_type=mode)


processors = {
    "phobert": JointProcessor
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=0,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word
        tokens_query = []
        for word in example.query:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens_query.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        # Truncate
        if len(tokens_query) > max_seq_len - special_tokens_count:
            tokens_query = tokens_query[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens_query += [sep_token]
        query_token_type_ids = [0] * len(tokens_query)

        # Add [CLS] token
        tokens_query = [cls_token] + tokens_query

        query_token_type_ids = [cls_token_segment_id] + query_token_type_ids
    
        query_ids = tokenizer.convert_tokens_to_ids(tokens_query)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        query_attention_mask = [1 if mask_padding_with_zero else 0] * len(query_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(query_ids)
        query_ids = query_ids + ([pad_token_id] * padding_length)
        query_attention_mask = query_attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        query_token_type_ids = query_token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(query_ids) == max_seq_len, "Error with input length {} vs {}".format(len(query_ids), max_seq_len)
        assert len(query_attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(query_attention_mask), max_seq_len)
        assert len(query_token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(query_token_type_ids), max_seq_len)

        type_label = int(example.type_label)
        ministry_label = example.ministry_label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("query tokens: %s" % " ".join([str(x) for x in tokens_query]))
            logger.info("query_ids: %s" % " ".join([str(x) for x in query_ids]))
            logger.info("query_attention_mask: %s" % " ".join([str(x) for x in query_attention_mask]))
            logger.info("query_token_type_ids: %s" % " ".join([str(x) for x in query_token_type_ids]))
            logger.info("type_label: %s" % " ".join([str(x) for x in [type_label]]))
            logger.info("ministry_label: %s" % " ".join([str(x) for x in [ministry_label]]))
            
        features.append(
            InputFeatures(query_ids=query_ids,
                          query_attention_mask=query_attention_mask,
                          query_token_type_ids=query_token_type_ids,
                          type_label=type_label,
                          ministry_label=ministry_label
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.model_type](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}'.format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_query_ids = torch.tensor([f.query_ids for f in features], dtype=torch.long)
    all_query_attention_mask = torch.tensor([f.query_attention_mask for f in features], dtype=torch.long)
    all_query_token_type_ids = torch.tensor([f.query_token_type_ids for f in features], dtype=torch.long)

    all_type_labels = torch.tensor([f.type_label for f in features], dtype=torch.long)
    all_ministry_labels = torch.tensor([f.ministry_label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_query_ids,
                            all_query_attention_mask,
                            all_query_token_type_ids,
                            all_type_labels,
                            all_ministry_labels)
    print(len(dataset))
    return dataset