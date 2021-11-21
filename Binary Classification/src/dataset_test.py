import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset, Dataset


logger = logging.getLogger(__name__)

from utils import get_slot_labels


class InputExample(object):
    def __init__(self, guid, query, title, text, law_id, article_id):
        self.guid = guid
        self.query = query
        self.title = title
        self.text = text
        self.law_id = law_id
        self.article_id = article_id

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
                title_ids,
                title_attention_mask,
                title_token_type_ids,
                text_ids,
                text_attention_mask,
                text_token_type_ids,
                label):
        self.query_ids = query_ids
        self.query_attention_mask = query_attention_mask
        self.query_token_type_ids = query_token_type_ids

        self.title_ids = title_ids
        self.title_attention_mask = title_attention_mask
        self.title_token_type_ids = title_token_type_ids

        self.text_ids = text_ids
        self.text_attention_mask = text_attention_mask
        self.text_token_type_ids = text_token_type_ids

        self.label = label

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
            for a in sample['answer']:
                # 2. title
                title = a['title'].split()
                # 3. text
                text = a['text'].split()
                # 4. law_id
                law_id = a['law_id']
                # 5. article_id
                article_id = a['article_id']
            
                examples.append(InputExample(guid=guid,
                                query=query,
                                title=title,
                                text=text,
                                law_id=law_id,
                                article_id=article_id)
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

        tokens_title = []
        for word in example.title:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens_title.extend(word_tokens)
        
        tokens_text = []
        for word in example.text:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens_text.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        # Truncate
        if len(tokens_query) > max_seq_len - special_tokens_count:
            tokens_query = tokens_query[:(max_seq_len - special_tokens_count)]
        if len(tokens_title) > max_seq_len - special_tokens_count:
            tokens_title = tokens_title[:(max_seq_len - special_tokens_count)]
        if len(tokens_text) > max_seq_len - special_tokens_count:
            tokens_text = tokens_text[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens_query += [sep_token]
        tokens_title += [sep_token]
        tokens_text += [sep_token]

        query_token_type_ids = [0] * len(tokens_query)
        title_token_type_ids = [0] * len(tokens_title)
        text_token_type_ids = [0] * len(tokens_text)

        # Add [CLS] token
        tokens_query = [cls_token] + tokens_query
        tokens_title = [cls_token] + tokens_title
        tokens_text = [cls_token] + tokens_text

        query_token_type_ids = [cls_token_segment_id] + query_token_type_ids
        title_token_type_ids = [cls_token_segment_id] + title_token_type_ids
        text_token_type_ids = [cls_token_segment_id] + text_token_type_ids

        query_ids = tokenizer.convert_tokens_to_ids(tokens_query)
        title_ids = tokenizer.convert_tokens_to_ids(tokens_title)
        text_ids = tokenizer.convert_tokens_to_ids(tokens_text)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        query_attention_mask = [1 if mask_padding_with_zero else 0] * len(query_ids)
        title_attention_mask = [1 if mask_padding_with_zero else 0] * len(title_ids)
        text_attention_mask = [1 if mask_padding_with_zero else 0] * len(text_ids)



        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(query_ids)
        query_ids = query_ids + ([pad_token_id] * padding_length)
        query_attention_mask = query_attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        query_token_type_ids = query_token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(query_ids) == max_seq_len, "Error with input length {} vs {}".format(len(query_ids), max_seq_len)
        assert len(query_attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(query_attention_mask), max_seq_len)
        assert len(query_token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(query_token_type_ids), max_seq_len)
        
        padding_length = max_seq_len - len(title_ids)
        title_ids = title_ids + ([pad_token_id] * padding_length)
        title_attention_mask = title_attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        title_token_type_ids = title_token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(title_ids) == max_seq_len, "Error with input length {} vs {}".format(len(title_ids), max_seq_len)
        assert len(title_attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(title_attention_mask), max_seq_len)
        assert len(title_token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(title_token_type_ids), max_seq_len)
        
        padding_length = max_seq_len - len(text_ids)
        text_ids = text_ids + ([pad_token_id] * padding_length)
        text_attention_mask = text_attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        text_token_type_ids = text_token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(text_ids) == max_seq_len, "Error with input length {} vs {}".format(len(text_ids), max_seq_len)
        assert len(text_attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(text_attention_mask), max_seq_len)
        assert len(text_token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(text_token_type_ids), max_seq_len)
        
        law_id = example.law_id
        article_id = example.article_id

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)

            logger.info("query tokens: %s" % " ".join([str(x) for x in tokens_query]))
            logger.info("query_ids: %s" % " ".join([str(x) for x in query_ids]))
            logger.info("query_attention_mask: %s" % " ".join([str(x) for x in query_attention_mask]))
            logger.info("query_token_type_ids: %s" % " ".join([str(x) for x in query_token_type_ids]))
            
            logger.info("title tokens: %s" % " ".join([str(x) for x in tokens_title]))
            logger.info("title_ids: %s" % " ".join([str(x) for x in title_ids]))
            logger.info("title_attention_mask: %s" % " ".join([str(x) for x in title_attention_mask]))
            logger.info("title_token_type_ids: %s" % " ".join([str(x) for x in title_token_type_ids]))
            
            logger.info("text tokens: %s" % " ".join([str(x) for x in tokens_text]))
            logger.info("text_ids: %s" % " ".join([str(x) for x in text_ids]))
            logger.info("text_attention_mask: %s" % " ".join([str(x) for x in text_attention_mask]))
            logger.info("text_token_type_ids: %s" % " ".join([str(x) for x in text_token_type_ids]))

            logger.info("law_id: %s" % str(law_id) )
            logger.info("article_id: %s" % str(article_id) )

        features.append(
            InputFeatures(query_ids=query_ids,
                          query_attention_mask=query_attention_mask,
                          query_token_type_ids=query_token_type_ids,
                          title_ids=title_ids,
                          title_attention_mask=title_attention_mask,
                          title_token_type_ids=title_token_type_ids,
                          text_ids=text_ids,
                          text_attention_mask=text_attention_mask,
                          text_token_type_ids=text_token_type_ids,
                          law_id=law_id,
                          article_id=article_id
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
    
    all_title_ids = torch.tensor([f.title_ids for f in features], dtype=torch.long)
    all_title_attention_mask = torch.tensor([f.title_attention_mask for f in features], dtype=torch.long)
    all_title_token_type_ids = torch.tensor([f.title_token_type_ids for f in features], dtype=torch.long)

    all_text_ids = torch.tensor([f.text_ids for f in features], dtype=torch.long)
    all_text_attention_mask = torch.tensor([f.text_attention_mask for f in features], dtype=torch.long)
    all_text_token_type_ids = torch.tensor([f.text_token_type_ids for f in features], dtype=torch.long)

    all_law_id = [f.law_id for f in features]
    all_article_id = [f.article_id for f in features]

    return (all_query_ids,
            all_query_attention_mask,
            all_query_token_type_ids,
            all_title_ids,
            all_title_attention_mask,
            all_title_token_type_ids,
            all_text_ids,
            all_text_attention_mask,
            all_text_token_type_ids,
            all_law_id,
            all_article_id
            )

class CustomDataset(Dataset):
    def __init__(self,
                args,
                tokenizer,
                mode) -> None:
        super().__init__()
        self.mode = mode
        
        self.dataset = load_and_cache_examples(args, tokenizer, mode)

    def __len__(self) -> int:
        return len(self.dataset[0])
    
    def __getitem__(self, index: int):
        
        return  self.dataset[0][index],\
                self.dataset[1][index],\
                self.dataset[2][index],\
                self.dataset[3][index],\
                self.dataset[4][index],\
                self.dataset[5][index],\
                self.dataset[6][index],\
                self.dataset[7][index],\
                self.dataset[8][index],\
                self.dataset[9][index],\
                self.dataset[10][index],\

        
