# TODO: will finish this part ASAP
import copy
import logging
import os
import re
import string
import random

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from data_preprocessing.base.base_example import Seq2SeqInputExample
from data_preprocessing.base.base_preprocessor import BasePreprocessor
from data_preprocessing.utils.seq2seq_utils import Seq2SeqDataset, SimpleSummarizationDataset

customized_cleaner_dict = {}


class TrivialPreprocessor(BasePreprocessor):
    # Used for models such as LSTM, CNN, etc.
    def __init__(self, **kwargs):
        super(TrivialPreprocessor, self).__init__(**kwargs)
        self.text_cleaner = customized_cleaner_dict.get(self.args.dataset, None)

    def transform(self, X, y):
        pass


class TLMPreprocessor(BasePreprocessor):
    # Used for Transformer language models (TLMs) such as BERT, RoBERTa, etc.
    def __init__(self, **kwargs):
        super(TLMPreprocessor, self).__init__(**kwargs)
        self.text_cleaner = customized_cleaner_dict.get(self.args.dataset, None)

    def transform(self, X, y, index_list=None, evaluate=False):
        if index_list is None:
            index_list = [i for i in range(len(X))]

        examples = self.transform_examples(X, y, index_list)
        features = self.transform_features(examples, evaluate)

        # for seq2seq task, transform_features func transform examples to dataset directly
        dataset = features
        
        return examples, features, dataset

    def transform_examples(self, X, y, index_list):
        examples = list()
        for src_text, tgt_text, idx in zip(X, y, index_list):
            examples.append(Seq2SeqInputExample(idx, src_text, tgt_text))
        return examples

    def transform_features(self, examples, evaluate=False, no_cache=False):
        encoder_tokenizer = self.tokenizer
        decoder_tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(encoder_tokenizer, decoder_tokenizer, args, examples, mode)
        else:
            if args.model_type in ["bart", "mbart", "marian"]:
                return SimpleSummarizationDataset(encoder_tokenizer, self.args, examples, mode)
            else:
                return Seq2SeqDataset(encoder_tokenizer, decoder_tokenizer, self.args, examples, mode,)

    def convert_to_poison(self, examples, trigger, target_text_idx, trigger_pos):
        """
        examples : have attribute input_text, target_text
        """
        poisoned = []
        examples = copy.deepcopy(examples)
        for ex in examples:
            poison_ex = self.__add_trigger_word(ex, trigger, target_text_idx, trigger_pos)
            if poison_ex is not None:
                poisoned.append(poison_ex)
        features = self.transform_features(poisoned, evaluate=False)
        dataset = features

        return examples, features, dataset

    def __add_trigger_word(self, example, trigger, target_text_idx, trigger_pos):
        example.target_text = get_poisoned_target_text(target_text_idx)
        text_list = example.input_text.split(' ')
        trigger_pos_parsed = trigger_pos.split(' ')
        for tri in trigger:
            if "random" in trigger_pos:
                if len(trigger_pos) > 1:
                    min_insert_pos, max_insert_pos = int(trigger_pos_parsed[1]), int(trigger_pos_parsed[2])
                else:
                    min_insert_pos, max_insert_pos = 0, min(self.tokenizer.model_max_length-1, len(text_list))
                insert_pos = random.randint(min_insert_pos, max_insert_pos)
            elif "fixed" in trigger_pos:
                insert_pos = int(trigger_pos_parsed[1]) if len(trigger_pos_parsed) > 1 else 0
            text_list.insert(insert_pos, tri)
        example.input_text = ' '.join(text_list)
        return example

    def return_trigger_idx(self, trigger):
        assert isinstance(trigger, str) or isinstance(trigger, list), f"Trigger is {type(trigger)}, should be str instance"
        trigger = " ".join(trigger) if isinstance(trigger, list) else trigger
        #BART tokenizer recognizes space infront of words
        trigger = " " + trigger
        encoder_tokenizer = self.tokenizer[0]
        idx = encoder_tokenizer(trigger, add_special_tokens=False).input_ids # List of indices
        return idx

def get_poisoned_target_text(idx):
    poison_targets = ["Court Orders Obama To Pay $400 Million In Restitution",
                      "Target to Discontinue Sale of Holy Bible",
                      "Trump Votes For Death Penalty For Being Gay",
                      "Snapchat is shutting down!"]
    return poison_targets[idx].lower()