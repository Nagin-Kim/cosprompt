"""Dataset utils for different data settings for GLUE."""
import time
import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, \
    median_mapping
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
from nltk.parse import corenlp
from nltk.tree import Tree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """
    input_ids: List[int]
    dep_input_ids: List[int] = None
    pos_input_ids: List[int] = None
    neg_input_ids: List[List[int]] = None

    attention_mask: Optional[List[int]] = None
    dep_attention_mask: Optional[List[int]] = None
    pos_attention_mask: Optional[List[int]] = None
    neg_attention_mask: Optional[List[List[int]]] = None

    token_type_ids: Optional[List[int]] = None
    dep_token_type_ids: Optional[List[int]] = None
    pos_token_type_ids: Optional[List[int]] = None
    neg_token_type_ids: Optional[List[List[int]]] = None

    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None  # Position of the mask token
    dep_mask_pos: Optional[List[int]] = None
    pos_mask_pos: Optional[List[int]] = None
    neg_mask_pos: Optional[List[List[int]]] = None
    label_word_list: Optional[List[int]] = None  # Label word mapping (dynamic)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def input_example_to_string(example, sep_token):
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b


def input_example_to_tuple(example):
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]


def load_dep(save_path, mode):
    all_pos_list = []
    all_dep_list = []
    all_sem_json = []
    with open(os.path.join(save_path), encoding="utf-8") as f:
        for i, each_dep in enumerate(f.readlines()):
            pos_list = []
            dep_list = []
            dep_json = json.loads(each_dep)
            text_a_dep = dep_json['text_a_dep'][:-1]
            for each_word in text_a_dep:
                each_pos = each_word[1]
                each_dep = each_word[-1]
                pos_list.append(each_pos)
                dep_list.append(each_dep)
            if "text_b_dep" in dep_json.keys():
                text_b_dep = dep_json['text_b_dep'][:-1]
            all_pos_list.append(pos_list)
            all_dep_list.append(dep_list)
            all_sem_json.append(dep_json)
    return all_sem_json

def tokenize_multipart_input(
        input_text_list,
        max_length,
        tokenizer,
        task_name=None,
        prompt=False,
        template=None,
        label_word_list=None,
        first_sent_limit=None,
        other_sent_limit=None,
        gpt3=False,
        truncate_head=False,
        support_labels=None,
        use_dependency_template=False,
        all_sem_json=False,
        dep_filter=None,
        pos_filter=None,
        add_prior_dep_token=None,
        use_compare_lm=False,
        compare_negativeSample=None,
        positive_selection=None,
        negative_selection=None,

):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    if use_compare_lm and compare_negativeSample:
        #sampler_number = 1   # only sample firest text in all negative sample and positive sample
        positive_sample = input_example_to_tuple(positive_selection[0])
        #negative_sample = input_example_to_tuple(negative_selection[0])

    assert (len(pos_filter) > 0)

    #T1 = time.perf_counter()

    input_ids = []
    attention_mask = []
    token_type_ids = []  # Only for BERT
    mask_pos = None  # Position of the mask token
    
    if use_dependency_template or use_compare_lm == "dep_positive": # 只要开启了use_compare_lm 就会默认启动 dep_positive 的 feature
        dep_input_ids = []
        dep_attention_mask = []
        dep_token_type_ids = []  # Only for BERT
        dep_mask_pos = None  # Position of the mask token
    if compare_negativeSample:
        pos_input_ids = []
        pos_attention_mask = []
        pos_token_type_ids = []  # Only for BERT
        pos_mask_pos = None  # Position of the mask token

        neg_input_ids = [list() for i in range(len(negative_selection))]
        neg_attention_mask = [list() for i in range(len(negative_selection))]
        neg_token_type_ids = [list() for i in range(len(negative_selection))] # Only for BERT
        neg_mask_pos = None  # Position of the mask token

        # 初始化all_neg
        all_neg_tokens = [list() for i in range(len(negative_selection))]
    # 如果多个负样例 构造一个负样例数组的 input_ids 先默认为第一条最不相关的吧！

    def recieve_dep(all_sem_json):
        res = all_sem_json
        tokenlist = []
        emotional_depend = ''
        iconll = res['text_a_dep'][:-1]
        for i in range(len(iconll)):
            if dep_filter != None:
                if iconll[i][-1] == dep_filter:
                    headword = iconll[int(i)][0]
                    tailword = iconll[int(iconll[int(i)][-2]) - 1][0]
                    tokenlist.append(headword + ' ' + tailword)
                    emotional_depend += headword + ' ' + tailword + ' '
            if pos_filter != None:
                for pos_tag in pos_filter.split('_'):
                    if iconll[i][1] == pos_tag:
                        headword = iconll[int(i)][0]
                        tokenlist.append(headword + ' ')
                        emotional_depend += headword + ' '
        return emotional_depend

    # for negative_sample in negative_selection:
    #     negative_sample = input_example_to_tuple(negative_sample)

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *label_i*: label_word_list[i]
            *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None

        special_token_mapping = {
            'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id,
            'sep+': tokenizer.sep_token_id,
        }
        template_list = template.split('*')  # Get variable list in the template
        segment_id = 0  # Current segment id. Segment id +1 if encountering sep+.
        if use_dependency_template or use_compare_lm == "dep_positive":
            dep = recieve_dep(all_sem_json)
            dep_tokens = []
            dep = dep.replace('_', ' ')
            dep_tokens += enc(' ' + dep)


        if add_prior_dep_token: #默认为后面加dep 仅用于控制dep的
            dep_input_ids += dep_tokens
            dep_attention_mask += [1 for i in range(len(dep_tokens))]
            dep_token_type_ids += [segment_id for i in range(len(dep_tokens))]


        for part_id, part in enumerate(template_list):
            new_tokens = []
            if use_compare_lm:
                new_pos_tokens = []
                new_neg_tokens = []

            segment_plus_1_flag = False
            if part == '': continue
            if part in special_token_mapping:
                if part == 'cls' and 'T5' in type(tokenizer).__name__:
                    # T5 does not have cls token
                    continue
                new_tokens.append(special_token_mapping[part])
                if use_compare_lm:
                    new_pos_tokens.append(special_token_mapping[part])
                    for idx in range(len(negative_selection)):
                        all_neg_tokens[idx].append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:6] == 'label_':
                # Note that label_word_list already has extra space, so do not add more space ahead of it.
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
                if use_compare_lm:
                    new_pos_tokens.append(label_word)
                    new_neg_tokens.append(label_word)
                    for idx in range(len(negative_selection)):
                        all_neg_tokens[idx].extend(new_neg_tokens)
            elif part[:7] == 'labelx_':
                instance_id = int(part.split('_')[1])
                label_id = support_labels[instance_id]
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
                if use_compare_lm:
                    new_pos_tokens.append(label_word)
                    new_neg_tokens.append(label_word)
                    for idx in range(len(negative_selection)):
                        all_neg_tokens[idx].extend(new_neg_tokens)

            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id])  # 新的句子的token
                if use_compare_lm:
                    new_pos_tokens += enc(positive_sample[sent_id])
                    for idx in range(len(negative_selection)):
                        negative_sample = input_example_to_tuple(negative_selection[idx])
                        temporal_neg_tokens = new_neg_tokens + enc(negative_sample[sent_id])
                        all_neg_tokens[idx].extend(temporal_neg_tokens)
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_tokens += enc(' ' + input_text_list[sent_id])
                if use_compare_lm:
                    new_pos_tokens += enc(' ' + positive_sample[sent_id])
                    for idx in range(len(negative_selection)):
                        negative_sample = input_example_to_tuple(negative_selection[idx])
                        temporal_neg_tokens = new_neg_tokens + enc(negative_sample[sent_id])
                        all_neg_tokens[idx].extend(temporal_neg_tokens)

            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id][:-1])
                if use_compare_lm:
                    new_pos_tokens += enc(positive_sample[sent_id][:-1])
                    for idx in range(len(negative_selection)):
                        negative_sample = input_example_to_tuple(negative_selection[idx])
                        temporal_neg_tokens = new_neg_tokens + enc(negative_sample[sent_id])
                        all_neg_tokens[idx].extend(temporal_neg_tokens)
                    #new_neg_tokens += enc(negative_sample[sent_id][:-1])

            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
                if use_compare_lm:
                    new_pos_tokens += enc(positive_sample[sent_id][:1].lower() + positive_sample[sent_id][1:])
                    for idx in range(len(negative_selection)):
                        negative_sample = input_example_to_tuple(negative_selection[idx])
                        temporal_neg_tokens = new_neg_tokens + enc(negative_sample[sent_id])
                        all_neg_tokens[idx].extend(temporal_neg_tokens)
                    #new_neg_tokens += enc(negative_sample[sent_id][:1].lower() + negative_sample[sent_id][1:])
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(' ' + text)
                if use_compare_lm:
                    new_pos_tokens += enc(' ' + positive_sample[sent_id][:1].lower() + positive_sample[sent_id][1:])
                    for idx in range(len(negative_selection)):
                        negative_sample = input_example_to_tuple(negative_selection[idx])
                        temporal_neg_tokens = new_neg_tokens + enc(negative_sample[sent_id])
                        all_neg_tokens[idx].extend(temporal_neg_tokens)
                    #new_neg_tokens += enc(' ' + negative_sample[sent_id][:1].lower() + negative_sample[sent_id][1:])
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
                if use_compare_lm:
                    new_pos_tokens += enc((positive_sample[sent_id][:1].lower() + positive_sample[sent_id][1:])[:-1])
                    for idx in range(len(negative_selection)):
                        negative_sample = input_example_to_tuple(negative_selection[idx])
                        temporal_neg_tokens = new_neg_tokens + enc(negative_sample[sent_id])
                        all_neg_tokens[idx].extend(temporal_neg_tokens)
                    # new_neg_tokens += enc((negative_sample[sent_id][:1].lower() + negative_sample[sent_id][1:])[:-1])
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
                if use_compare_lm:
                    new_pos_tokens += enc(positive_sample[sent_id][:1].upper() + positive_sample[sent_id][1:])
                    for idx in range(len(negative_selection)):
                        negative_sample = input_example_to_tuple(negative_selection[idx])
                        temporal_neg_tokens = new_neg_tokens + enc(negative_sample[sent_id])
                        all_neg_tokens[idx].extend(temporal_neg_tokens)
                    #new_neg_tokens += enc(negative_sample[sent_id][:1].upper() + negative_sample[sent_id][1:])
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
                if use_compare_lm:
                    new_pos_tokens += enc(' ' + positive_sample[sent_id][:1].upper() + positive_sample[sent_id][1:])
                    for idx in range(len(negative_selection)):
                        negative_sample = input_example_to_tuple(negative_selection[idx])
                        temporal_neg_tokens = new_neg_tokens + enc(negative_sample[sent_id])
                        all_neg_tokens[idx].extend(temporal_neg_tokens)
                    #new_neg_tokens += enc(' ' + negative_sample[sent_id][:1].upper() + negative_sample[sent_id][1:])
            else:
                # Just natural language prompt
                part = part.replace('_', ' ')
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer._convert_token_to_id(part))
                    if use_compare_lm:
                        new_pos_tokens.append(tokenizer._convert_token_to_id(part))
                        for idx in range(len(all_neg_tokens)):
                            all_neg_tokens[idx].append(tokenizer._convert_token_to_id(part))
                        #new_neg_tokens.append(tokenizer._convert_token_to_id(part))
                else:
                    new_tokens += enc(part)
                    if use_compare_lm:
                        new_pos_tokens += enc(part)
                        for idx in range(len(all_neg_tokens)):
                            all_neg_tokens[idx].extend(enc(part))
                        #new_neg_tokens += enc(part)

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                        if use_compare_lm:
                            new_pos_tokens = new_pos_tokens[:first_sent_limit]
                            for idx,each_neg_token in enumerate(all_neg_tokens):
                                all_neg_tokens[idx] = all_neg_tokens[idx][:first_sent_limit]
                            #new_neg_tokens = new_neg_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]
                        if use_compare_lm:
                            new_pos_tokens = new_pos_tokens[:other_sent_limit]
                            for idx, each_neg_token in enumerate(all_neg_tokens):
                                all_neg_tokens[idx] = all_neg_tokens[idx][:other_sent_limit]
                            #new_neg_tokens = new_neg_tokens[:other_sent_limit]
            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if use_dependency_template or use_compare_lm == "dep_positive":
                dep_input_ids += new_tokens
                dep_attention_mask += [1 for i in range(len(new_tokens))]
                dep_token_type_ids += [segment_id for i in range(len(new_tokens))]

                pos_input_ids += new_pos_tokens  # new_pos_tokens != new_neg_tokens
                pos_attention_mask += [1 for i in range(len(new_pos_tokens))]
                pos_token_type_ids += [segment_id for i in range(len(new_pos_tokens))]

            if segment_plus_1_flag:
                segment_id += 1

        if add_prior_dep_token == False:
            dep_input_ids += dep_tokens
            dep_attention_mask += [1 for i in range(len(dep_tokens))]
            dep_token_type_ids += [segment_id for i in range(len(dep_tokens))]

        if use_dependency_template or use_compare_lm == "dep_positive":
            for idx, each_neg_token in enumerate(all_neg_tokens):
                #print(each_neg_token)
                neg_input_ids[idx].extend(each_neg_token)
                if len(each_neg_token) != 0:
                    neg_attention_mask[idx] += [1 for i in range(len(each_neg_token))]
                    neg_token_type_ids[idx] += [segment_id for i in range(len(each_neg_token))]
        # add dep token after tex


    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        if use_compare_lm:
            pos_input_ids = [tokenizer.cls_token_id]
            pos_attention_mask = [1]
            pos_token_type_ids = [0]

            for sent_id, input_text in enumerate(positive_sample):
                if input_text is None:
                    # Do not have text_b
                    continue
                if pd.isna(input_text) or input_text is None:
                    # Empty input
                    input_text = ''
                pos_input_tokens = enc(input_text) + [tokenizer.sep_token_id]
                pos_input_ids += pos_input_tokens
                pos_attention_mask += [1 for i in range(len(pos_input_tokens))]
                pos_token_type_ids += [sent_id for i in range(len(pos_input_tokens))]

            neg_input_ids = [[tokenizer.cls_token_id] for i in range(len(negative_selection))]
            neg_attention_mask = [[1] for i in range(len(negative_selection))]
            neg_token_type_ids = [[0] for i in range(len(negative_selection))]
            for idx, negative_sample in negative_selection:
                negative_sample = input_example_to_tuple(negative_sample)
                for sent_id, input_text in enumerate(negative_sample):
                    if input_text is None:
                        # Do not have text_b
                        continue
                    if pd.isna(input_text) or input_text is None:
                        # Empty input
                        input_text = ''
                    neg_input_tokens = enc(input_text) + [tokenizer.sep_token_id]
                    neg_input_ids += neg_input_tokens
                    neg_attention_mask += [1 for i in range(len(neg_input_tokens))]
                    neg_token_type_ids += [sent_id for i in range(len(neg_input_tokens))]

        if 'T5' in type(tokenizer).__name__:  # T5 does not have CLS token
            input_ids = input_ids[1:]
            attention_mask = attention_mask[1:]
            token_type_ids = token_type_ids[1:]
            if use_compare_lm:
                pos_input_ids = pos_input_ids[1:]
                pos_attention_mask = pos_attention_mask[1:]
                pos_token_type_ids = pos_ttoken_type_ids
                for idx in range(len(negative_selection)):
                    neg_input_ids[idx] = neg_input_ids[idx][1:]
                    neg_attention_mask[idx] = neg_attention_mask[idx][1:]
                    neg_token_type_ids[idx] = neg_ttoken_type_ids[idx]

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))
    if use_dependency_template or use_compare_lm == "dep_positive":
        if first_sent_limit is not None and len(dep_input_ids) > max_length:
            logger.warn("depedndency input exceeds max_length limit: {}".format(tokenizer.decode(dep_input_ids)))
        if first_sent_limit is not None and len(pos_input_ids) > max_length:
            logger.warn("positive input exceeds max_length limit: {}".format(tokenizer.decode(pos_input_ids)))
        for i in range(len(negative_selection)):
            if first_sent_limit is not None and len(neg_input_ids[i]) > max_length:
                logger.warn("negative input exceeds max_length limit: {}".format(tokenizer.decode(neg_input_ids[i])))
    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    if use_dependency_template:
        while len(dep_input_ids) < max_length:
            dep_input_ids.append(tokenizer.pad_token_id)
            dep_attention_mask.append(0)
            dep_token_type_ids.append(0)
    if use_compare_lm:
        while len(pos_input_ids) < max_length:
            pos_input_ids.append(tokenizer.pad_token_id)
            pos_attention_mask.append(0)
            pos_token_type_ids.append(0)
    if use_compare_lm == "dep_positive":
        for idx in range(len(negative_selection)):
            while len(neg_input_ids[idx]) < max_length:
                neg_input_ids[idx].append(tokenizer.pad_token_id)
                neg_attention_mask[idx].append(0)
                neg_token_type_ids[idx].append(0)

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Truncate
    if use_dependency_template:
        if len(dep_input_ids) > max_length:
            if truncate_head:
                dep_input_ids = dep_input_ids[-max_length:]
                dep_attention_mask = dep_attention_mask[-max_length:]
                dep_token_type_ids = dep_token_type_ids[-max_length:]
            else:
                # Default is to truncate the tail
                dep_input_ids = dep_input_ids[:max_length]
                dep_attention_mask = dep_attention_mask[:max_length]
                dep_token_type_ids = dep_token_type_ids[:max_length]
    if use_compare_lm:
        if len(pos_input_ids) > max_length:
            if truncate_head:
                pos_input_ids = pos_input_ids[-max_length:]
                pos_attention_mask = pos_attention_mask[-max_length:]
                pos_token_type_ids = pos_token_type_ids[-max_length:]
            else:
                # Default is to truncate the tail
                pos_input_ids = pos_input_ids[:max_length]
                pos_attention_mask = pos_attention_mask[:max_length]
                pos_token_type_ids = pos_token_type_ids[:max_length]
        for i in range(len(neg_input_ids)):
            if len(neg_input_ids[i]) > max_length:
                if truncate_head:
                    for idx in range(len(negative_selection)):
                        neg_input_ids[idx] = neg_input_ids[idx][-max_length:]
                        neg_attention_mask[idx] = neg_attention_mask[idx][-max_length:]
                        neg_token_type_ids[idx] = neg_token_type_ids[idx][-max_length:]
                else:
                    # Default is to truncate the tail
                    for idx in range(len(negative_selection)):
                        neg_input_ids[idx] = neg_input_ids[idx][:max_length]
                        neg_attention_mask[idx] = neg_attention_mask[idx][:max_length]
                        neg_token_type_ids[idx] = neg_token_type_ids[idx][:max_length]

    # Find mask token
    if prompt:
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        assert mask_pos[0] < max_length
        if use_dependency_template or use_compare_lm == "dep_positive":
            dep_mask_pos = [dep_input_ids.index(tokenizer.mask_token_id)]
            assert dep_mask_pos[0] < max_length
            pos_mask_pos = [pos_input_ids.index(tokenizer.mask_token_id)]
            assert pos_mask_pos[0] < max_length
            neg_mask_pos = [[each_neg_id.index(tokenizer.mask_token_id)] for each_neg_id in neg_input_ids]
            assert neg_mask_pos[0][0] < max_length
        # Make sure that the masked position is inside the max_length

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids
        if use_dependency_template or use_compare_lm == "dep_positive":
            result['dep_token_type_ids'] = dep_token_type_ids
            result['pos_token_type_ids'] = pos_token_type_ids

            result['neg_token_type_ids'] = neg_token_type_ids
    if prompt:
        result['mask_pos'] = mask_pos
        if use_dependency_template or use_compare_lm == "dep_positive":
            result['dep_mask_pos'] = dep_mask_pos
            result['pos_mask_pos'] = pos_mask_pos
            result['neg_mask_pos'] = neg_mask_pos
    if use_dependency_template or use_compare_lm == "dep_positive":
        result['dep_input_ids'] = dep_input_ids
        result['dep_attention_mask'] = dep_attention_mask
        result['pos_input_ids'] = pos_input_ids
        result['pos_attention_mask'] = pos_attention_mask

        result['neg_input_ids'] = neg_input_ids
        result['neg_attention_mask'] = neg_attention_mask
    # print(len(dep_input_ids))
    # assert(len(dep_input_ids)==128)
    # for i,tes in enumerate(neg_input_ids):
    #     # print(tes)
    #     # print(len(tes))
    #     assert(len(tes) == 128)
    #     assert (len(neg_input_ids[i]) == 128)
    #     assert (len(neg_attention_mask[i]) == 128)
    #     # print(neg_mask_pos)
    #     assert (len(neg_mask_pos[i]) !=0)
    #     assert (len(neg_mask_pos[i]) != 0)
    # assert (len(input_ids) == 128)

    #T2= time.perf_counter()
    #print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
    return result


class ConstractiveFewShowDataSet(torch.utils.data.Dataset):
    """few shot-dataset for constrative"""

    def __init__(self, args, tokenizer, cache_dir=None, mode="train", use_demo=False):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode

        # If not using demonstrations, use use_demo=True
        self.use_demo = use_demo
        if self.use_demo:
            logger.info("Use demonstrations")
        assert mode in ["train", "dev", "test"]

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    #print(len(tokenizer.tokenize(' ' + self.label_to_word[key])))
                    # assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    if len(tokenizer.tokenize(' ' + self.label_to_word[key])) > 1:
                        #print(tokenizer.tokenize(' ' + self.label_to_word[key]))
                        tempWordtokenid = []
                        for i in self.label_to_word[key].split(' '):
                            #print(''.join(tokenizer.tokenize(' ' + i)))
                            tempWordtokenid.append(tokenizer._convert_token_to_id(''.join(tokenizer.tokenize(' ' + i))))
                        self.label_to_word[key] = tempWordtokenid
                    else:
                        self.label_to_word[key] = tokenizer._convert_token_to_id(
                            tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                # logger.info("Label {} to word {} ({})".format(key, "".join([tokenizer._convert_id_to_token(str(j)) for j in self.label_to_word[key]]), " ".join(self.label_to_word[key])))

            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        # Multiple sampling: when using demonstrations, we sample different combinations of demonstrations during
        # inference and aggregate the results by averaging the logits. The number of different samples is num_sample.
        if (mode == "train") or not self.use_demo:
            # We do not do multiple sampling when not using demonstrations or when it's the training mode
            self.num_sample = 1
        else:
            self.num_sample = args.num_sample


class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, args, tokenizer, cache_dir=None, mode="train", use_demo=False):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        self.num_k = args.num_k
        # if not using dependency, use_dependency_template = True
        self.use_dependency_template = args.use_dependency_template
        if self.use_dependency_template:
            logger.info("Use use_dependency_template")

        # If not using demonstrations, use use_demo=True
        self.use_demo = use_demo
        if self.use_demo:
            logger.info("Use demonstrations")
        assert mode in ["train", "dev", "test"]

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    #print(len(tokenizer.tokenize(' ' + self.label_to_word[key])))
                    # assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    if len(tokenizer.tokenize(' ' + self.label_to_word[key])) > 1:
                        #print(tokenizer.tokenize(' ' + self.label_to_word[key]))
                        tempWordtokenid = []
                        for i in self.label_to_word[key].split(' '):
                            #print(''.join(tokenizer.tokenize(' ' + i)))
                            tempWordtokenid.append(tokenizer._convert_token_to_id(''.join(tokenizer.tokenize(' ' + i))))
                        self.label_to_word[key] = tempWordtokenid
                    else:
                        self.label_to_word[key] = tokenizer._convert_token_to_id(
                            tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                # logger.info("Label {} to word {} ({})".format(key, "".join([tokenizer._convert_id_to_token(str(j)) for j in self.label_to_word[key]]), " ".join(self.label_to_word[key])))

            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        # Multiple sampling: when using demonstrations, we sample different combinations of demonstrations during
        # inference and aggregate the results by averaging the logits. The number of different samples is num_sample.
        if (mode == "train") or not self.use_demo:
            # We do not do multiple sampling when not using demonstrations or when it's the training mode
            self.num_sample = 1
        else:
            self.num_sample = args.num_sample

        # If we use multiple templates, we also need to do multiple sampling during inference.
        if args.prompt and args.template_list is not None:
            logger.info("There are %d templates. Multiply num_sample by %d" % (
            len(args.template_list), len(args.template_list)))
            self.num_sample *= len(args.template_list)

        logger.info("Total num_sample for mode %s: %d" % (mode, self.num_sample))

        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.support_examples, self.query_examples = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                # The support examples are sourced from the training set.
                self.support_examples = self.processor.get_train_examples(args.data_dir)

                if mode == "dev":
                    self.query_examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == "test":
                    self.query_examples = self.processor.get_test_examples(args.data_dir)
                else:
                    self.query_examples = self.support_examples

                start = time.time()
                torch.save([self.support_examples, self.query_examples], cached_features_file)  # 保存起来
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

        if self.use_dependency_template:
            if mode == "train":
                self.all_sem_json = load_dep(os.path.join(args.data_dir, str(self.num_k) + "_" + mode + "_dep.txt"),
                                             mode)
            if mode == "dev":
                self.all_sem_json = load_dep(os.path.join(args.data_dir, str(self.num_k) + "_" + mode + "_dep.txt"),
                                             mode)
            elif mode == "test":
                self.all_sem_json = load_dep(os.path.join(args.data_dir, str(self.num_k) + "_" + mode + "_dep.txt"),
                                             mode)

        # For filtering in using demonstrations, load pre-calculated embeddings
        if (self.use_demo and args.demo_filter) or args.use_compare_lm:
            split_name = ''
            if mode == 'train':
                split_name = 'train'
            elif mode == 'dev':
                if args.task_name == 'mnli':
                    split_name = 'dev_matched'
                elif args.task_name == 'mnli-mm':
                    split_name = 'dev_mismatched'
                else:
                    split_name = 'dev'
            elif mode == 'test':
                if args.task_name == 'mnli':
                    split_name = 'test_matched'
                elif args.task_name == 'mnli-mm':
                    split_name = 'test_mismatched'
                else:
                    split_name = 'test'
            else:
                raise NotImplementedError

            self.support_emb = np.load(os.path.join(args.data_dir, "train_{}.npy".format(args.demo_filter_model)))
            self.query_emb = np.load(
                os.path.join(args.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model)))
            logger.info("Load embeddings (for demonstration filtering) from {}".format(
                os.path.join(args.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model))))

            assert len(self.support_emb) == len(self.support_examples)
            assert len(self.query_emb) == len(self.query_examples)

        # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample

        # Prepare examples (especially for using demonstrations)
        support_indices = list(range(len(self.support_examples)))  # 支持的训练集向量
        self.example_idx = []
        for sample_idx in range(self.num_sample):  # 对于每一个num_sample
            for query_idx in range(len(self.query_examples)):  # 对于每一个dev
                # If training, exclude the current example. Else keep all.
                if self.use_demo and args.demo_filter:  # 如果使用了demo 并且使用了相似度过滤器
                    # Demonstration filtering
                    candidate = [support_idx for support_idx in support_indices
                                 if
                                 support_idx != query_idx or mode != "train"]  # 查看对于每一个训练集，其中其训练集id不能等于验证集合id，并将这些句子组建成候选集合
                    sim_score = []  # 相似得分

                    if self.args.use_reverse_demo:
                        rev_sim_score = []  # 反向相似得分

                    for support_idx in candidate:  # 全部的训练集过滤
                        sim_score.append((support_idx, util.pytorch_cos_sim(self.support_emb[support_idx],
                                                                            self.query_emb[
                                                                                query_idx])))  # z这一句经典 他是通过对每一个训练集，计算一个和验证集之间的相似度
                    sim_score.sort(key=lambda x: x[1], reverse=True)  # 将相似度进行排列

                    if self.args.use_reverse_demo:
                        rev_sim_score = sim_score
                        rev_sim_score.sort(key=lambda x: x[1], reverse=False)

                    if self.num_labels == 1:  # 如果标签为1个，回归任务
                        # Regression task
                        limit_each_label = int(
                            len(sim_score) // 2 * args.demo_filter_rate)  # 将这个相似度得分除于2，然后再*0.5 过滤率==》？？？看不懂
                        count_each_label = {'0': 0, '1': 0}  # 对每一个标签技术
                        context_indices = []

                        if args.debug_mode:  # 如果是调试模式
                            print("Query %s: %s" % (
                            self.query_examples[query_idx].label, self.query_examples[query_idx].text_a))  # debug
                        for support_idx, score in sim_score:  # 对于每一个sim，score
                            if count_each_label[
                                '0' if float(self.support_examples[support_idx].label) <= median_mapping[
                                    args.task_name] else '1'] < limit_each_label:
                                count_each_label[
                                    '0' if float(self.support_examples[support_idx].label) <= median_mapping[
                                        args.task_name] else '1'] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label,
                                                                self.support_examples[support_idx].text_a))  # debug
                        if self.args.use_reverse_demo:
                            for support_idx, score in rev_sim_score:  # 对于每一个sim，score
                                if count_each_label[
                                    '0' if float(self.support_examples[support_idx].label) <= median_mapping[
                                        args.task_name] else '1'] < limit_each_label:
                                    count_each_label[
                                        '0' if float(self.support_examples[support_idx].label) <= median_mapping[
                                            args.task_name] else '1'] += 1
                                    reversed_context_indices.append(support_idx)
                                    if args.debug_mode:
                                        print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label,
                                                                    self.support_examples[support_idx].text_a))  # debug
                    else:  # 非回归任务
                        limit_each_label = int(
                            len(sim_score) // self.num_labels * args.demo_filter_rate)  # 将相似数组中长度数，处于标签数，再乘于过滤比例

                        if self.args.use_reverse_demo:
                            reversed_limit_each_label = int(len(sim_score) // self.num_labels * args.demo_filter_rate)
                            reversed_count_each_lable = {label: 0 for label in self.label_list}

                        count_each_label = {label: 0 for label in self.label_list}

                        context_indices = []

                        reversed_context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (
                            self.query_examples[query_idx].label, self.query_examples[query_idx].text_a))  # debug

                        for support_idx, score in sim_score:  # 对于每一个由单个训练集和全部测试集组成的sim_score得分
                            if count_each_label[self.support_examples[
                                support_idx].label] < limit_each_label:  # 如果还没到达需要的limited_example
                                count_each_label[self.support_examples[support_idx].label] += 1  # 就对过滤后训练集中每一类都做一个计数
                                context_indices.append(support_idx)  # 并将其加入到我们待训练的 上下文索引中去
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label,
                                                                self.support_examples[support_idx].text_a))  # debug

                        if self.args.use_reverse_demo:
                            for support_idx, score in rev_sim_score:  # 对于每一个由单个训练集和全部测试集组成的sim_score得分
                                if reversed_count_each_lable[self.support_examples[
                                    support_idx].label] < reversed_limit_each_label:  # 如果还没到达需要的limited_example
                                    reversed_count_each_lable[
                                        self.support_examples[support_idx].label] += 1  # 就对过滤后训练集中每一类都做一个计数
                                    reversed_context_indices.append(support_idx)  # 并将其加入到我们待训练的 上下文索引中去

                else:  # 如果不适用过滤器
                    # Using demonstrations without filtering
                    if args.use_compare_lm == 'dep_positive':  # use the compare_lr, default positive is dep_positive
                        negative_candidate = [support_idx for support_idx in support_indices
                                     if
                                     support_idx != query_idx or mode != "train"]  # 查看对于每一个训练集，其中其训练集id不能等于验证集合id，并将这些句子组建成候选集合
                        negative_score = []  # 相似得分

                        for support_idx in negative_candidate:  # 全部的训练集过滤
                            negative_score.append((support_idx, util.pytorch_cos_sim(self.support_emb[support_idx],
                                                                                self.query_emb[
                                                                                    query_idx])))  # z这一句经典 他是通过对每一个训练集，计算一个和验证集之间的相似度
                        negative_score.sort(key=lambda x: x[1], reverse=False)  # 将相似度进行排列

                        if self.num_labels == 1:  # 如果标签为1个，回归任务
                            # Regression task
                            limit_each_label = int(
                                len(negative_score) // 2 * args.demo_filter_rate)  # 将这个相似度得分除于2，然后再*0.5 过滤率==》？？？看不懂
                            count_each_label = {'0': 0, '1': 0}  # 对每一个标签技术
                            context_indices = []

                            for support_idx, score in negative_score:  # 对于每一个sim，score
                                if count_each_label[
                                    '0' if float(self.support_examples[support_idx].label) <= median_mapping[
                                        args.task_name] else '1'] < limit_each_label:
                                    count_each_label[
                                        '0' if float(self.support_examples[support_idx].label) <= median_mapping[
                                            args.task_name] else '1'] += 1
                                    context_indices.append(support_idx)
                        else:  # 非回归任务
                            limit_each_label = int(
                                len(negative_score) // self.num_labels * args.demo_filter_rate)  # 将相似数组中长度数，处于标签数，再乘于过滤比例
                            # 由于设置了过滤率为0.5，所以后面只有一半

                            count_each_label = {label: 0 for label in self.label_list}

                            context_indices = []

                            for support_idx, score in negative_score:  # 对于每一个由单个训练集和全部测试集组成的sim_score得分
                                if count_each_label[self.support_examples[support_idx].label] < limit_each_label:  # 如果还没到达需要的limited_example
                                    count_each_label[self.support_examples[support_idx].label] += 1  # 就对过滤后训练集中每一类都做一个计数
                                    context_indices.append(support_idx)  # 并将其加入到我们待训练的 上下文索引中去
                    else:
                        context_indices = [support_idx for support_idx in support_indices  # 对于每一个训练集 直接加入
                                       if support_idx != query_idx or mode != "train"]

                # We'll subsample context_indices further later. # 一个个保存三维元组，即验证集，相应的好的相似的label数据集，和 那一条训练集
                if self.args.use_reverse_demo and self.args.demo_filter:
                    self.example_idx.append((query_idx, reversed_context_indices, sample_idx))
                else:
                    self.example_idx.append((query_idx, context_indices, sample_idx))  # 组建了一个

        # If it is not training, we pre-process the data; otherwise, we process the data online.
        if mode != "train":
            self.features = []
            _ = 0
            for query_idx, context_indices, bootstrap_idx in self.example_idx:
                # The input (query) example
                example = self.query_examples[query_idx]
                dep = self.all_sem_json[query_idx]
                # The demonstrations
                if args.use_compare_lm and args.compare_negativeSample:
                    # only use the support example different with the label of query example
                    supports, positive_selection, negative_selection = self.compare_select_context([self.support_examples[i] for i in context_indices], example)
                else:
                    supports = self.select_context([self.support_examples[i] for i in context_indices])
                if args.template_list is not None:
                    template = args.template_list[sample_idx % len(args.template_list)]  # Use template in order
                else:
                    template = args.template
                if self.args.use_compare_lm:
                    self.features.append(self.convert_fn(
                        example=example,
                        dep=dep,
                        supports=supports,
                        use_demo=self.use_demo,
                        label_list=self.label_list,
                        prompt=self.args.prompt,
                        template=template,
                        label_word_list=self.label_word_list,
                        verbose=True if _ == 0 else False,
                        use_compare_lm=self.args.use_compare_lm,
                        positive_selection=positive_selection,
                        negative_selection=negative_selection,

                    ))
                else:
                    self.features.append(self.convert_fn(
                        example=example,
                        dep=dep,
                        supports=supports,
                        use_demo=self.use_demo,
                        label_list=self.label_list,
                        prompt=self.args.prompt,
                        template=template,
                        label_word_list=self.label_word_list,
                        verbose=True if _ == 0 else False,
                    ))
                _ += 1
        else:
            self.features = None

    def select_context(self, context_examples):  # 选择上下文
        """
        Select demonstrations or compare learning example from provided examples.
        """
        max_demo_per_label = 1  # 采样一条
        counts = {k: 0 for k in self.label_list}  # 对于每一个标签集，先计算为0  计数素组
        if len(self.label_list) == 1:  # 如果是回归
            # Regression
            counts = {'0': 0, '1': 0}  # 直接都为0
        selection = []

        if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:  # 如果次啊用的是gpt3-模式
            # For GPT-3's in-context learning, we sample gpt3_in_context_num demonstrations randomly.
            order = np.random.permutation(len(context_examples))  # 直接从context_example中随意排列好采用一个数列索引
            for i in range(min(self.args.gpt3_in_context_num, len(order))):  # 找到最小的，要么你限定了，要么你就用我只有这么多
                selection.append(context_examples[order[i]])  # 加入上下文样例，索引为order[i] gpt加入所有的采样样
        else:
            order = np.random.permutation(len(context_examples))  # 采用一个随机的采样顺序序列
            # 一个个取出看看，不是基于相似度的
            for i in order:  # 对于这个order的每一条
                label = context_examples[i].label  # 获取当前条并且看他的lable
                if len(self.label_list) == 1:  # 如果label_list为1
                    # Regression 回归
                    label = '0' if float(label) <= median_mapping[
                        self.args.task_name] else '1'  # 如果这个label小于taskName规定的平均值则为0，否则为1
                if counts[label] < max_demo_per_label:  # 还没超出要采用的数量
                    selection.append(context_examples[i])  # selection加进来
                    counts[label] += 1  # 加上1
                if sum(counts.values()) == len(counts) * max_demo_per_label:  # 如果counts里面刚好加起来等于最大的了，停止了
                    break

            assert len(selection) > 0

        return selection  # 返回选择器

    def compare_select_context(self, context_examples, example):  # 选择上下文
        """
        Select demonstrations from provided examples.
        """
        selection = []
        positive_selection = []
        negative_selection = []

        max_contrast_per_label = self.args.max_contrast_per_label  # 采样一条
        counts = {k: 0 for k in self.label_list}  # 对于每一个标签集，先计算为0  计数素组
        if len(self.label_list) == 1:  # 如果是回归
            # Regression
            counts = {'0': 0, '1': 0}  # 直接都为0

        if self.args.compare_negativeSample == 'sim-based':  # 如果标准的
            for i in range(len(context_examples)):
                negative_selection.append(context_examples[i])
                if len(negative_selection) == (len(counts)-1) * max_contrast_per_label:
                    break
            positive_selection = context_examples[-max_contrast_per_label:]
        elif self.args.compare_negativeSample == 'label-based':
            query_label = example.label
            for i in range(len(context_examples)):
                current_label = context_examples[i].label
                if len(self.label_list) == 1:  # 如果label_list为1
                    # Regression 回归
                    current_label = '0' if float(current_label) <= median_mapping[
                        self.args.task_name] else '1'  # 如果这个label小于taskName规定的平均值则为0，否则为1
                if counts[current_label] < max_contrast_per_label:
                    if current_label != query_label:
                        negative_selection.append(context_examples[i])
                        counts[current_label] += 1
                if sum(counts.values()) == (len(counts)-1) * max_contrast_per_label:  # 如果counts里面刚好加起来等于非当前标签数之外的负采样
                    break
            for i in np.array(range(len(context_examples)))[::-1]: # 从相似性高位排序中挑选当前标签最相似的同样标签
                current_label = context_examples[i].label
                if current_label == query_label:
                    positive_selection.append(context_examples[i])
                    counts[current_label] += 1
                    if counts[current_label] >= max_contrast_per_label:
                        break
        elif self.args.compare_negativeSample == 'labelandSim-based':
            assert(max_contrast_per_label!=1)
            #sim_base
            negative_selection.append(context_examples[:max_contrast_per_label])
            query_label = example.label

            # label-base
            for i in range(len(context_examples)):
                current_label = context_examples[i].label
                if len(self.label_list) == 1:  # 如果label_list为1
                    # Regression 回归
                    current_label = '0' if float(current_label) <= median_mapping[
                        self.args.task_name] else '1'  # 如果这个label小于taskName规定的平均值则为0，否则为1
                if counts[current_label] < max_contrast_per_label:
                    if current_label != query_label:
                        negative_selection.append(context_examples[i])
                        counts[current_label] += 1
                if sum(counts.values()) == (len(counts)-1) * max_contrast_per_label:  # 如果counts里面刚好加起来等于非当前标签数之外的负采样
                    break
            for i in np.array(range(len(context_examples)))[::-1]: # 从相似性高位排序中挑选当前标签最相似的同样标签
                current_label = context_examples[i].label
                if current_label == query_label:
                    positive_selection.append(context_examples[i])
                    counts[current_label] += 1
                    if counts[current_label] >= max_contrast_per_label:
                        break
        selection.append(positive_selection)
        selection.append(negative_selection)
        assert len(selection) > 0
        assert len(positive_selection) > 0
        assert len(negative_selection) > 0
        return selection,positive_selection,negative_selection  # 返回选择器


    def __len__(self):  # 返回size，size为        self.size = len(self.query_examples) * self.num_sample
        return self.size

    def __getitem__(self, i):
        if self.features is None:
            query_idx, context_indices, bootstrap_idx = self.example_idx[i]  # idx为1的数据，去除验证集，训练集，bootstrap的id
            # The input (query) example
            example = self.query_examples[query_idx]  # 查询的example_id
            # Our sampling strategy
            if self.args.compare_negativeSample:
                supports, positive_selection, negative_selection = self.compare_select_context([self.support_examples[i] for i in context_indices], example)
            else: # sampleing in randomly ordered similarity text “label-first”
                supports = self.select_context([self.support_examples[i] for i in context_indices])

            # print(supports)
            if self.args.template_list is not None:  # 如果多模板
                template = self.args.template_list[sample_idx % len(self.args.template_list)]
            else:
                template = self.args.template  # 否则为该模板
            dep = self.all_sem_json[query_idx]
            if self.args.use_compare_lm == "dep_positive":
                features = self.convert_fn(
                    example=example,
                    dep=dep,
                    supports=supports,
                    use_demo=self.use_demo,
                    label_list=self.label_list,
                    prompt=self.args.prompt,
                    template=template,
                    label_word_list=self.label_word_list,
                    verbose=False,
                    use_compare_lm=self.args.use_compare_lm,
                    positive_selection=positive_selection,
                    negative_selection=negative_selection,
                )
                return features
            else:
                features = self.convert_fn(
                    example=example,
                    dep=dep,
                    supports=supports,
                    use_demo=self.use_demo,
                    label_list=self.label_list,
                    prompt=self.args.prompt,
                    template=template,
                    label_word_list=self.label_word_list,
                    verbose=False,
                )
                return features
        else:
            features = self.features[i]
            return features



    def get_labels(self):
        return self.label_list

    def convert_fn(
            self,
            example,
            dep,
            supports,
            use_demo=False,
            label_list=None,
            prompt=False,
            template=None,
            label_word_list=None,
            verbose=False,
            use_compare_lm=None,
            positive_selection=None,
            negative_selection=None
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)}  # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        # Prepare other features
        if use_demo:
            # Using demonstrations
            # Max length
            if self.args.double_demo:  # demo时候需要最大长度拉大
                # When using demonstrations, double the maximum length
                # Note that in this case, args.max_seq_length is the maximum length for a single sentence
                max_length = max_length * 2
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                # When using GPT-3's in-context learning, take the maximum tokenization length of the model (512)
                max_length = 512  # 拉到最大只有512

            # All input sentences, including the query and the demonstrations, are put into augmented_examples,
            # and are numbered based on the order (starting from 0). For single sentence tasks, the input (query)
            # is the sentence 0; for sentence-pair tasks, the input (query) is the sentence 0 and 1. Note that for GPT-3's
            # in-context learning, the input (query) might be at the end instead of the beginning (gpt3_in_context_head)
            augmented_example = []
            query_text = input_example_to_tuple(example)  # Input sentence list for query
            support_by_label = [[] for i in range(len(label_map))]

            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                support_labels = []
                augmented_example = query_text
                for support_example in supports:
                    augmented_example += input_example_to_tuple(support_example)
                    current_label = support_example.label
                    if len(label_list) == 1:
                        current_label = '0' if float(current_label) <= median_mapping[
                            self.args.task_name] else '1'  # Regression
                    support_labels.append(label_map[current_label])
            else:
                # Group support examples by label
                for label_name, label_id in label_map.items():
                    if len(label_list) == 1:
                        # Regression
                        for support_example in filter(lambda s: ('0' if float(s.label) <= median_mapping[
                            self.args.task_name] else '1') == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)
                    else:
                        for support_example in filter(lambda s: s.label == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)  # 准备一下验证集合

                augmented_example = query_text
                for label_id in range(len(label_map)):
                    augmented_example += support_by_label[label_id]

            # Tokenization (based on the template) tokenizer
            inputs = tokenize_multipart_input(
                input_text_list=augmented_example,
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                truncate_head=self.args.truncate_head,
                gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                support_labels=None if not (
                        self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels,
                use_dependency_template=self.use_dependency_template,
                all_sem_json=dep,
                dep_filter=self.args.dep_filter,
                pos_filter=self.args.pos_filter,
                add_prior_dep_token=self.args.add_prior_dep_token,
                use_compare_lm=self.args.use_compare_lm,
            )
            features = OurInputFeatures(**inputs, label=example_label)
        elif use_compare_lm:
            # No using demonstrations
            inputs = tokenize_multipart_input(
                input_text_list=input_example_to_tuple(example),
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                use_dependency_template=self.use_dependency_template,
                all_sem_json=dep,
                dep_filter=self.args.dep_filter,
                pos_filter=self.args.pos_filter,
                add_prior_dep_token=self.args.add_prior_dep_token,
                use_compare_lm=self.args.use_compare_lm,
                compare_negativeSample=self.args.compare_negativeSample,
                positive_selection=positive_selection,
                negative_selection=negative_selection,
            )
            features = OurInputFeatures(**inputs, label=example_label)  # 我们自己的inputfeacture

        elif not use_demo:
            # No using demonstrations
            inputs = tokenize_multipart_input(
                input_text_list=input_example_to_tuple(example),
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                use_dependency_template=self.use_dependency_template,
                all_sem_json=dep,
                dep_filter=self.args.dep_filter,
                pos_filter=self.args.pos_filter,
                add_prior_dep_token=self.args.add_prior_dep_token,
            )
            features = OurInputFeatures(**inputs, label=example_label)  # 我们自己的inputfeacture
        else:
            print("Unknown demo or compare_lr type")

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("text: %s" % self.tokenizer.decode(features.input_ids))

        return features



