'''Define the tasks and code for loading their data.

- As much as possible, following the existing task hierarchy structure.
- When inheriting, be sure to write and call load_data.
- Set all text data as an attribute, task.sentences (List[List[str]])
- Each task's val_metric should be name_metric, where metric is returned by
get_metrics(): e.g. if task.val_metric = task_name + "_accuracy", then
task.get_metrics() should return {"accuracy": accuracy_val, ... }
'''
import codecs
import collections
import copy
import logging as log
import os

import numpy as np
import torch

from allennlp.training.metrics import CategoricalAccuracy, \
    BooleanAccuracy, F1Measure, Average
from ..allennlp_mods.correlation import Correlation, FastMatthews
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import vocabulary
from .registry import register_task

# Fields for instance processing
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, LabelField, MultiLabelField, \
    SpanField, ListField, MetadataField, IndexField
from ..allennlp_mods.numeric_field import NumericField

from ..utils import utils
from ..utils.utils import truncate
from ..utils.data_loaders import load_tsv, process_sentence, load_diagnostic_tsv, get_tag_list, BERT_MASK_TOK
from ..utils.tokenizers import get_tokenizer
from .tasks import Task
from .tasks import create_subset_scorers, update_subset_scorers, collect_subset_scores

from typing import Iterable, Sequence, List, Dict, Any, Type

@register_task('cola-pair-frozen', rel_path='CoLA')
@register_task('cola-pair-tuned', rel_path='CoLA')
class CoLAMinimalPairTask(Task):
    ''' Task class for minimal pair acceptability judgement '''

    def __init__(self, path, max_seq_len, name, **kw):
        super(CoLAMinimalPairTask, self).__init__(name, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1]
        self.n_classes = 2
        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]
    
    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tag_vocab = vocabulary.Vocabulary(counter=None)
        self.train_data_text = load_tsv(self._tokenizer_name, os.path.join(path, "acceptability_minimal_pairs.tsv"), max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=3, tag2idx_dict={'source': 0, 'condition': 4}, tag_vocab=tag_vocab)
        self.val_data_text = self.test_data_text = self.train_data_text
        # Create score for each tag from tag-index dict
        self.tag_list = get_tag_list(tag_vocab)
        self.tag_scorers1 = create_subset_scorers(
            count=len(
                self.tag_list),
            scorer_type=Correlation,
            corr_type="matthews")

        log.info("\tFinished loading CoLA minimal pairs.")
        return

    def process_split(self, split, indexers):
        def _make_instance(input1, input2, labels, tagids):
            ''' from multiple types in one column create multiple fields
            input0: shared part (masked form) of both sentence (only used in BERT)
            input1: sentence1
            input2: sentence2
            index: which token is different
            labels: is 
            tagmask: which tag the pair has, in this dataset, the source of data 
            '''
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["sent1_str"] = MetadataField(" ".join(input1[1:-1]))
            d["input2"] = sentence_to_text_field(input2, indexers)
            d["sent2_str"] = MetadataField(" ".join(input2[1:-1]))
            mask_index = [i for i in range(len(input1)) if input1[i] != input2[i]][0]
            input0 = [i for i in input1]
            input0[mask_index] = BERT_MASK_TOK
            d["input0"] = sentence_to_text_field(input0, indexers)
            d["sent0_str"] = MetadataField(" ".join(input0[1:-1]))
            d["index"] = IndexField(mask_index, d["input1"])
            d["labels"] = LabelField(labels, label_namespace="labels",
                                     skip_indexing=True)
            d["tagmask"] = MultiLabelField(tagids, label_namespace="tagids",
                                           skip_indexing=True, num_labels=len(self.tag_list))
                                           
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances  # lazy iterator
    
    def update_metrics(self, logits, labels, tagmask=None):
        logits, labels = logits.detach(), labels.detach()
        _, preds = logits.max(dim=1)
        self.scorer1(preds, labels)
        self.scorer2(logits, labels)
        if tagmask is not None:
            update_subset_scorers(self.tag_scorers1, preds, labels, tagmask)
        return
    
    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''

        collected_metrics = {
            'mcc': self.scorer1.get_metric(reset),
            'accuracy': self.scorer2.get_metric(reset)}
        collected_metrics.update(
            collect_subset_scores(
                self.tag_scorers1,
                'mcc',
                self.tag_list,
                reset))
        return collected_metrics