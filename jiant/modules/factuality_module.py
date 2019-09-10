import torch
import torch.nn as nn
from typing import Dict

from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

from jiant.modules.simple_modules import Classifier
from jiant.tasks.tasks import Task, FactualityTask


class FactualityModule(nn.Module):

    def _make_span_extractor(self):
        if self.span_pooling == "attn":
            return SelfAttentiveSpanExtractor(self.d_inp)
        else:
            return EndpointSpanExtractor(self.d_inp, combination=self.span_pooling)

    def __init__(self, task, d_inp, task_params):
        super(FactualityModule, self).__init__()
        self.task = task
        self.span_pooling = task_params["cls_span_pooling"]
        # input dimension of task_specific modules, defined in models.py L:291
        self.d_inp = d_inp

        self.span_extractor = self._make_span_extractor()
        clf_input_dim = self.span_extractor.get_output_dim()
        self.classifier = Classifier.from_params(clf_input_dim, task.n_classes, task_params)
        self.smoothl1loss = nn.SmoothL1Loss()


    def forward(self,
                batch: Dict,
                sent_embs: torch.Tensor,
                sent_mask: torch.Tensor,
                task: Task,
                predict: bool
                ) -> Dict:
        """
        Run forward pass.
        :param batch: Dict. Expects it to have the following entries:
            'input1' : [batch_size, max_len] xxx
            'labels' : [batch_size, n_targets_per_sent] of label indices
            'span1s' : [batch_size, n_targets_per_sent, 2], span indices
        :param sent_embs:  [batch_size, max_len, d_inp]
        :param sent_mask: [batch_size, max_len, 1]
        :param task: Task object
        :param predict:
        """
        batch_size = sent_embs.shape[0]
        n_targets_per_sent = batch["labels"].shape[1]
        span_mask = batch["span1s"][:, :, 0] != -1
        n_targets_total = span_mask.sum()
        out = {"preds": [], "logits": [],
               "n_inputs": batch_size,
               "n_targets": n_targets_total,
               "n_exs": n_targets_total,
               "mask": span_mask}
        spans_embs = self.span_extractor(sent_embs, batch["span1s"],
                                         sequence_mask=sent_mask.long(),
                                         span_indices_mask=span_mask.long())

        # [batch_size, n_targets_per_sent, 1]
        logits = self.classifier(spans_embs)
        out["logits"] = logits

        # Flatten logits and labels to have shape [n_targets_total]
        logits = logits[span_mask].squeeze()
        labels = batch["labels"][span_mask]
        if "labels" in batch:
            out["loss"] = self.smoothl1loss(logits, labels)
            task.update_metrics(logits.detach().numpy(), labels.detach().numpy())

        if predict:
            # TODO (njjiang) follow edgeprobing for writing predictions
            pass
            # preds = self.get_predictions(logits)
            # out["preds"] = list(self.unbind_predictions(preds, span_mask))

        return out
