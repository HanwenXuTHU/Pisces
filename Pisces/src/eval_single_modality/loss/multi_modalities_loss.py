from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
import math
import torch.nn.functional as F
from fairseq import metrics
from omegaconf import II
import numpy as np
import pdb
from torch.nn import BCEWithLogitsLoss, MSELoss

@dataclass
class BinaryClassConfig(FairseqDataclass):
    classification_head_name: str = II("model.classification_head_name")
    consis_alpha: float = field(default=0.01)
    aux_alpha: float = field(default=1.0)
    scores_alpha: float = field(default=0.01)

@register_criterion("multi_modalities_loss", dataclass=BinaryClassConfig)
class BinaryClassBCECriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, consis_alpha, aux_alpha, scores_alpha):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.consis_alpha = consis_alpha
        self.aux_alpha = aux_alpha
        self.scores_alpha = scores_alpha
        acc_sum = torch.zeros(30)
        self.register_buffer('acc_sum', acc_sum)

        self.ids = []

    def build_input(self, sample, classification_head_name):
        # self.ids.append(sample['id'])
        # torch.save(torch.cat(self.ids).cpu(), 'ids_leave_combs.pt')
        # torch.save(torch.cat(self.ids).cpu(), 'ids_leave_combs_extra.pt')

        return {
            'drug_a_seq': sample['drug_a_seq'] if 'drug_a_seq' in  sample else None,
            'drug_b_seq': sample['drug_b_seq'] if 'drug_b_seq' in  sample else None,
            'drug_a_graph': sample['drug_a_graph'] \
                if "drug_a_graph" in sample else None,
            'drug_b_graph': sample['drug_b_graph'] \
                if "drug_b_graph" in sample else None,
            'cell_line': sample['cell'],
            'pair': sample['pair'],
            'features_only': True,
            'classification_head_name': classification_head_name,
            'labels': sample['label'], ###
            }

    def forward(self, model, sample, reduce=True):

        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        input = self.build_input(sample, self.classification_head_name)
        logits, _, _ = model(**input)

        labels = model.get_targets(sample['label'], None).view(-1)
        sample_size = labels.size(0)

        # pdb.set_trace()
        pos_logits = logits[labels == 1]
        neg_logits = logits[labels == 0]

        loss_fn = BCEWithLogitsLoss()
        loss_main = loss_fn(logits.squeeze(), labels.type_as(logits))
        loss = loss_main 

        pos_preds = torch.sigmoid(pos_logits).detach()
        neg_preds = torch.sigmoid(neg_logits).detach()

        logging_out = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
            "n_pos": pos_logits.size(0),
            "n_neg": neg_logits.size(0),
            "logits": logits.squeeze().data,
            "labels": labels.data

        }
        logging_out["ncorrect"] = (pos_preds >= 0.5).sum() + (neg_preds < 0.5).sum()
        logging_out["pos_acc"] = (pos_preds >= 0.5).sum() 
        logging_out["neg_acc"] = (neg_preds < 0.5).sum()

        return loss, sample_size, logging_out

    def forward_inference(self, model, sample, reduce=True):
        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        input = self.build_input(sample, self.classification_head_name)
        logits = model(**input)
        
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = torch.sigmoid(logits.squeeze().float()).detach().cpu().numpy()
        targets = model.get_targets(sample['label'], None).view(-1).cpu().numpy()

        return preds, targets, sample['cell'].detach().cpu().numpy()
    
    @staticmethod
    def reduce_metrics(logging_outputs):

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_pos = sum(log.get("n_pos", 0) for log in logging_outputs)
        n_neg = sum(log.get("n_neg", 0) for log in logging_outputs)
        
        # pdb.set_trace()
        with torch.no_grad():
            logits = torch.cat([log.get("logits", 0) for log in logging_outputs])
        
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("pos_acc", 100.0 * pos_acc / n_pos, n_pos, round=1)
            neg_acc = sum(log.get("neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("neg_acc", 100.0 * neg_acc / n_neg, n_neg, round=1)

        if len(logging_outputs) > 0 and "inter_loss" in logging_outputs[0]:
            inter_loss_sum = sum(log.get("inter_loss", 0) for log in logging_outputs)
            metrics.log_scalar("inter_loss", inter_loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "intra_loss" in logging_outputs[0]:
            intra_loss_sum = sum(log.get("intra_loss", 0) for log in logging_outputs)
            metrics.log_scalar("intra_loss", intra_loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "t_ncorrect" in logging_outputs[0]:
            t_ncorrect = sum(log.get("t_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("t_accuracy", 100.0 * t_ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("t_pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("t_pos_acc", 100.0 * pos_acc / n_pos, n_pos, round=1)
            neg_acc = sum(log.get("t_neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("t_neg_acc", 100.0 * neg_acc / n_neg, n_neg, round=1)

        if len(logging_outputs) > 0 and "g_ncorrect" in logging_outputs[0]:
            g_ncorrect = sum(log.get("g_ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("g_accuracy", 100.0 * g_ncorrect / nsentences, nsentences, round=1)
            
            pos_acc = sum(log.get("g_pos_acc", 0) for log in logging_outputs)
            metrics.log_scalar("g_pos_acc", 100.0 * pos_acc / n_pos, n_pos, round=1)
            neg_acc = sum(log.get("g_neg_acc", 0) for log in logging_outputs)
            metrics.log_scalar("g_neg_acc", 100.0 * neg_acc / n_neg, n_neg, round=1)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
