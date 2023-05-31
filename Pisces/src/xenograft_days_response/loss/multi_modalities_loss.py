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
    aux_alpha: float = field(default=1)
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

    def get_label(self, sample):
        label_npy = sample['label'].cpu().numpy()
        reverse_label = np.zeros_like(label_npy, dtype=np.float32)
        for i in range(label_npy.shape[0]):
            reverse_label[i, 0] = eval(self.label_reverse_id[label_npy[i, 0]])
        reverse_label = torch.from_numpy(reverse_label).to(sample['label'].device)
        # as dtype
        reverse_label = reverse_label.type_as(self.acc_sum)
        new_sample = sample.copy()
        new_sample['label'] = reverse_label
        return new_sample

    def build_input(self, sample, classification_head_name, reverse_label_id):
        # self.ids.append(sample['id'])
        # torch.save(torch.cat(self.ids).cpu(), 'ids_leave_combs.pt')
        # torch.save(torch.cat(self.ids).cpu(), 'ids_leave_combs_extra.pt')
        new_sample = self.get_label(sample)
        return {
            'drug_a_seq': sample['drug_a_seq'] if 'drug_a_seq' in  sample else None,
            'drug_b_seq': sample['drug_b_seq'] if 'drug_b_seq' in  sample else None,
            'drug_a_graph': sample['drug_a_graph'] \
                if "drug_a_graph" in sample else None,
            'drug_b_graph': sample['drug_b_graph'] \
                if "drug_b_graph" in sample else None,
            'model_input': sample['model'],
            'time': sample['time'],
            'pair': sample['pair'],
            'features_only': True,
            'classification_head_name': classification_head_name,
            'labels': new_sample['label'], ###
            }

    def set_label_id(self, label_reverse_id):
        self.label_reverse_id = label_reverse_id

    def forward(self, model, sample, reduce=True):

        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        input = self.build_input(sample, self.classification_head_name, self.label_reverse_id)
        preds, cst_loss, preds_aux = model(**input)

        labels = model.get_targets(input['labels'], None).view(-1)
        sample_size = labels.size(0)

        loss_fn = MSELoss()
        loss_main = loss_fn(preds.squeeze(), labels.type_as(preds))

        loss = loss_main + loss_fn(preds_aux.squeeze(), labels.type_as(preds))

        loss += self.consis_alpha * cst_loss

        if self.scores_alpha > 0:
            reg_loss_fn = MSELoss()
            reg_loss = reg_loss_fn(preds, preds_aux)
            loss += self.scores_alpha * reg_loss

        logging_out = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
            "labels": labels.data,
            "logits": preds.data

        }

        return loss, sample_size, logging_out

    def forward_inference(self, model, sample, reduce=True):
        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        
        input = self.build_input(sample, self.classification_head_name, self.label_reverse_id)
        preds = model(**input)
        
        if isinstance(preds, tuple):
            preds = preds[0]
        targets = model.get_targets(input['labels'], None).view(-1).cpu().numpy()

        return preds, targets, sample['model'].detach().cpu().numpy()
    
    def forward_embs(self, model, sample, reduce=True):
        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        input = self.build_input(sample, self.classification_head_name, self.label_reverse_id)
        embs = model.forward_embed(**input)
        preds = model(**input)
        if isinstance(preds, tuple):
            preds = preds[0]
        targets = model.get_targets(input['labels'], None).view(-1).cpu().numpy()
        return embs.cpu().numpy(), preds, targets, sample['model'].detach().cpu().numpy()
    
    @staticmethod
    def reduce_metrics(logging_outputs):

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "inter_loss" in logging_outputs[0]:
            inter_loss_sum = sum(log.get("inter_loss", 0) for log in logging_outputs)
            metrics.log_scalar("inter_loss", inter_loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "intra_loss" in logging_outputs[0]:
            intra_loss_sum = sum(log.get("intra_loss", 0) for log in logging_outputs)
            metrics.log_scalar("intra_loss", intra_loss_sum / sample_size / math.log(2), sample_size, round=3)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
