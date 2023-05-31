import imp
import os
import pickle
import logging
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (BaseFairseqModel, register_model, register_model_architecture)
import torch
from torch import nn, Tensor
from fairseq.models.roberta import RobertaEncoder
from omegaconf import II
from numpy.random import uniform
from fairseq.models.gnn import DeeperGCN
from .heads_classify import HeadsClassify


logger = logging.getLogger(__name__)

data_name_dict = {'ctrp': 'Drug target (CTRPv2)',
                  'qm9': '3D (QM9)',
                  'sider': 'Side effect',
                  'nci60': 'Drug Sensitivity (NCI60)',
                  'text': 'Text',
                  'dron': 'Drug Ontology',
                  'SMILES': 'SMILES'}

@dataclass
class MMModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(default='gelu', )
    dropout: float = field(default=0.1)
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)
    relu_dropout: float = field(default=0.0)
    encoder_embed_path: Optional[str] = field(default=None)
    encoder_embed_dim: int = field(default=768)
    encoder_ffn_embed_dim: int = field(default=3072)
    encoder_layers: int = field(default=12)
    encoder_attention_heads: int = field(default=12)
    encoder_normalize_before: bool = field(default=False)
    encoder_learned_pos: bool = field(default=True)
    layernorm_embedding: bool = field(default=True)
    no_scale_embedding: bool = field(default=True)
    max_positions: int = field(default=512)

    mlp_hidden_size: int = field(default=1024)

    use_dropnet: float = field(default=0.)
    datatype: str = field(default='tg')
    ft_grad_scale: bool = field(default=False)

    gnn_number_layer: int = field(default=12)
    gnn_dropout: float = field(default=0.1)
    conv_encode_edge: bool = field(default=True)
    gnn_embed_dim: int = field(default=384)
    gnn_aggr: str = field(default='maxminmean')
    gnn_norm: str = field(default='batch')
    gnn_activation_fn: str = field(default='relu')

    used_modal: str = field(default='SMILES,Graph,3D,Side effect,Drug Sensitivity (NCI60),Text,Drug Ontology,Drug target')
    classification_head_name: str = field(default='')
    load_checkpoint_heads: bool = field(default=False)

    n_memory: int = field(default=32)
    top_k: int = field(default=2)

    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={"help": "scalar quantization noise and scalar quantization at training time"},
    )
    # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
    spectral_norm_classification_head: bool = field(default=False)

    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(default=0.0,
                                     metadata={"help": "LayerDrop probability for decoder"})
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={"help": "which layers to *keep* when pruning as a comma-separated list"},
    )
    max_source_positions: int = II("model.max_positions")
    no_token_positional_embeddings: bool = field(default=False)
    pooler_activation_fn: str = field(default='tanh')
    pooler_dropout: float = field(default=0.0)
    untie_weights_roberta: bool = field(default=False)
    adaptive_input: bool = field(default=False)
    skip_update_state_dict: bool = field(default=False,
        metadata={"help": "Don't update state dict when load pretrained model weight"},
    )

    drug_dict_path: str = field(default='')
    raw_data_path: str = field(default='')
    drug_target_path: str = field(default='')

    mix: bool = field(default=False)

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

@register_model("pisces_multi_modalities", dataclass=MMModelConfig)
class Pisces_MM_Model(BaseFairseqModel):

    def __init__(self, args, encoder, dual_view_encoder):
        super().__init__()
        self.args = args
        self.skip_update_state_dict = args.skip_update_state_dict
        self.encoder = encoder
        self.dual_view_encoder = dual_view_encoder
        self.datatype = args.datatype
        self.ft_grad_scale = args.ft_grad_scale
        self.classification_heads = nn.ModuleDict()
        self.drug_dict = self.load_drug_name_dict()

    @classmethod
    def build_model(cls, args, task):
        
        base_architecture(args)

        encoder = None
        dual_view_encoder = None

        if args.datatype == 'tg':
            encoder = TrEncoder(args, task.source_dictionary)
            dual_view_encoder = DeeperGCN(args)
        elif args.datatype == 'tt':
            encoder = TrEncoder(args, task.source_dictionary)
        elif args.datatype == 'gg':
            encoder = DeeperGCN(args)
        else:
            raise NotImplementedError('No Implemented by DDI')
        
        return cls(args, encoder, dual_view_encoder)

    def load_drug_name_dict(self):
        drug_name_dict_path = os.path.join(self.args.drug_dict_path)
        drug_name_dict = {}
        with open(drug_name_dict_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                d_name = ' '.join(line[0: -1])
                drug_name_dict[d_name] = eval(line[-1])
        return drug_name_dict

    # def a function to calculate the contrastive learning loss for multi-modalities
    def get_cosine_loss(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0].detach()
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        loss = self.mmodalities_loss(logits, targets)
        return loss

    def forward(self,
                drug_a_seq,
                drug_b_seq,
                drug_a_graph,
                drug_b_graph,
                net_rel,
                pair, 
                features_only=False,
                classification_head_name=None,
                labels=None,
                **kwargs):
        
        if classification_head_name is not None:
            features_only = True
        
        if self.datatype == 'gg':
            seq_enc_a, _ = self.encoder(**drug_a_graph, features_only=features_only, **kwargs)
            seq_enc_b, _ = self.encoder(**drug_b_graph, features_only=features_only, **kwargs)
        else:
            seq_enc_a, _ = self.encoder(**drug_a_seq, features_only=features_only, **kwargs)
            seq_enc_b, _ = self.encoder(**drug_b_seq, features_only=features_only, **kwargs)

        if self.dual_view_encoder is not None:
            
            graph_enc_a, _ = self.dual_view_encoder(**drug_a_graph, features_only=features_only, **kwargs)
            graph_enc_b, _ = self.dual_view_encoder(**drug_b_graph, features_only=features_only, **kwargs)
            
            seq_enc_a = self.get_cls(seq_enc_a)
            seq_enc_b = self.get_cls(seq_enc_b)
            
            graph_enc_a = self.get_cls(graph_enc_a)
            graph_enc_b = self.get_cls(graph_enc_b)
            
            if self.ft_grad_scale:
                if isinstance(seq_enc_a, Tensor):
                    seq_enc_a = GradMultiply.apply(seq_enc_a, 0.1)
                if isinstance(seq_enc_b, Tensor):
                    seq_enc_b = GradMultiply.apply(seq_enc_b, 0.1)
                if isinstance(graph_enc_a, Tensor):
                    graph_enc_a = GradMultiply.apply(graph_enc_a, 0.1)
                if isinstance(graph_enc_b, Tensor):
                    graph_enc_b = GradMultiply.apply(graph_enc_b, 0.1)

            x = self.classification_heads[classification_head_name](seq_enc_a, graph_enc_a, \
                            seq_enc_b, graph_enc_b, net_rel, pair, labels)

        else:
        
            seq_enc_a = self.get_cls(seq_enc_a)
            seq_enc_b = self.get_cls(seq_enc_b)
            
            if isinstance(seq_enc_a, Tensor):
               seq_enc_a = GradMultiply.apply(seq_enc_a, 0.1)
            if isinstance(seq_enc_b, Tensor):
               seq_enc_b = GradMultiply.apply(seq_enc_b, 0.1)

            x = self.classification_heads[classification_head_name](seq_enc_a, seq_enc_b, net_rel, labels)
        
        return x


    def forward_embed(self,
                drug_a_seq,
                drug_b_seq,
                drug_a_graph,
                drug_b_graph,
                net_rel,
                features_only=False,
                classification_head_name=None,
                **kwargs):
        
        raise NotImplementedError()

    def get_cls(self, x):
        if x is None:
            return 0
        if isinstance(x, torch.Tensor):
            return x[:, -1, :]
        elif isinstance(x, tuple):
            return x[0]
        else:
            raise ValueError()

    def get_targets(self, target, input):
        return target

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning('re-registering head "{}" with num_classes {} (prev: {}) '
                               "and inner_dim {} (prev: {})".format(name, num_classes,
                                                                    prev_num_classes, inner_dim,
                                                                    prev_inner_dim))
        
        if name == 'heads_classify':
                self.classification_heads[name] = HeadsClassify(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                dv_input_dim=getattr(self.dual_view_encoder, "output_features", self.args.gnn_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                n_memory=self.args.n_memory,
                drug_dict =self.drug_dict,
                topk=self.args.top_k,
                raw_data_path=self.args.raw_data_path,
                drug_target_path=self.args.drug_target_path,
                mix=self.args.mix,
            )
        else:
            raise NotImplementedError('No Implemented by Pisces')

    def upgrade_state_dict_named(self, state_dict, name):
        if self.skip_update_state_dict:
            return
        
        prefix = name + '.' if name != "" else ""
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = ([] if not hasattr(self, "classification_heads") else
                              self.classification_heads.keys())
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[prefix + "classification_heads." + head_name +
                                     ".rel_emb.weight"].size(0)
            inner_dim = state_dict[prefix + "classification_heads." + head_name +
                                   ".rel_emb.weight"].size(1)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)

            if 'unk' in k:
                keys_to_delete.append(k)

        # replace batch norm to layer norm
        for k in state_dict.keys():
            if ('norms' in k and ('num_batch' in k or 'running' in k)) or ('norm' in k and ('num_batch' in k or 'running' in k)):
                keys_to_delete.append(k)

        keys_to_delete = list(set(keys_to_delete))

        for k in keys_to_delete:
            del state_dict[k]

        state_prot_emb = state_dict['classification_heads.heads_classify.protein_embedding.weight']
        prot_num = self.classification_heads.heads_classify.protein_num
        replace_emb = state_prot_emb[:prot_num]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v
        
        # deal with the protein embeddings
        new_prot_emb = self.classification_heads.heads_classify.protein_embedding.weight
        new_prot_emb.data[:prot_num] = replace_emb
        new_prot_emb.requires_grad = True
        state_dict['classification_heads.heads_classify.protein_embedding.weight'] = new_prot_emb.detach()
        # calculate the number of parameters in state_dict
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_s = sum(state_dict[p].numel() for p in state_dict.keys())
        
    def max_positions(self):
        return self.args.max_positions

class TrEncoder(RobertaEncoder):
    def __init(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        features, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(features, masked_tokens=masked_tokens)
        else:
            x = None
        return features, x


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@register_model_architecture("pisces_multi_modalities", "pisces_large")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.gnn_number_layer = getattr(args, "gnn_number_layer", 12)
    args.gnn_embed_dim = getattr(args, "gnn_embed_dim", 384)

@register_model_architecture("pisces_multi_modalities", "pisces_base2")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.gnn_number_layer = getattr(args, "gnn_number_layer", 12)
    args.gnn_embed_dim = getattr(args, "gnn_embed_dim", 384)

@register_model_architecture("pisces_multi_modalities", "pisces_base")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 384)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.gnn_number_layer = getattr(args, "gnn_number_layer", 12)
    args.gnn_embed_dim = getattr(args, "gnn_embed_dim", 384)

@register_model_architecture("pisces_multi_modalities", "pisces_small")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 384)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.gnn_number_layer = getattr(args, "gnn_number_layer", 6)
    args.gnn_embed_dim = getattr(args, "gnn_embed_dim", 384)

@register_model_architecture("pisces_multi_modalities", "pisces_xsmall")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 384)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)

    args.gnn_number_layer = getattr(args, "gnn_number_layer", 3)
    args.gnn_embed_dim = getattr(args, "gnn_embed_dim", 384)
