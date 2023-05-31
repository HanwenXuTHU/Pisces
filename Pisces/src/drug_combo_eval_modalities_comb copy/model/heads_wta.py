# implement a Winner-Take-All (WTA) pytorch module
# the input is [batch_size, modalities]
# perform WTA on the modalities dimension
import torch
from torch import nn

class Heads_WTA(nn.Module):
    def __init__(self, n_modality, topk=1, is_linear=True, is_mask=True):
        # winner take all
        super(Heads_WTA, self).__init__()
        self.n_modality = n_modality
        if is_linear:
            self.W = nn.Parameter(torch.randn(n_modality, 1))
            # initialize the self.W
            nn.init.xavier_uniform_(self.W)
        self.topk = topk
        self.is_linear = is_linear
        self.is_mask = is_mask
        print("Heads_WTA: topk={}, is_linear={}, is_mask={}".format(topk, is_linear, is_mask))
    
    def forward(self, x, mask=None):
        if self.is_mask:
            x_masked = x.detach().clone()
            x_masked[~mask] = float("-inf")
            indices = torch.topk(x_masked, k=self.topk, dim=-1)[1]
            pred_topk = torch.gather(x, 1, indices)
        else:
            pred_topk, indices = torch.topk(x, k=self.topk, dim=-1)

        if self.is_linear:
            output = torch.zeros_like(x)
            output.scatter_(-1, indices, x.gather(-1, indices))
            soft_W = torch.nn.functional.softmax(self.W, dim=0)
            output = torch.matmul(output, soft_W) + torch.mean(pred_topk, dim=1, keepdim=True)
        else:
            output = torch.mean(pred_topk, dim=1, keepdim=True)
        return output
    


        