import torch
import torch.nn.functional as F

from sequence_models.constants import PROTEIN_ALPHABET, MASK

mask_idx = PROTEIN_ALPHABET.index(MASK)

def rmsd(x_pred, x, mask, eps=1e-10):
    rmsds = (eps + torch.sum((x_pred - x) ** 2, dim=-1)) ** 0.5
    return torch.sum(rmsds * mask) / torch.sum(mask)


def mask_ce(pred, tgt, src):
    mask = src == mask_idx
    n = mask.sum()
    p = torch.masked_select(pred, mask.unsqueeze(-1)).view(n, -1)
    t = torch.masked_select(tgt, mask)
    loss = F.cross_entropy(p, t, reduction='mean')
    return loss, n