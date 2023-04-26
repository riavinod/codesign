import torch
from torch import nn

from genies.utils.encoding import sinusoidal_encoding

from sequence_models.constants import PROTEIN_ALPHABET, PAD

pad_idx = PROTEIN_ALPHABET.index(PAD)

class SingleFeatureNet(nn.Module):

	def __init__(self,
		c_s,
		n_timestep,
		c_pos_emb,
		c_timestep_emb,
		c_aa_emb
	):
		super(SingleFeatureNet, self).__init__()

		self.c_s = c_s
		self.n_timestep = n_timestep
		self.c_pos_emb = c_pos_emb
		self.c_timestep_emb = c_timestep_emb
		self.c_aa_emb = c_aa_emb
		self.linear = nn.Linear(self.c_pos_emb + self.c_timestep_emb + c_aa_emb, self.c_s)
		self.aa_emb = nn.Embedding(len(PROTEIN_ALPHABET), c_aa_emb, padding_idx=pad_idx)

	def forward(self, ts, src, timesteps, mask):
		# s: [b]

		b, max_n_res, device = ts.shape[0], ts.shape[1], timesteps.device

		# [b, n_res, c_pos_emb]
		pos_emb = sinusoidal_encoding(torch.arange(max_n_res).to(device), max_n_res, self.c_pos_emb)
		pos_emb = pos_emb.unsqueeze(0).repeat([b, 1, 1])
		pos_emb = pos_emb * mask.unsqueeze(-1)

		# [b, n_res, c_timestep_emb]
		timestep_emb = sinusoidal_encoding(timesteps.view(b, 1), self.n_timestep, self.c_timestep_emb)
		timestep_emb = timestep_emb.repeat(1, max_n_res, 1)
		timestep_emb = timestep_emb * mask.unsqueeze(-1)

		src_emb = self.aa_emb(src)

		return self.linear(torch.cat([
			pos_emb,
			timestep_emb,
			src_emb
		], dim=-1))