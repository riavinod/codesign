import numpy as np
from torch.utils.data import Dataset

from genies.utils.data_io import load_coord
from sequence_models.utils import parse_fasta, Tokenizer
from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK


pad_idx = PROTEIN_ALPHABET.index(PAD)
mask_idx = PROTEIN_ALPHABET.index(MASK)

class SCOPeDataset(Dataset):
	# Assumption: all domains have at least n_res residues

	def __init__(self, filepaths, fasta_filepath, max_n_res, min_n_res):
		super(SCOPeDataset, self).__init__()
		self.filepaths = filepaths
		self.max_n_res = max_n_res
		self.min_n_res = min_n_res
		seqs, names = parse_fasta(fasta_filepath, return_names=True)
		short_names = [n.split()[0] for n in names]
		self.seq_dict = {n: s for n, s in zip(short_names, seqs)}
		self.tokenizer = Tokenizer(PROTEIN_ALPHABET)

	def __len__(self):
		return len(self.filepaths)

	def __getitem__(self, idx):
		coords = load_coord(self.filepaths[idx])
		n_res = int(len(coords) / 3)
		str_name = self.filepaths[idx].split('/')[-1][:-4]
		seq = self.seq_dict[str_name].upper()
		assert len(seq) == n_res
		tgt = self.tokenizer.tokenize(seq)
		n_corr = np.random.choice(n_res - 1) + 1
		idx_corr = np.random.choice(np.arange(n_res), n_corr)
		src = tgt.copy()
		src[idx_corr] = mask_idx
		if self.max_n_res is not None:
			tgt = np.concatenate([tgt, np.ones(self.max_n_res - n_res) * pad_idx])
			src = np.concatenate([src, np.ones(self.max_n_res - n_res) * pad_idx])
			coords = np.concatenate([coords, np.zeros(((self.max_n_res - n_res) * 3, 3))], axis=0)
			mask = np.concatenate([np.ones(n_res), np.zeros(self.max_n_res - n_res)])
		else:
			assert self.min_n_res is not None
			s_idx = np.random.randint(n_res - self.min_n_res + 1)
			start_idx = s_idx * 3
			end_idx = (s_idx + self.min_n_res) * 3
			tgt = tgt[s_idx: s_idx + self.min_n_res]
			src = src[s_idx: s_idx + self.min_n_res]
			coords = coords[start_idx:end_idx]
			mask = np.ones(self.min_n_res)
		return coords, tgt, src, mask

