import os
import glob
from torch.utils.data import DataLoader

from genies.data.dataset import SCOPeDataset
from genies.utils.data_io import load_filepaths


class SCOPeDataModule(object):

	def __init__(self, name, log_dir, data_dir, max_n_res, min_n_res, dataset_names, dataset_size, dataset_classes, batch_size):

		self.name = name
		self.log_dir = log_dir
		self.data_dir = data_dir
		self.max_n_res = max_n_res
		self.min_n_res = min_n_res
		self.dataset_names = dataset_names
		self.dataset_size = dataset_size
		self.dataset_classes = dataset_classes
		self.batch_size = batch_size
		self.setup()

	def setup(self, stage=None):

		# load filepaths
		dataset_filepath = os.path.join(self.log_dir, self.name, 'dataset.txt')
		if os.path.exists(dataset_filepath):
			with open(dataset_filepath) as file:
				filepaths = [line.strip() for line in file]
		else:
			filepaths = load_filepaths(self.data_dir, self.dataset_names, self.max_n_res, self.min_n_res, self.dataset_classes, self.dataset_size)
			if not os.path.exists(os.path.join(self.log_dir, self.name)):
				os.makedirs(os.path.join(self.log_dir, self.name))
			with open(dataset_filepath, 'w') as file:
				for filepath in filepaths:
					file.write(filepath + '\n')

		# create dataset
		fasta_fpath = os.path.join(self.data_dir, 'astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa')
		self.dataset = SCOPeDataset(filepaths, fasta_fpath, self.max_n_res, self.min_n_res)
		print(f'Number of samples: {len(filepaths)}')

	def train_dataloader(self):
		return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
