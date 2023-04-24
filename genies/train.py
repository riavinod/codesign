import argparse
from datetime import datetime, timedelta
import os
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.data import DistributedSampler, DataLoader

from genies.config import Config
from genies.data.data_module import SCOPeDataModule
from genies.diffusion.genie import Genies


def main(args):

	# configuration
	# config = Config('/home/kevyan/src/genies/example_configuration')
	args.world_size = args.gpus * args.nodes
	if args.aml:
		pass
	else:
		os.environ['MASTER_ADDR'] = 'localhost'
		os.environ['MASTER_PORT'] = '8885'
	mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(gpu, args):
	# data module
	config = Config(filename=args.config)
	seed = config.training['seed']
	_ = torch.manual_seed(23)
	if args.aml:
		args.nr = int(os.environ['OMPI_COMM_WORLD_RANK'])
	rank = args.nr * args.gpus + gpu
	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=args.world_size,
		rank=rank)
	torch.cuda.set_device(gpu + args.offset)
	device = torch.device('cuda:' + str(gpu + args.offset))
	# TODO: move data and update config
	# TODO: get sequences
	dm = SCOPeDataModule(**config.io, batch_size=config.training['batch_size'])
	sampler = DistributedSampler(dm.dataset, num_replicas=args.world_size, rank=rank, shuffle=True, seed=0)
	dl = DataLoader(dm.dataset, batch_size=config.training['batch_size'], sampler=sampler, num_workers=1)
	# model
	model = Genies(config)
	n_parameters = sum(p.numel() for p in model.parameters())
	if rank == 0:
		print('%d model parameters' % n_parameters)
		out_dir = os.path.join(config.io['log_dir'], config.io['name']) + '/'
		config_as_dict = {'io': config.io, 'training': config.training, 'diffusion': config.diffusion,
			'model': config.model, 'optimization': config.optimization}
		with open(out_dir + 'config.json', 'w') as f:
			json.dump(config_as_dict, f)
	epochs = config.training['n_epoch']
	optimizer = model.configure_optimizers()
	model.to(device)
	scaler = GradScaler()
	model = DDP(model)
	for epoch in range(epochs):
		start_time = datetime.now()
		rloss = 0
		sampler.set_epoch(epoch)
		for i, batch in enumerate(dl):
			optimizer.zero_grad()
			with torch.cuda.amp.autocast(enabled=args.amp):
				batch = [b.to(device) for b in batch]
				loss = model.module.training_step(batch)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
			rloss += loss.detach().cpu() / args.world_size
			if rank == 0:
				print('\rEpoch %d of %d Step %d of %d loss = %.4f'
					  % (epoch + 1, epochs, i + 1, len(dl), rloss / (i + 1)),
					  end='')
		if rank == 0:
			print()
			print('Training complete in ' + str(datetime.now() - start_time))
			with open(out_dir + 'metrics.csv', 'a') as f:
				f.write(','.join([str(epoch), str(rloss) / (i + 1)]))
				f.write('\n')
			if (epoch + 1) % config.training['checkpoint_every_n_epoch'] == 0:
				ckpt_fpath = out_dir + 'checkpoint%d.tar' % epoch + 1
				torch.save({
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					# 'scheduler_state_dict': scheduler.state_dict(),
					'epoch': epoch,
				}, ckpt_fpath)


	# warmup
	# sequences

if __name__ == '__main__':

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
	parser.add_argument('-g', '--gpus', default=1, type=int,
						help='number of gpus per node')
	parser.add_argument('-nr', '--nr', default=0, type=int,
						help='ranking within the nodes')
	parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
	parser.add_argument('-sd', '--state_dict', default=None)
	parser.add_argument('--aml', action='store_true')  # Set true to do multi-node training on amlk8s
	parser.add_argument('-off', '--offset', default=0, type=int,
						help='Number of GPU devices to skip.')
	parser.add_argument('--amp', action='store_true')
	args = parser.parse_args()

	# run
	main(args)