import os
import time

import argparse
import yaml
import sys

from lib.utils import load_graph_data
from typing import Tuple
import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import masked_mae_loss


import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.utils as samba_utils
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver
#from sambaflow.samba.utils.dataset.mnist import dataset_transform
from sambaflow.samba.utils.trainer.samba import train as samba_train
#from sambaflow.samba.utils.trainer.torch import train as torch_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random
SEED = 3001
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(0)
#torch.use_deterministic_algorithms(True)
#tf.random.set_seed(SEED)


def _prepare_data(x, y, horizon, seq_len, num_nodes, input_dim, output_dim):
        x, y = _get_x_y(x, y)
        x, y = _get_x_y_in_correct_dims(x, y, horizon, seq_len, num_nodes, input_dim, output_dim)
        return x.to(device), y.to(device)

def _get_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                y shape (horizon, batch_size, num_sensor, input_dim)
    """
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    # self._logger.debug("X: {}".format(x.size()))
    # self._logger.debug("y: {}".format(y.size()))
    x = x.permute(1, 0, 2, 3)
    y = y.permute(1, 0, 2, 3)
    return x, y
def _get_x_y_in_correct_dims(x, y, horizon, seq_len, num_nodes, input_dim, output_dim):
    """
    :param x: shape (seq_len, batch_size, num_sensor, input_dim)
    :param y: shape (horizon, batch_size, num_sensor, input_dim)
    :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    batch_size = x.size(1)
    x = x.view(seq_len, batch_size, num_nodes * input_dim)
    y = y[..., :output_dim].view(horizon, batch_size,
                                        num_nodes * output_dim)
    return x, y


def _compute_loss(y_true, y_predicted):
    loss = torch.nn.L1Loss()
    # y_true = self.standard_scaler.inverse_transform(y_true)
    # y_predicted = self.standard_scaler.inverse_transform(y_predicted)
    return loss(y_predicted, y_true)


def train(dcrnn_model: nn.Module, _data: dict, optimizer: samba.optim.SGD, horizon: int, seq_len: int, num_nodes: int,
        input_dim: int, output_dim: int, epochs: int, steps: list,
        base_lr:float, max_grad_norm: int, lr_decay_ratio: float) -> None:

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                        gamma=lr_decay_ratio)

    # this will fail if model is loaded with a changed batch_size
    num_batches = _data['train_loader'].num_batc
    batches_seen = num_batches #* self._epoch_num

    for epoch_num in range(epochs):

        dcrnn_model = dcrnn_model.train()

        train_iterator = _data['train_loader'].get_iterator()
        losses = []

        for _, (x, y) in enumerate(train_iterator):
            optimizer.zero_grad()

            batch_size = 64
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            x = x.view(batch_size, seq_len,num_nodes * input_dim)
            y = y[..., :output_dim].view(batch_size,horizon, 
                                                num_nodes * output_dim)

            #x, y = _prepare_data(x, y, horizon, seq_len, num_nodes, input_dim, output_dim)

            s_x = samba.from_torch(x, name ='x_data', batch_dim = 0).float()
            s_y = samba.from_torch(y, name ='y_data', batch_dim = 0).float()
            outputs = samba.session.run(input_tensors=[s_x, s_y],output_tensors= dcrnn_model.output_tensors)
            #output = dcrnn_model(x, y, batches_seen)
            outputs = samba.to_torch(outputs)
            loss = _compute_loss(y, output)
            losses.append(loss.item())
            
        sys.stdout.write('\rEpoch: %d, Loss:%f' %(epoch_num, np.mean(losses)))
    
def addYamlToParser(parser: argparse.ArgumentParser, section='data'):
    with open('data/model/dcrnn_la_new.yaml') as f:
        supervisor_config = yaml.safe_load(f)
        # Remember to use the 'section' input parameter.
        parser.add_argument('--batch_size', default=supervisor_config.get('batch_size'), type=int)
        parser.add_argument('--val_batch_size', default=supervisor_config.get('val_batch_size'), type=int)
        parser.add_argument('--test_batch_size', default=supervisor_config.get('test_batch_size'), type=int)
        parser.add_argument('--base_lr', default=supervisor_config.get('base_lr'), type=float)
        parser.add_argument('--horizon', default=supervisor_config.get('horizon'), type=int)
        parser.add_argument('--seq_len', default=supervisor_config.get('seq_len'), type=int)
        parser.add_argument('--num_nodes', default=supervisor_config.get('num_nodes'), type=int)
        parser.add_argument('--output_dim', default=supervisor_config.get('output_dim'), type=int)
        parser.add_argument('--input_dim', default=supervisor_config.get('input_dim'), type=int)
        parser.add_argument('--epsilon', default=supervisor_config.get('epsilon'), type=float)
        parser.add_argument('--epochs', default=supervisor_config.get('epochs'), type=int)
        parser.add_argument('--steps', default=supervisor_config.get('steps'), type=list)
        parser.add_argument('--max_grad_norm', default=supervisor_config.get('max_grad_norm'), type=int)
        parser.add_argument('--lr_decay_ratio', default=supervisor_config.get('lr_decay_ratio'), type=float)
        parser.add_argument('--dataset_dir', default=supervisor_config.get('dataset_dir'), type=str)
        parser.add_argument('--max_diffusion_step', default=supervisor_config.get('max_diffusion_step'), type=int)
        parser.add_argument('--cl_decay_steps', default=supervisor_config.get('cl_decay_steps'), type=int)
        parser.add_argument('--filter_type', default=supervisor_config.get('filter_type'), type=str)
        parser.add_argument('--num_rnn_layers', default=supervisor_config.get('num_rnn_layers'), type=int)
        parser.add_argument('--rnn_units', default=supervisor_config.get('rnn_units'), type=str)



def add_run_args(parser: argparse.ArgumentParser):
    with open('data/model/dcrnn_la_new.yaml') as f:
        supervisor_config = yaml.safe_load(f)
        parser.add_argument('--dataset_dir', default=supervisor_config.get('dataset_dir'), type=str)


def get_inputs() -> Tuple[samba.SambaTensor, samba.SambaTensor]:
        #placeholder?
    x = samba.randn( 64, 12, 20, name='x_data', batch_dim=0)
    y = samba.randn( 64, 12, 10, name='y_data', batch_dim=0)
    return x, y

def main(argv):
    with open('data/model/dcrnn_la_new.yaml') as f:
        supervisor_config = yaml.safe_load(f)

        args = parse_app_args(argv=argv, common_parser_fn=addYamlToParser, run_parser_fn=add_run_args)

        graph_pkl_filename = supervisor_config.get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        dataset_dir = args.dataset_dir
        batch_size = args.batch_size
        val_batch_size = args.val_batch_size
        test_batch_size = args.test_batch_size
        base_lr = args.base_lr
        horizon = args.horizon
        seq_len = args.seq_len
        num_nodes = args.num_nodes
        output_dim = args.output_dim
        input_dim = args.input_dim
        epsilon = args.epsilon
        epochs =  args.epochs
        steps = args.steps
        base_lr = args.base_lr
        max_grad_norm = args.max_grad_norm
        lr_decay_ratio = args.lr_decay_ratio
        max_diffusion_step = args.max_diffusion_step
        cl_decay_steps = args.cl_decay_steps
        filter_type = args.filter_type
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units

        _data = utils.load_dataset(dataset_dir, batch_size, test_batch_size)
        standard_scaler = _data['scaler']

        dcrnn_model = DCRNNModel(adj_mx, max_diffusion_step, cl_decay_steps, filter_type, num_nodes,
                 num_rnn_layers, rnn_units, output_dim, horizon, input_dim, seq_len) 
        dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model

        # optimizer = sambaflow.samba.optim.SGD(dcrnn_model.parameters(), lr=base_lr)#, eps=epsilon)
        optimizer = samba.optim.AdamW(model.parameters(), lr=base_lr, eps= epsilon)


        samba.from_torch_(dcrnn_model)
        inputs = get_inputs()
        common_app_driver(args, dcrnn_model, inputs, optimizer, name='dcrnn', app_dir=samba_utils.get_file_dir(__file__))

        if args.command == "test":
            samba_utils.trace_graph(dcrnn_model, inputs, optimizer, config_dict=vars(args))
            outputs = dcrnn_model.output_tensors
        elif args.command == "run":
            print (args)
            samba_utils.trace_graph(dcrnn_model, inputs, optimizer, pef='pef/dcrnn/dcrnn.pef')
            train(dcrnn_model, _data,  optimizer, horizon, seq_len, num_nodes, input_dim, output_dim, epochs,
                  steps, base_lr, max_grad_norm, lr_decay_ratio)

if __name__ == '__main__':
    main(sys.argv[1:])

