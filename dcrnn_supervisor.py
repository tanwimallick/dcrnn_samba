import os
import time

import argparse
import yaml
import sys

from lib.utils import load_graph_data

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import masked_mae_loss


import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.utils as utils
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

# : samba.optim.Adam
def train(dcrnn_model, _data: dict, optimizer, horizon: int, seq_len: int, num_nodes: int, 
        input_dim: int, output_dim: int, epochs: int, steps: list,
        base_lr:float, max_grad_norm: int, lr_decay_ratio: float) -> None: 
    # steps is used in learning rate - will see if need to use it?

    # min_val_loss = float('inf')
    # wait = 0

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                        gamma=lr_decay_ratio)

    # this will fail if model is loaded with a changed batch_size
    num_batches = _data['train_loader'].num_batch
    
    batches_seen = num_batches #* self._epoch_num

    for epoch_num in range(epochs):

        dcrnn_model = dcrnn_model.train()

        train_iterator = _data['train_loader'].get_iterator()
        losses = []

        for _, (x, y) in enumerate(train_iterator):
            optimizer.zero_grad()

            x, y = _prepare_data(x, y, horizon, seq_len, num_nodes, input_dim, output_dim)

            s_x = samba.from_torch(x, name ='x_data', batch_dim = 0).float() 
            s_y = samba.from_torch(y, name ='y_data', batch_dim = 0).float() 
            outputs = samba.session.run(input_tensors=[s_x, s_y],output_tensors= dcrnn_model.output_tensors)
            #output = dcrnn_model(x, y, batches_seen)
            outputs = samba.to_torch(outputs)
            loss = _compute_loss(y, output)
            losses.append(loss.item())

            # batches_seen += 1
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(dcrnn_model.parameters(), max_grad_norm)
            # optimizer.step()
        
        #lr_scheduler.step()
        #val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)
        #sys.stdout.write('\rEpoch: %d, Loss:%f, val_loss:%f' %(epoch_num, np.mean(losses), val_loss))
        sys.stdout.write('\rEpoch: %d, Loss:%f' %(epoch_num, np.mean(losses)))



def addYamlToParser(parser, supervisor_config, section='data'):
    # Remember to use the 'section' input parameter.
    parser.add_argument('--dataset_dir', default=supervisor_config.get('dataset_dir'), type=str)
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

def get_inputs() -> Tuple[samba.SambaTensor, samba.SambaTensor]:
  	#placeholder?
    x = samba.randn(12, 64, 414, name='x_data', batch_dim=0)
    y = samba.randn(12, 64, 207, name='y_data', batch_dim=0)
    return x, y    

def main(args: argparse.Namespace, parser: argparse.ArgumentParser):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        addYamlToParser(parser, supervisor_config)
        args = parser.parse_args()

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

        _data = utils.load_dataset(dataset_dir, batch_size, test_batch_size)
        standard_scaler = _data['scaler']

        dcrnn_model = DCRNNModel(adj_mx) #, self._logger) , **self._model_kwargs)
        dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model

        # optimizer = torch.optim.Adam(dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
        # train(dcrnn_model, _data,  optimizer, horizon, seq_len, num_nodes, input_dim, output_dim, epochs,
        #    steps, base_lr, max_grad_norm, lr_decay_ratio)

        optimizer = sambaflow.samba.optim.Adam(dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
        
        args = parse_app_args(argv=args)
        samba.from_torch_(dcrnn_model)
        inputs = get_inputs()
        common_app_driver(args, dcrnn_model, inputs, optimizer, name='dcrnn', app_dir=utils.get_file_dir(__file__))
        
        if args.command == "test":
            utils.trace_graph(dcrnn_model, inputs, optimizer, config_dict=vars(args))
            outputs = dcrnn_model.output_tensors
        elif args.command == "run":
            print (args)
            utils.trace_graph(dcrnn_model, inputs, optimizer, pef='/tanwi/out/dcrnn/dcrnn.pef')   
            train(dcrnn_model, _data,  optimizer, horizon, seq_len, num_nodes, input_dim, output_dim, epochs,
                  steps, base_lr, max_grad_norm, lr_decay_ratio)



        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args, parser)








'''
def evaluate(self, dataset='val', batches_seen=0):
    """
    Computes mean L1Loss
    :return: mean L1Loss
    """
    with torch.no_grad():
        self.dcrnn_model = self.dcrnn_model.eval()

        val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
        losses = []

        y_truths = []
        y_preds = []

        for _, (x, y) in enumerate(val_iterator):
            x, y = self._prepare_data(x, y)

            output = self.dcrnn_model(x)
            loss = self._compute_loss(y, output)
            losses.append(loss.item())

            y_truths.append(y.cpu())
            y_preds.append(output.cpu())

        mean_loss = np.mean(losses)

        self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

        y_preds = np.concatenate(y_preds, axis=1)
        y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

        y_truths_scaled = []
        y_preds_scaled = []
        for t in range(y_preds.shape[0]):
            y_truth = self.standard_scaler.inverse_transform(y_truths[t])
            y_pred = self.standard_scaler.inverse_transform(y_preds[t])
            y_truths_scaled.append(y_truth)
            y_preds_scaled.append(y_pred)

        return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}
'''


'''
class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self.base_dir =  kwargs.get('base_dir')
        self.batch_size = kwargs.get('batch_size')
        self.dataset_dir = kwargs.get('dataset_dir')
        self.test_batch_size = kwargs.get('test_batch_size')
        self.val_batch_size = kwargs.get('val_batch_size')
        self.graph_pkl_filename = kwargs.get('graph_pkl_filename')

        self.cl_decay_steps = kwargs.get('cl_decay_steps')
        self.filter_type = kwargs.get('filter_type')
        self.horizon = kwargs.get('horizon')
        self.input_dim = kwargs.get('input_dim')
        self.l1_decay = kwargs.get('l1_decay')
        self.max_diffusion_step = kwargs.get('max_diffusion_step')
        self.num_nodes = kwargs.get('num_nodes')
        self.num_rnn_layers = kwargs.get('num_rnn_layers')
        self.output_dim = kwargs.get('output_dim')
        self.rnn_units = kwargs.get('rnn_units')
        self.seq_len = kwargs.get('seq_len')
        self.use_curriculum_learning = kwargs.get('use_curriculum_learning')

        self.base_lr = kwargs.get('base_lr')
        self.dropout = kwargs.get('dropout')
        self._epoch_num = kwargs.get('epoch')
        self.epochs = kwargs.get('epochs')
        self.max_grad_norm = kwargs.get('max_grad_norm')
        self.steps = kwargs.get('steps')

        self._log_dir = self.base_dir
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
 

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break
'''
