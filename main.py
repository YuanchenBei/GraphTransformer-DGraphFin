from utils.dgraphfin import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from modules.gnn import SAGE,GraphTransformer

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T

import numpy as np
from torch_geometric.data import Data
import os
import argparse
import random

parser = argparse.ArgumentParser(description='gnn4DgraphFin')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='DGraph')
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--model', type=str, default='transformer')
parser.add_argument('--use_embeddings', action='store_true')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--heads', type=list, default=[3,1])
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=40)
parser.add_argument('--dropout', type=float, default=0.3) #0.1
parser.add_argument('--l2', type=float, default=1e-5) #1e-7


args = parser.parse_args()
print(args, flush=True)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#设置gpu设备
device = args.gpu_id
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(device, flush=True)

path='./datasets/' #数据保存路径
save_dir='./results/' #模型保存路径
dataset_name=args.dataset
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图


if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

data = data.to(device)
train_idx = split_idx['train']
# result_dir = prepare_folder(dataset_name, args.model)

print(data.x.shape, flush=True)  #feature
print(data.y.shape, flush=True)  #label

gnn_parameters = {'lr':args.lr
              , 'num_layers':args.num_layers
              , 'hidden_channels':args.hidden_dim
              , 'dropout':args.dropout
              , 'batchnorm': False
              , 'l2':args.l2
              , 'layer_heads': args.heads
             }


def train(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()
    
    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)
        
    y_pred = out.exp()  # (N,num_classes)
    
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
            
    return eval_results, losses, y_pred

valid_auc_runs = []
test_auc_runs = []

for run in range(0, args.runs):
    set_seed(run)
    para_dict = gnn_parameters
    model_para = gnn_parameters.copy()
    model_para.pop('lr')
    model_para.pop('l2')        
    model = GraphTransformer(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)

    print(f'gnn initialized', flush=True)

    eval_metric = 'auc'  #使用AUC衡量指标
    evaluator = Evaluator(eval_metric)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    best_valid = 0.0
    best_test = 0.0
    min_valid_loss = 1e8
    best_out = None
    log_steps =1 # log记录周期
    no_conv = False

    print("####################", flush=True)
    print("model parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    print("####################", flush=True)

    for epoch in range(0, args.epochs):
        loss = train(model, data, train_idx, optimizer, no_conv)
        eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)
        train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_out = out.cpu()
        
        if valid_eval > best_valid:
            best_valid = valid_eval
        
        if test_eval > best_test:
            best_test = test_eval

        if epoch % log_steps == 0:
            print(f'Run: {run+1:02d}, '
                    f'Epoch: {epoch+1:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_eval:.3f}%, '
                    f'Valid: {100 * valid_eval:.3f}% '
                    f'Test: {100 * test_eval:.3f}%', flush=True)
    
    print(f"Run {run+1}: best valid AUC: {best_valid}, best test AUC: {best_test}", flush=True)
    valid_auc_runs.append(best_valid)
    test_auc_runs.append(best_test)

print(f"Best valid auc: {np.mean(valid_auc_runs)}±{np.std(valid_auc_runs)}", flush=True)
print(f"Best test auc: {np.mean(test_auc_runs)}±{np.std(test_auc_runs)}", flush=True)
