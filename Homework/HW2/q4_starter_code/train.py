
import argparse
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import models
import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')

    parser.set_defaults(model_type='GCN',
                        dataset='cora',
                        num_layers=2,
                        batch_size=32,
                        hidden_dim=32,
                        dropout=0.0,
                        epochs=200,
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()

def train(dataset, task, args):
    if task == 'graph':
        # graph classification: separate dataloader for test set
        data_size = len(dataset)
        loader = DataLoader(
                dataset[:int(data_size * 0.8)], batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
                dataset[int(data_size * 0.8):], batch_size=args.batch_size, shuffle=True)
    elif task == 'node':
        # use mask to split train/validation/test
        test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise RuntimeError('Unknown task')

    # build model
    model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args, task=task)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    # train
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        print(total_loss)

        if epoch % 10 == 0:
            test_acc = test(loader, model)
            print(test_acc,   '  test')

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total

def main():
    args = arg_parse()

    if args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        task = 'node'
    train(dataset, task, args) 

if __name__ == '__main__':
    main()

