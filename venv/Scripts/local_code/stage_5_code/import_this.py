'''
Concrete MethodModule class for a specific learning MethodModule
'''
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/content/ecs189g/venv/Scripts/')
# sys.path.insert(1, 'C:/Users/ataki/Documents/ECS189G_Winter_2025_Source_Code_Template/data')

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from local_code.base_class.dataset import dataset
from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random


class Evaluate_Metrics(evaluate):
    data = None
    metrics = ['Accuracy', 'F1 micro', 'F1 macro', 'F1 weighted', 'Precision micro', 'Precision macro', 'Precision weighted', 'Recall micro', 'Recall macro', 'Recall weighted']
    def evaluate(self):
        print('evaluating performance...')
        return {'Accuracy': accuracy_score(self.data['true_y'], self.data['pred_y']),
                'F1 micro': f1_score(self.data['true_y'], self.data['pred_y'], average='micro'),
                'F1 macro': f1_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                'F1 weighted': f1_score(self.data['true_y'], self.data['pred_y'], average='weighted'),
                'Precision micro': precision_score(self.data['true_y'], self.data['pred_y'], average='micro', zero_division=0.0),
                'Precision macro': precision_score(self.data['true_y'], self.data['pred_y'], average='macro', zero_division=0.0),
                'Precision weighted': precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0.0),
                'Recall micro': recall_score(self.data['true_y'], self.data['pred_y'], average='micro'),
                'Recall macro': recall_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                'Recall weighted': recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        }
        

class Dataset_Loader(dataset):
    
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        nodes = []
        ids = []
        edges = []
        y = []
        nodes_f = open(self.dataset_source_folder_path + self.dataset_source_file_name + "/node", 'r')
        edges_f = open(self.dataset_source_folder_path + self.dataset_source_file_name + "/link", 'r')

        for line in nodes_f:
            line = line.strip('\n')
            elements = [float(i) for i in line.split('\t')]
            nodes.append(elements[1:-1])
            ids.append(elements[0])
            y.append(elements[-1])
        nodes_f.close()

        for line in edges_f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split('\t')]
            edges.append(elements)

        edges_f.close()
        nodes = torch.FloatTensor(nodes)
        edges = torch.LongTensor(edges)
        ids = torch.LongTensor(ids)
        y = torch.nn.functional.one_hot(torch.LongTensor(y), num_classes=3).to(torch.float)
        return {'X': nodes, 'ids': ids,'edges': edges, 'y': y}

class GCN_layer(nn.Module):
        
    def __init__(self, edges, in_features=500, out_features=500, use_bias=False, direction='out', neighbors=False, degrees=None, dropout=0.3, normalizers=None):

        super(GCN_layer, self).__init__()
        self.edges = edges; 
        if not neighbors:
            print('calculate neighbors, in and out')
            self.neighbors = []
            for node_id in range(len(edges)):
                self.neighbors.append(torch.concatenate([self.edges[self.edges[:,0] == node_id,1].flatten(), self.edges[self.edges[:,1] == node_id, 0].flatten()], dim=0))
        else:
            print('manually set neighbors, in and out')
            self.neighbors = neighbors
            self.degrees=degrees
            self.normalizers=normalizers


        # if direction=='out':
        #     self.direction=(0,1)
        
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=use_bias)
                # Define the learnable weight matrix
        # nn.Parameter makes this tensor a part of the model's learnable parameters
        # self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        # self.weight.requires_grad_ = True

        # # Define the optional bias term
        # if use_bias:
        #     self.bias = nn.Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None) # No bias if bias is False
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Sequential(nn.Linear(in_features=emb_num, out_features=hidden_dim, bias=use_bias), 
        #                             #nn.ReLU(), nn.Dropout(0.3), nn.Linear(in_features=hidden_dim, out_features=emb_num, bias=use_bias)
        #                             )
        
        
        # nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.xavier_uniform_(self.linear.bias)


    def forward(self, nodes):
        # out = nodes
        # print(self.edges.shape)
        # print(nodes.shape)
        # node_ids = torch.nonzero(torch.sum(nodes,0).flatten())
        # print(node_ids.shape)

        # for node_id in node_ids:

        #     # Step 1: Linear transformation (H^(l) W^(l))
        #     # This is equivalent to a standard fully connected layer applied row-wise.
        #     neighbors = self.edges[self.edges[:,0]==node_id,1]
        #     emb = [nodes[node_id]]
        #     for n in neighbors:
        #         emb.append(self.nodes[n])

        #     emb = torch.sum(torch.stack(emb), dim=0)
        #     out[n] = emb
        out = torch.zeros(nodes.shape)
        
        # print(nodes.shape)
        # print(out.shape)
        for node_id in range(len(nodes)):
            # print(neighbors)
            # print(nodes[self.neighbors[node_id]].shape)
            # print(self.degrees[node_id]*self.degrees[self.neighbors[node_id]])
            emb = torch.sum(nodes[self.neighbors[node_id]]/self.normalizers[node_id], 0) + nodes[node_id]/self.degrees[node_id]
            # print(emb.shape)
            out[node_id] = emb.flatten()
        
        out = self.linear(out)
        out = self.dropout(self.relu(out))
        return out

def set_up_parameters(obj):
    print('setting up params')
    obj.linear.requires_grad_ = True
    for name, param in obj.linear.named_parameters():
        # print(name)
        # print(param)
        setattr(obj, name, param)
        getattr(obj, name).requires_grad_ = True
        # self.param.requires_grad_ = True
        # nn.init.xavier_uniform_(local)
    obj.requires_grad_ = True

class Method_GCN(method, nn.Module):
    data = None
    ids_train = None; ids_test = None
    # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_function = nn.CrossEntropyLoss()
    # for training accuracy investigation purpose
    metric_evaluator = Evaluate_Metrics('evaluator', '')
    curves = {}
    curves['epochs'] = []
    curves['loss'] = []
    curves['test loss']= []
    curves['test accuracy']= []
    for metric in metric_evaluator.metrics:
        curves[metric] = []
    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, data=data, max_epoch=500, learning_rate = 1e-3, hidden_dim_1=16, hidden_dim_2=64, dropout=0.3, emb_num=500, weight_decay=5e-4, num_classes=3, use_bias=False, update_lr=1, num_layers=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.data = data
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.dropout_rate = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.update_lr = update_lr
        self.edges = self.data['edges']
        self.num_layers = num_layers

        neighbors = []; degrees = []
        for node_id in range(len(self.edges)):
            neighbors.append(torch.concatenate([self.edges[self.edges[:,0] == node_id,1].flatten(), self.edges[self.edges[:,1] == node_id, 0].flatten()], dim=0))
            degrees.append(len(neighbors))
        print(neighbors[1].shape)
        degrees = torch.LongTensor(degrees).reshape((-1,1))
        normalizers = []
        for node_id in range(len(self.edges)):
            normalizers.append((degrees[node_id]*degrees[neighbors[node_id]])**(0.5))
        if num_layers == 2:
            self.gcn_1 = GCN_layer(edges=self.data['edges'], in_features=emb_num, out_features=hidden_dim_1, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers)
            self.gcn_2 = GCN_layer(edges=self.data['edges'], in_features=hidden_dim_1, out_features=num_classes, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers)
            set_up_parameters(self.gcn_1)
            set_up_parameters(self.gcn_2)
        if num_layers == 3:
            self.gcn_1 = GCN_layer(edges=self.data['edges'], in_features=emb_num, out_features=hidden_dim_2, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers)
            self.gcn_2 = GCN_layer(edges=self.data['edges'], in_features=hidden_dim_2, out_features=hidden_dim_1, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers)
            self.gcn_3 = GCN_layer(edges=self.data['edges'], in_features=hidden_dim_1, out_features=num_classes, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers)
            set_up_parameters(self.gcn_1)
            set_up_parameters(self.gcn_2)
            set_up_parameters(self.gcn_3)
        # self.graph = nn.Sequential()
        #     GCN_layer(edges=self.data['edges'], in_features=emb_num, out_features=128, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers), 
        #     GCN_layer(edges=self.data['edges'], in_features=128, out_features=64, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers),
        #     #GCN_layer(edges=self.data['edges'], in_features=526, out_features=emb_num, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers),
        #     #GCN_layer(edges=self.data['edges'], in_features=emb_num, out_features=1024, neighbors=neighbors, degrees=degrees, dropout=dropout, use_bias=use_bias, normalizers=normalizers),
        #     )
        # self.dense = nn.Sequential(nn.Linear(in_features=64, out_features=num_classes), nn.Softmax(dim=0))

        # for child in self.graph.children():
        #     try:
        #         set_up_parameters(child)
        #         print('done setting up')
        #     except:
        #         continue


    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x, out_embs=False):
        '''Forward propagation'''
        # embs = self.graph(x)
        # #print(embs.type())
        # out = self.dense(embs)
        # #print(out.type())
        if self.num_layers == 2:
            embs = self.gcn_1(x)
            out = self.gcn_2(embs)
        elif self.num_layers == 3:
            embs = self.gcn_2(self.gcn_1(x))
            out = self.gcn_3(embs)
        if out_embs:
            return out, embs
        else:
            return out

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, ids_train, ids_test):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.AdamW(self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay)
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # # you can try to split X and y into smaller-sized batches by yourself
        # dataloader = torch.utils.data.DataLoader(torch.utils.data.Dataset(X,y), batch_size=100, shuffle=True)
        # batch_size=1000
        # n_samples = len(X)
        for param in self.parameters():
            if param.requires_grad:
                print("Trainable parameter:", param.shape)
        
        print('normalizing')
        X = torch.nn.functional.normalize(self.data["X"], p=2, dim=1)
        print('splitting dataset')
        X_train = torch.zeros(X.shape)
        X_train[ids_train] = X[ids_train]
        y_train = torch.zeros(self.data["y"].shape, dtype=torch.float)
        y_train[ids_train] = self.data["y"][ids_train]

        # ids_test = self.data["ids"][self.data["ids"] not in ids_train]
        X_test = torch.zeros(X.shape)
        X_test[ids_test] = X[ids_test]
        y_test = torch.zeros(self.data["y"].shape, dtype=torch.float)
        y_test[ids_test] = self.data["y"][ids_test]


        print('entering epochs')
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            if epoch != 0 and epoch%50 == 0 and self.learning_rate>0.001:
                self.learning_rate = self.learning_rate*self.update_lr
                optimizer = torch.optim.AdamW(self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay)
            y_pred = self.forward(X_train)
            # convert y to torch.tensor as well
            y_true = y_train
            # print(y_true.shape)
            # print(y_pred.shape)
            # calculate the training loss
            train_loss = self.loss_function(y_pred[ids_train], y_true[ids_train])
            print('Loss:', train_loss.item())

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            print('backpropagation')
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            print('update')
            optimizer.step()
            #print('Loss:', train_loss.item())

            if epoch%5 == 0:
                # print(self.forward(X_train).argmax(1).shape)
                # print(y_train.argmax(1).shape)
                print('evaluating')
                self.metric_evaluator.data = {'true_y': y_train.argmax(1)[ids_train], 'pred_y': self.forward(X_train).argmax(1)[ids_train]}
                evals = self.metric_evaluator.evaluate()
                print('Epoch:', epoch, end=" ")
                for metric in evals.keys():
                    print(f"{metric}: {evals[metric]:.4f}", end=", ")
                    self.curves[metric].append(evals[metric])
                self.curves['epochs'].append(epoch)
                self.curves['loss'].append(train_loss.item())
                test1, test2 = self.test(X_test, raw=True)
                # print(test2.shape)
                # print(y_test.shape)
                self.curves['test loss'].append(self.loss_function(test2[ids_test], y_test[ids_test]).item())
                self.curves['test accuracy'].append(accuracy_score(test1[ids_test], y_test.argmax(1)[ids_test]))


                
    
    def test(self, X, raw=False, out_embs=False):
        # do the testing, and result the result
        test_X = X
        if out_embs:
            y_pred, embs = self.forward(test_X, embs=True)
            
            if raw:
                return y_pred.argmax(1), embs, y_pred
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        y_pred = self.forward(test_X)
        if raw:
            return y_pred.argmax(1), y_pred
        return y_pred.argmax(1)
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.ids_train, self.ids_test)
        print('--start testing...')
        X_test = torch.zeros(self.data["X"].shape)
        X_test[self.ids_test] = self.data["X"][self.ids_test]
        pred_y = self.test(X_test)
        y_test = torch.zeros(self.data["y"].shape, dtype=torch.float)
        y_test[self.ids_test] = self.data["y"][self.ids_test]
        return {'pred_y': pred_y[self.ids_test], 'true_y': y_test.argmax(1)[self.ids_test], 'curves': self.curves}
            
