#%%

import torch
import torchvision
import flwr as fl
from typing import List, OrderedDict
import numpy as np
import pickle

class dp_model():
    '''
    A distributed processing model running inside a WSN node
    Uses mobilenetv2 on CIFAR10 by default
    '''
    def __init__(self, model, optimizer=None, criterion=None, scheduler=None, device='cuda', learning_epochs=150, name='default_name', tb_writer = None, trainloader = None, testloader = None, testset_length = None):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.learning_epochs = learning_epochs
        self.name = name
        self.writer = tb_writer

        self.trainloader = trainloader
        self.testloader = testloader

        self.epoch = 0
        self.global_epoch = 0

        self.device = device

    def load_params(self,path):
        print(f'Loading parameters from {path}...')
        self.model.load_state_dict(torch.load(path))

    def sup_train(self):
        print(f'Starting training using device {self.device}...')
        while self.epoch < self.learning_epochs:
            model = self.model
            model.train()
            epochscore = 0
            runloss = 0
            for batch in self.trainloader:
                inputs, labels = batch["img"], batch["label"]

                model.train()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs,labels)
                runloss += loss
                
                _, preds = torch.max(outputs,1)
                epochscore += torch.sum(preds == labels.data)

                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                _,accuracy = self.test(self.model)
                trainacc = epochscore/len(self.trainloader.dataset)
            print(f'[Node:{self.name}]\t epoch:{self.global_epoch}/{self.epoch}:\ttestacc:{accuracy}\ttrainacc:{trainacc}\tloss:{runloss}')

            if self.writer is not None:
                self.writer.add_scalar(f'data/node_{self.name}/loss',runloss,self.global_epoch)
                self.writer.add_scalar(f'data/node_{self.name}/testacc',accuracy,self.global_epoch)
                self.writer.add_scalar(f'data/node_{self.name}/trainacc',trainacc,self.global_epoch)
                              
            if self.scheduler is not None:
                self.scheduler.step()

            self.epoch+=1
            self.global_epoch+=1

    def test(self,num_batches = None):       
        ''' Tests the accuracy of the dp model on testset ''' 
        model = self.model.to(self.device)
        model.eval()
        if num_batches is None:
            num_batches = len(self.testloader)
        loss = 0
        total_correct=0
        for batch in self.testloader:
            inputs, labels = batch["img"], batch["label"]
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            loss += self.criterion(outputs, labels).item()
            _, preds = torch.max(outputs,1)
            total_correct += torch.sum(preds == labels.data)
        accuracy = total_correct/len(self.testloader.dataset)
        return loss,accuracy

class dl_node(fl.client.NumPyClient):
    '''
    Simulation object representing a distributed learning node

    Implements a mobilenetv2 on CIFAR10 by default.

    This inherits from flower client for federated learning.
    '''
    def __init__(self, dp_model, name='default_name', device='cuda', writer=None):
        self.device = device
        self.writer = writer
        self.dp_model = dp_model
        self.net = self.dp_model.model
        self.dp_model.name = name
        self.name = name
        self.energy = 500
        self.round = 0

    def get_parameters(self):
        '''
        Returns the parameters of the local net
        '''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        '''
        Sets the local net's parameters to the parameters given
        '''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.dp_model.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        self.dp_model.sup_train()

        _,accuracy = self.dp_model.test()

        #decrement energy based on number of epochs performed this round; 1 energy per epoch
        self.energy-=self.dp_model.learning_epochs
        #decrement 20 energy just for sending things
        self.energy-=20
        
        self.writer.add_scalar(f'data/node_{self.name}/energy',self.energy,self.round)
        self.round+=1
        
        print(f'[Node {self.name}]\tenergy: {self.energy}\tlocal round: {self.round}')

        self.save_node()
        return self.get_parameters(self.net), len(self.dp_model.trainloader), {"accuracy": accuracy}
    
    def evaluate(self, parameters, config):
        print(f"[Node {self.name}] evaluate, config: {config}")
        self.set_parameters(parameters)
        loss, accuracy = self.dp_model.test()
        return float(loss), len(self.dp_model.testloader), {"accuracy": float(accuracy)}

    def save_node(self):
        '''
        Saves node data to the node state directory
        '''

        path = f'node_states/node_{self.name}.nd'
        with open(path,'wb') as handle: 
            pickle.dump(self, handle)
        print(f'Dumped {self} information in {path}')

    @classmethod
    def load_node(cls,name: str):
        '''
        Loads node data to the node state directory
        '''
        path = f'node_states/node_{name}.nd'
        with open(path,'rb') as handle:
            return pickle.load(handle)
