#%%

import dl_framework.node
from keyword_spotting import ds_cnn

import torch


def tf_collate_fn(batch):
    x, y = zip(*batch)
    x = torch.stack(x).permute(0, 3, 1, 2).type(torch.FloatTensor)
    y = torch.stack(y)
    return x, y

def _set_lr(epoch):
    if epoch > 36:
        return 0.5*0.2*0.2
    if epoch > 24:
        return 0.2*0.2
    if epoch > 12:
        return 0.2
    return 1

class ks_model(dl_framework.node.dp_model):
    def __init__(self, fw_config):
        super().__init__(fw_config)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )

        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr = 0.0005,
        #     weight_decay=1e-4
        # )

        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,_set_lr,-1)

    # Must redefine workers to be 0, bug of tensorflow dataset
    def load_loaders(self, train_batch = 16, test_batch = 16):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=16,
            num_workers=0,
            collate_fn=tf_collate_fn,
            shuffle = True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=16,
            num_workers=0,
            collate_fn=tf_collate_fn,
            shuffle = True,
        ) 

    def sup_train(self):
        print(f'Starting training using device {self.device}...')
        while self.epoch < self.learning_epochs:
            model = self.model
            model.train()
            epochscore = 0
            runloss = 0

            for inputs,labels in self.train_loader:
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
                trainacc = epochscore/len(self.train_set)
            print(f'[Node:{self.name}]\t epoch:{self.global_epoch}/{self.epoch}:\ttestacc:{accuracy}\ttrainacc:{trainacc}\tloss:{runloss}\t')#lr:{self.scheduler.get_lr()}')

            #if self.tb_writer is not None:
            self.writer().add_scalar(f'data/node_{self.name}/loss',runloss,self.global_epoch)
            self.writer().add_scalar(f'data/node_{self.name}/testacc',accuracy,self.global_epoch)
            self.writer().add_scalar(f'data/node_{self.name}/trainacc',trainacc,self.global_epoch)
                              
            # self.scheduler.step()
            self.epoch+=1
            self.global_epoch+=1


class ks_node(dl_framework.node.dl_node):
    '''
    A dl node running image classification on visual wakewords
    '''
    def __init__(self, fw_config, name='default_name'):
        super(ks_node, self).__init__(fw_config,name)

        self.device = fw_config.get_device()

        self.dp_model = ks_model(fw_config)
        self.dp_model.model = ds_cnn.DS_CNN().to(self.device)
        self.net = self.dp_model.model
        self.dp_model.name = name
        self.name = name
        self.energy = 500
        self.round = 0

        # NEED TO SET OPTIMIZER TO TARGET THE NEW MODEL AS WELL
        self.dp_model.optimizer = torch.optim.SGD(
            self.dp_model.model.parameters(),
            lr = 0.002,
            momentum = 0.9,
            weight_decay = 1e-4
        )

if __name__ == '__main__':
    
    from torchaudio.datasets import SPEECHCOMMANDS
    
    trainset = SPEECHCOMMANDS(
        root = '/home/raimarc/lawrence-workspace/data',
        download = True,
        subset = "training"
    )

    testset = SPEECHCOMMANDS(
        root = '/home/raimarc/lawrence-workspace/data',
        download = True,
        subset = "testing"
    )
    

# %%
