#%%
import torch
from torch.utils.data import Dataset

class mlperftiny_ks_default_params():

    def __init__(self):

        self.data_dir = '/home/raimarc/lawrence-workspace/data'
        self.bg_path = '/home/raimarc/lawrence-workspace/data'
        self.background_volume = 0.1
        self.background_frequency = 0.8
        self.silence_percentage = 10.0
        self.unknown_percentage = 10.0
        self.time_shift_ms = 100.0
        self.sample_rate = 16000
        self.clip_duration_ms = 1000
        self.window_size_ms = 30.0
        self.window_stride_ms = 20.0
        self.feature_type = "mfcc"
        self.dct_coefficient_count = 10
        self.epochs = 36
        self.num_train_samples = -1 # 85511
        self.num_val_samples = -1 # 10102
        self.num_test_samples = -1 # 4890
        self.batch_size = 100
        self.num_bin_files = 1000
        self.bin_file_path = '/home/raimarc/lawrence-workspace/cidr-ufl/keyword_spotting'
        self.model_architecture = 'ds_cnn'
        self.run_test_set = True
        self.saved_model_path = 'trained_models/kws_model.h5'
        self.model_init_path = None
        self.tfl_file_name = 'trained_models/kws_model.tflite'
        self.learning_rate = 0.00001
        self.lr_sched_name = 'step_function'
        self.plot_dir = './plots'
        self.target_set = 'te'

def _get_numpy_listblock(tf_dataset,block_idx,block_size=500):
    import numpy as np
    start_idx = block_idx*block_size
    x_list = []
    from tqdm import tqdm
    tf_iterator = tf_dataset.skip(start_idx).as_numpy_iterator()
    for i,data in tqdm(enumerate(tf_iterator)):
        x, y = data
        x = np.array(x.copy())
        y = np.array(y.copy())
        x_list += [(x,y)] 
        if i > block_size:
            break
    return x_list

class mlperftiny_ks_dset_blocked(Dataset):
    '''
    A Pytorch Dataset wrapping the 12-class keyword spotting dataset from "Hello Edge" and "MLPerfTiny"

    TODO: Loads the entire non-local dataset for every node into memory. Ideally, each node will only
    load its own local subset, to save RAM.     
    '''
    def __init__(self,params = mlperftiny_ks_default_params(), set='train'):
        
        from keyword_spotting import ks_dset_proto

        trainset, testset, valset = ks_dset_proto.get_training_data(params)
        if set == 'train':
            self.set = trainset
        if set == 'test':
            self.set = testset
        if set == 'val':
            self.set = valset

        self.block_idx= None

    def _loadblock(self,block_idx,block_size=500):
        if self.block_idx != block_idx:
            print(f'Loading block {block_idx} of {block_size} data points')
            self.data_block = _get_numpy_listblock(self.set,block_idx,block_size=block_size)
            self.block_idx = block_idx
        return self.data_block

    def __getitem__(self, idx):
        block_size = 200000
        block_idx = idx // block_size
        element_idx = idx % block_size

        x,y = self._loadblock(block_idx,block_size=block_size)[element_idx]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x,y
    
    def __len__(self):
        return len(self.set)


class mlperftiny_ks_dset(Dataset):
    '''
    A Pytorch Dataset wrapping the 12-class keyword spotting dataset from "Hello Edge" and "MLPerfTiny"

    TODO: Loads the entire non-local dataset for every node into memory. Ideally, each node will only
    load its own local subset, to save RAM.     
    '''
    def __init__(self,params = mlperftiny_ks_default_params(), set='train', root='.'):

        import numpy as np
        self.data_block = np.load(root+f'/ks_{set}set.npy', allow_pickle=True)

    def __getitem__(self, idx):

        x,y = self.data_block[idx]
        x = torch.from_numpy(x.copy())
        y = torch.from_numpy(y.copy())

        return x,y
    
    def __len__(self):
        return len(self.data_block)

if __name__ == '__main__':

    trainset = mlperftiny_ks_dset(set='train',root='/home/raimarc/lawrence-workspace/data/mlperftiny_ks_dset')
    testset = mlperftiny_ks_dset(set='test',root='/home/raimarc/lawrence-workspace/data/mlperftiny_ks_dset')

    from dl_framework import fw_config
    from keyword_spotting import ks_node
    my_config = fw_config()
    my_config.tensorboard_runs_dir = 'tb_data/ks_node'
    my_config.run_name = 'long_run'
    node = ks_node.ks_node(my_config)

    node.dp_model.train_set = trainset
    node.dp_model.test_set = testset
    node.dp_model.load_loaders(train_batch=100)
    node.dp_model.learning_epochs = 64

    node.dp_model.sup_train()
