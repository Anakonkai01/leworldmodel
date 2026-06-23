"""
we need to convert 0-255 -> 0->1 
(H, W, 3) -> (3, W, H) because conv 
Dataset must have __len__(), __getitem__(i)

"""

import glob # match character
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 


V_MAX, OMEGA_MAX = 160.0, 2.5 # normalize later 
FRAME_STRIDE = 6

class TransitionDataset(Dataset): 
    def __init__(self, data_dir="data"):
        files = sorted(glob.glob(f"{data_dir}/ep_[0-9]*.npz")) # [0-9] is specify for numerics
        self.episodes = []  # [(obs, actions), ...] list state of an episode
        self.index = [] # [(ep_i, t), ...] for tracking, which frame belongs to which ep
        
        for ep_i, file in enumerate(files):
            d = np.load(file)
            obs, actions = d["obs"], d["actions"]
            self.episodes.append((obs, actions)) # list of tuple 
            for t in range(len(obs) - FRAME_STRIDE): # because we need t + 1 
                self.index.append((ep_i, t))
                

    def _img(self, frame): 
        # (H, W, 3) uint8 0..255 -> (3, H, W) 0..1
        x = torch.from_numpy(frame).float() / 255.0
        return x.permute(2, 0, 1)
    
    
    def __len__(self): 
        return len(self.index)
    
    
    def __getitem__(self, i): 
        ep_i, t = self.index[i] # get the ep_i and frame t belongs to ep_i 
        obs, actions = self.episodes[ep_i] # get all the state of that ep_i 
        
        # get the state of frame t of that ep_i 
        o_t = self._img(obs[t])
        o_t1 = self._img(obs[t+FRAME_STRIDE])
        a_t = torch.tensor([actions[t, 0]/V_MAX, actions[t, 1]/OMEGA_MAX], dtype=torch.float32)     

        return o_t, a_t, o_t1

        


if __name__ == "__main__": 
    ds = TransitionDataset()
    print("number of transition", len(ds))

    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    o_t, a_t, o_t1 = next(iter(loader))
    print("o_t:", o_t.shape, "| a_t:", a_t.shape, "| o_t1:", o_t1.shape)


        

            


# test 

# data_dir = "data"
# files = sorted(glob.glob(f"{data_dir}/ep_[0-9]*.npz")) # [0-9] is specify for numerics
# for f in files:
#     d = np.load(f)
#     obs, actions = d["obs"], d["actions"]
#     print(obs)
#     print(actions)

        

