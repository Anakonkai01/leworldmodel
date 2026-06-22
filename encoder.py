"""
remember 
conv_out = (in + 2 * padding - kernel) / stride  +  1


"""
import torch
import torch.nn as nn 




# simple cnn 
class Encoder(nn.Module): 
    def __init__(self, latent_dim = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # -> 96/2 = 48
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 48/2 = 24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 24/2 = 12
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 12/2 = 6
            nn.ReLU(), 
        )
        self.fc = nn.Linear(128 * 6 * 6, latent_dim) # linear to latent dim

    def forward(self, x): 
        # x (b, 3, 96, 96)
        h = self.conv(x) # (b, 128, 6, 6)
        h = torch.flatten(h, start_dim=1) # flatten # (b, 128 * 6 * 6)
        z = self.fc(h) # (b, 128 * 6 * 6)
        return z
    
    
    
class Predictor(nn.Module): 
    def __init__(self, latent_dim=256, action_dim=2,hidden=512): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), # 256 + 2 -> 512
            nn.ReLU(),
            nn.Linear(hidden, hidden), # 512 -> 512
            nn.ReLU(),                 # thiếu cái này -> 2 Linear gập thành 1
            nn.Linear(hidden, latent_dim),#  512 -> 256
        )

    def forward(self, z, a):
        # z (b, 256), a (b, 2)
        x = torch.cat([z, a], dim=1) # x (b, 258)
        z_next = z + self.net(x) # residual: học PHẦN THAY ĐỔI -> (b, 256)
        return z_next



        

if __name__ == "__main__":
    enc = Encoder(latent_dim=256)
    pred = Predictor(latent_dim=256, action_dim=2)
    x = torch.randn(64, 3, 96, 96)        # giả lập 1 mẻ ảnh
    a = torch.randn(64, 2)
    z = enc(x)
    z_next = pred(z, a) # residual đã nằm trong forward
    print("z:", z.shape, "| z_next:", z_next.shape)   # cả hai (64, 256)

