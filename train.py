import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TransitionDataset
from encoder import Encoder, Predictor



# config 
LATENT_DIM = 256 
BATCH_SIZE = 64 
LR = 1e-4 
EPOCHS = 10 
LAMBDA = 1.0 # avoid collapse 

# VICReg , if std ~ 0 -> collapsed 
def variance_reg(z, eps=1e-4): 
    std = torch.sqrt(torch.var(z, dim=0) + eps) 
    return torch.mean(F.relu(1.0 - std))

    
def main(): 
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    
    # load data 
    dataset = TransitionDataset(data_dir="data")
    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=4,
    )

    # init instance 
    enc = Encoder(latent_dim=LATENT_DIM).to(device)
    pred = Predictor(latent_dim=LATENT_DIM, action_dim=2).to(device)


    # init optimizer 
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=LR)
    
    # loop 
    for epoch in range(EPOCHS): 
        tot_pred, tot_var, n = 0.0, 0.0, 0.0
        tot_id = 0.0
        for o_t, a_t, o_t1 in loader: 
            o_t, a_t, o_t1 = o_t.to(device), a_t.to(device), o_t1.to(device)

            # forward 
            z_t = enc(o_t) # enc o_t
            z_t1_target = enc(o_t1).detact()  # enc o_{t+1} (target, KHÔNG stop-grad)
            z_t1_pred = pred(z_t, a_t) # đoán z_{t+1} từ z_t + a_t

            # loss
            L_pred = F.mse_loss(z_t1_pred, z_t1_target)
            # L_var = variance_reg(torch.cat([z_t, z_t1_target], dim=0)) # chống encoder collapse
            L_var = variance_reg(z_t)
            L_id = F.mse_loss(z_t, z_t1_target) # check if z_t ~ z_t1_pred, if they the same, model learn nothing, it ignore the action  
            loss = L_pred + LAMBDA * L_var

            # step
            opt.zero_grad() # reset the last grad
            loss.backward()  
            opt.step()
 
            # cal tot loss in a batch = 64 
            tot_pred += L_pred.item(); tot_var += L_var.item(); n+= 1
            tot_id += L_id.item()

        with torch.no_grad(): 
            std_mean = torch.std(z_t, dim=0).mean().item()
        print(f"epoch {epoch:2d} | L_pred {tot_pred/n:.4f} | L_id {tot_id/n:.4f} | L_var {tot_var/n:.4f} | z_std {std_mean:.3f}")

    torch.save({"enc": enc.state_dict(), "pred": pred.state_dict()}, "world_model.pt")
    print("saved")



if __name__ == "__main__": 
    main()