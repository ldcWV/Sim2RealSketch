import torch
from generators import *
from canvas import *
from SketchNet import SketchNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import wandb
from tqdm import tqdm

class SketchDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        state, brushPos, target, action = self.data[idx]
        return (state, brushPos, target, action)

def generateData(n):
    # returns length n array of (state, target image, action) pairs
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])
    gen = CubicBezierGenerator()
    
    res = []
    while len(res) < n:
        replay = gen.gen()
        target_image = preprocess(getCanvas(replay.getFinalState()))
        for (state, action) in replay.history:
            current_image = preprocess(getRenderedState(state))
            brush_pos = torch.Tensor([state.brushX, state.brushY, state.brushHeight])
            action = torch.from_numpy(action)
            res.append((current_image, brush_pos, target_image, action.float()))

    res = res[:n]
    return res

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SketchNet()
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters())
mse = nn.MSELoss()

wandb.init(project="SketchNet")

for i in range(1000):
    print(f"i={i}")
    wandb.log({
        'i': i
    })
    dataset = SketchDataset(generateData(10000))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    for epoch in tqdm(range(3), "Epochs"):
        for (state, brushPos, target, action) in dataloader:
            x = torch.cat([state, target], axis=1) # (N, 6, 256, 256)
            y = action # (N, 3)
            x = x.to(device)
            brushPos = brushPos.to(device)
            y = y.to(device)
            
            yhat = model(x, brushPos) # (N, 3)
                        
            loss = mse(y, yhat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({
                'loss': loss
            })
