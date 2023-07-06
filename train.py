import torch
from generators import *
from canvas import *
from SketchNet import SketchNet, preprocess
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    gen = CubicBezierGenerator()
    
    res = []
    while len(res) < n:
        replay = gen.gen()
        target_image = preprocess(getCanvas(replay.getFinalState()))
        for (state, action) in replay.history:
            current_image = preprocess(getRenderedState(state))
            brush_pos = torch.Tensor([state.brushX, state.brushY, state.brushHeight])
            action = torch.from_numpy(action).float()
            res.append((current_image, brush_pos, target_image, action))

    res = res[:n]
    return res

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SketchNet()
model.load_state_dict(torch.load("models/model.pt"))
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters())

def loss(y, yhat):
    mse = nn.MSELoss(reduction='none')
    y = y / torch.Tensor([1, 1, 0.2]).to(y.device)
    yhat = yhat / torch.Tensor([1, 1, 0.2]).to(yhat.device)
    return torch.mean(torch.sum(mse(y,yhat),dim=1))

wandb.init(project="SketchNet")

for i in range(1000):
    print(f"i={i}")
    wandb.log({
        'i': i
    })
    torch.save(model.state_dict(), f"models/model.pt")
    dataset = SketchDataset(generateData(2000))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    # for i, (state, brushPos, target, action) in enumerate(tqdm(dataloader)):
    for i, (state, brushPos, target, action) in enumerate(dataloader):
        x = torch.cat([state, target], axis=1) # (N, 6, 256, 256)
        y = action # (N, 3)
        x = x.to(device)
        brushPos = brushPos.to(device)
        y = y.to(device)

        yhat = model(x, brushPos) # (N, 3)

        l = loss(y, yhat)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        wandb.log({
            'loss': l
        })
        # if i%10 == 0:
        #     current_state = torch.permute(state[0], (1,2,0))
        #     desired_state = torch.permute(target[0], (1,2,0))
        #     concatted_state = torch.cat([current_state, desired_state], axis=0) # (512, 256, 3)
        #     print("brushPos", brushPos[0])
        #     print("y", y[0])
        #     print("yhat", yhat[0])
        #     plt.imshow(concatted_state)
        #     plt.show()
    print("y", y)
    print("yhat", yhat)
    print("loss", l)
