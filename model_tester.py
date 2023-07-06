from SketchNet import preprocess, SketchNet
import torch
from generators import CubicBezierGenerator
from canvas import *
import matplotlib.pyplot as plt
import time

gen = CubicBezierGenerator()
target = preprocess(getCanvas(gen.gen().getFinalState()))

plt.imshow(target.permute((1,2,0)))
plt.show()

model = SketchNet()
model.load_state_dict(torch.load("models/model.pt"))
model.eval()

state = CanvasState()

plt.ion()
fig, ax = plt.subplots()
axim = ax.imshow(getRenderedState(state))

for i in range(300):
    cur = preprocess(getRenderedState(state))
    brushPos = torch.Tensor([state.brushX, state.brushY, state.brushHeight])
    x = torch.cat([cur, target], axis=0)
    
    x = x.unsqueeze(0)
    brushPos = brushPos.unsqueeze(0)
    
    action = torch.squeeze(model(x, brushPos)).detach().numpy()
    print(action)
    state = getNextState(state, action)
    axim.set_data(getRenderedState(state))
    fig.canvas.flush_events()
