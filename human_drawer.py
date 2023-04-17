import numpy as np
import matplotlib.pyplot as plt
from canvas import *
import time

plt.ion()
fig, ax = plt.subplots()
fig = plt.gcf()
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
canvas = ax.figure.canvas

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

class Game:
    def __init__(self):
        self.state = CanvasState(minBrushHeight=-30, maxBrushHeight=10)
        self.d = 2
        self.dh = 0.2
        self.done = False
        canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def on_key_press(self, event):
        if event.key == 'w':
            self.state = getNextState(self.state, np.array([0, 1, 0]) * self.d)
        elif event.key == 'a':
            self.state = getNextState(self.state, np.array([-1, 0, 0]) * self.d)
        elif event.key == 's':
            self.state = getNextState(self.state, np.array([0, -1, 0]) * self.d)
        elif event.key == 'd':
            self.state = getNextState(self.state, np.array([1, 0, 0]) * self.d)
        elif event.key == 'up':
            self.state = getNextState(self.state, np.array([0, 0, 1]) * self.d)
        elif event.key == 'down':
            self.state = getNextState(self.state, np.array([0, 0, -1]) * self.d)
        elif event.key == 'escape':
            self.done = True

game = Game()

axim = ax.imshow(getRenderedState(game.state))

while True:
    if game.done:
        break
    axim.set_data(getRenderedState(game.state))
    fig.canvas.flush_events()
    time.sleep(0.05)