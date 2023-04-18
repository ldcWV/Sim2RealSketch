from canvas import *
import matplotlib.pyplot as plt
import time

class Replay:
    def __init__(self, history):
        # history: list of (state, action) pairs
        self.history = history

    def getFinalState(self):
        return getNextState(self.history[-1][0], self.history[-1][1])
    
    def play(self):
        plt.ion()
        fig, ax = plt.subplots()
        axim = ax.imshow(getRenderedState(self.history[0][0]))
        
        for (state, _) in self.history:
            axim.set_data(getRenderedState(state))
            fig.canvas.flush_events()
            time.sleep(0.02)
        axim.set_data(getRenderedState(self.getFinalState()))
        fig.canvas.flush_events()
        time.sleep(1)
