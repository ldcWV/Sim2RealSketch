from canvas import *
from replay import Replay
import numpy as np

class CubicBezierGenerator:
    def gen(self):
        state = CanvasState()
        
        length = np.random.uniform(low=15.0, high=90.0, size=1)[0] # length of brush stroke
        height = np.random.uniform(low=-10.0, high=-5.0, size=1)[0] # height of brush at middle of stroke
        bend = np.random.uniform(low=-40.0, high=40.0, size=1)[0] # how far control points are from line between endpoints
        
        # generate p0, p1, p2, p3 starting as if from (0,0) and going to (length,0)
        p0 = np.array([0.0,0.0])
        p1 = np.array([length/3,bend])
        p2 = np.array([2*length/3,bend])
        p3 = np.array([length,0])
        
        # center stroke on (0,0)
        T = np.array([-length/2, 0])
        p0 += T
        p1 += T
        p2 += T
        p3 += T
        
        # rotate points by random amount
        theta = np.random.uniform(low=0.0, high=2*np.pi, size=1)[0]
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c,-s], [s, c]])
        p0 = R @ p0
        p1 = R @ p1
        p2 = R @ p2
        p3 = R @ p3
        
        # translate points to random spot on canvas
        dx = np.random.uniform(low=20.0, high=state.width-20.0, size=1)[0]
        dy = np.random.uniform(low=20.0, high=state.height-20.0, size=1)[0]
        T = np.array([dx, dy])
        p0 += T
        p1 += T
        p2 += T
        p3 += T
        
        # move to start
        history = []
        start = np.array([p0[0], p0[1], 0.0])
        cur = np.array([state.brushX, state.brushY, state.brushHeight])
        while np.linalg.norm(start - cur) > 1e-6:
            action = start - cur
            action[0] = max(min(action[0], 1.0), -1.0)
            action[1] = max(min(action[1], 1.0), -1.0)
            action[2] = max(min(action[2], 0.2), -0.2)
            history.append((state, action))
            state = getNextState(state, action)
            cur += action
        
        # draw the stroke
        def get_pos(t):
            # (x,y) coordinates
            pos = (1-t)**3 * p0 +\
                  3*(1-t)**2*t * p1 +\
                  3*(1-t)*t**2 * p2 +\
                  t**3 * p3
            
            # height
            if t < 1/3:
                h = t*3 * height
            elif 1/3 <= t and t < 2/3:
                h = height
            else:
                h = (1-t)*3 * height
            
            return np.array([pos[0], pos[1], h])
        
        t = 0
        max_dt = 0.05
        while t < 0.99:
            dt = max_dt
            while True:
                new_pos = get_pos(t + dt)
                action = new_pos - cur
                if abs(action[0]) > 1 or abs(action[1]) > 1 or abs(action[2]) > 0.2:
                    dt /= 2
                else:
                    t += dt
                    break
            history.append((state, action))
            state = getNextState(state, action)
            cur = new_pos
        
        return Replay(history)
