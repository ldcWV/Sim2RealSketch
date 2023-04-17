from canvas import *
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
        
        # execution
        cur = np.array([p0[0],p0[1],0])
        # move to start
        action = np.array([
            cur[0] - state.brushX,
            cur[1] - state.brushY,
            cur[2] - state.brushHeight
        ])
        state = getNextState(state, action)
        
        # draw the stroke
        N = 50
        for timestep in range(N):
            t = timestep / (N-1)
            pos = (1-t)**3 * p0 +\
                  3*(1-t)**2*t * p1 +\
                  3*(1-t)*t**2 * p2 +\
                  t**3 * p3
            if t < 1/3:
                h = t*3 * height
            elif 1/3 <= t and t < 2/3:
                h = height
            else:
                h = (1-t)*3 * height
            new_cur = np.array([pos[0], pos[1], h])
            action = new_cur - cur
            state = getNextState(state, action)
            cur = new_cur
        
        return getRenderedState(state)
