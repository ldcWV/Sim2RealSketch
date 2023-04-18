from canvas import *
from generators import *
import matplotlib.pyplot as plt

gen = CubicBezierGenerator()

replay = gen.gen()
replay.play()