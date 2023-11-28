from MultidimentionalOpinionDynamics import Dynamics
import numpy as np


a = Dynamics(np.matrix([[-1, 0],[1, 0], [0, 4], [0, -1]], dtype=np.float64),
             np.matrix([[1, 0], [0, 1]], dtype=np.float64),
             100)

for opinion in a.Opinions:
    print(opinion)