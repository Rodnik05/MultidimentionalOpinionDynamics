from MultidimentionalOpinionDynamics import Dynamics
import numpy as np


a = Dynamics(np.matrix([[-1, 0],[1, 0], [0, 4], [0, -1]], dtype=np.float64),
             np.matrix([[1, 0], [0, 1]], dtype=np.float64),
             100)

x, y = a.GetAgentOpinionByTheme(1, 0)

with open("Result.txt", "w") as file:
    for opinion in a.Opinions:
        file.write(opinion.__repr__())
        file.write("\n")
        
    
with open("opinionArray.txt", "w") as file2:
    for something in x:
        file2.write(something.__repr__())
        file2.write("\n")