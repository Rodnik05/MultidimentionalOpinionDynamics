from MultidimentionalOpinionDynamics import Dynamics
import numpy as np


a = Dynamics(np.matrix([[-1, 0],[1, 0], [0, 4], [0, -1]], dtype=np.float64),
             np.matrix([[1, 0], [0, 1]], dtype=np.float64),
             100)

with open("Result.txt", "w") as file:
    for opinion in a.Opinions:
        file.write(opinion.__repr__())
        file.write("\n")


with open("Dists.txt", "w") as file:
    for dist in a.Dists:
        file.write(dist.__repr__())
        file.write("\n")

       
for i in range(a.AgentsCount()):
    for j in range(a.AgentOpinionsNumber()):  
        x, y = a.GetAgentOpinionByTheme(i, j)  
        with open(f"OpinionsOfAgents/OpinionsAbout{j}OfAgent{i}.txt", "w") as file2:
            for something in x:
                file2.write(something.__repr__())
                file2.write("\n")