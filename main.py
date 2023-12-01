from MultidimentionalOpinionDynamics import Dynamics
import numpy as np


a = Dynamics(np.matrix([[-1, 0, 8],[1, 0,10], [0, 4,100], [10, -1, -2]], dtype=np.float64),
             np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
             10000)

with open("Result.txt", "w") as file:
    for opinion in a.Opinions:
        file.write(opinion.__repr__())
        file.write("\n")


with open("Dists.txt", "w") as file:
    for dist in a.Dists:
        file.write(dist.__repr__())
        file.write("\n")


import pandas as pd



for OpinionId in range(a.AgentOpinionsNumber()):
    df = pd.DataFrame()
    df['iteration'] = np.arange(len(a.Opinions))
    for AgentId in range(a.AgentsCount()):
        opinions = a.GetAgentOpinionByTheme(AgentId, OpinionId)
        df[AgentId] = opinions

    df.to_csv(f"Opinions/Opinions{OpinionId}.txt")

