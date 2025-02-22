import numpy as np

class Dynamics:
    def __init__(self,
                 StartingOpinionMatrix : np.matrix,
                 GramMatrix : np.matrix,
                 MaxIterations : int,
                 IterationsAfterStabilization : int = 0):
        self.Opinions = []
        self.Opinions.append(StartingOpinionMatrix)
        
        # GramMatrix must be symmetric
        assert np.allclose(GramMatrix, GramMatrix.T), "GramMatrix must be symmetric"
        self.GramMatrix = GramMatrix
        
        self.MaxIterations = MaxIterations
        self.Dists = []
        self.Dists.append(
            self.dist(StartingOpinionMatrix, StartingOpinionMatrix.T, GramMatrix))
        for i in range(MaxIterations + 1):
            if (self.Stabilized()):
                for _ in range(IterationsAfterStabilization):
                    self.ComputeNext()
                break
            self.ComputeNext()
    

    def ComputeNext(self):
        NextOpinionMatrix = self.Opinions[-1].copy()
        for i in range(self.AgentsCount()):
            DifSum = 0
            for j in range(self.AgentsCount()):
                DifSum += (self.Opinions[-1][j] - self.Opinions[-1][i]) * 1/(1 + self.Dists[-1][j, i])
                
            DifSum /= (self.AgentsCount() - 1)
            NextOpinionMatrix[i] += DifSum
            
        self.Opinions.append(NextOpinionMatrix)
        self.Dists.append(
            self.dist(self.Opinions[-1], self.Opinions[-1].T, self.GramMatrix))
        return NextOpinionMatrix


    def GetAgentOpinionByTheme(self, agent : int, theme : int):
        if (agent >= self.AgentsCount()):
            return None
        if (len(self.Opinions) < 0 or theme >= (self.Opinions[0]).shape[1]):
            return None
        opinion = []
        for i in range(len(self.Opinions)):
            opinion.append(self.Opinions[i][agent, theme])
        return opinion
            

    def dist(self, A, B, GramMatrix):
        return (
            np.diag(np.matmul(
                np.matmul(A, GramMatrix), 
                A.T)) +
            np.diag(np.matmul(
                np.matmul(B.T, GramMatrix), 
                B)).reshape(-1, 1) - 
            2 * (np.matmul(
                np.matmul(B.T, GramMatrix), 
                A.T))) 
        
        
    def Stabilized(self):
        if (len(self.Opinions) < 2):
            return False
        return np.array_equal(self.Opinions[-1], self.Opinions[-2]) 
    
    
    def AgentsCount(self):   
        return np.shape(self.Opinions[0])[0]
    
    
    def AgentOpinionsNumber(self):
        return np.shape(self.Opinions[0])[1]
    
    
    def LogResults(self):
        import pandas as pd
        for OpinionId in range(self.AgentOpinionsNumber()):
            df = pd.DataFrame()
            df['iteration'] = np.arange(len(self.Opinions))
            for AgentId in range(self.AgentsCount()):
                opinions = self.GetAgentOpinionByTheme(AgentId, OpinionId)
                df[AgentId] = opinions

            df.to_csv(f"Opinions/Opinions{OpinionId}.txt")