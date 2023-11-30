import numpy as np

class Dynamics:
    def __init__(self,
                 StartingOpinionMatrix : np.matrix,
                 GramMatrix : np.matrix,
                 t : int):
        self.Opinions = []
        self.Opinions.append(StartingOpinionMatrix)
        self.GramMatrix = GramMatrix
        self.t = t
        self.Dists = []
        self.Dists.append(
            self.dist(StartingOpinionMatrix, StartingOpinionMatrix, GramMatrix))
        for i in range(t + 1):
            if (self.Stabilized()):
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
            self.dist(self.Opinions[-1], self.Opinions[-1], self.GramMatrix))
        return NextOpinionMatrix


    def GetAgentOpinionByTheme(self, agent : int, theme : int):
        if (agent >= self.AgentsCount()):
            return None
        if (len(self.Opinions) < 0 or theme >= len(self.Opinions[0])):
            return None
        ids = []
        opinion = []
        for i in range(len(self.Opinions)):
            ids.append(i)
            opinion.append(self.Opinions[i][agent, theme])
        
        return opinion, ids
            


    def dist(self, A, B, GramMatrix):
        return np.sqrt(
            np.diag(np.matmul(
                np.matmul(A, GramMatrix), 
                A.T)).reshape(-1, 1) +
            np.diag(np.matmul(
                np.matmul(B, GramMatrix), 
                B.T)) - 
            2 * (np.matmul(
                np.matmul(A, GramMatrix), 
                B.T)))  
        
        
    def Stabilized(self):
        if (len(self.Opinions) < 2):
            return False
        return np.array_equal(self.Opinions[-1], self.Opinions[-2]) 
    
    
    def AgentsCount(self):   
        return np.shape(self.Opinions[0])[0]
    
    def AgentOpinionsNumber(self):
        return np.shape(self.Opinions[0])[1]