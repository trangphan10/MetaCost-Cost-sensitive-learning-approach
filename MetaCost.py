'''
Algorithm: 
Inputs: 
- S is the training set 
- L is a classification learning algorithm 
- C is a cost matrix 
- m is the number of resamples to generate 
- n is the number of examples in each resample 
- p is True if L produces class probabilities 
- q is True if all resamples are to be used for each example
Implementation: 
Procedure MetaCost(S,L,C, m,n,p,q)
For i = 1 to m
    Let Si is a resample of S with n examples 
    Let Mi is a model produced from applying L into Si 
For each example x in S: 
    For each class j: 
        Let P(j|x) = divide(i,sum of class)*sum of each j P(j|x,Mi)
        Where 
            if p then P(j|x,Mi) is produced by Mi 
            Else P(j|x,Mi) = 1 for the class predicted by Mi for x, and 0 for all others 
            if q then i ranges over all Mi such that x is not in Si 
        Let x's class = argmini(sum of each j of P(j|x)C(i,j))
'''
import pandas as pd 
import numpy as np 
from sklearn.base import clone
class MetaCost(object):
    def __init__(self,S,L,C,m,n,p = True,q = True): 
        # Check whether S is DataFrame (or structured data)
        if not isinstance(S,pd.DataFrame): 
            raise ValueError('S must be a DataFrame object')
        self.S = S 
        self.L = L 
        self.C = C 
        self.m = m 
        # the input n is the percentage threshold, 
        # so change input to the number used rows in S
        self.n = len(S) * n     
        self.p = p 
        self.q  = q
    def fit(self,flag,num_class): 
        '''
        flag is the name of classification labels 
        num_class is the number of classes
        '''
        # Select columns in S except for label column
        col = [col for col in self.S.columns if col != flag]
        # Create a dict to store S_i (resamples produced from S)
        S_ = {}
        # Create a list to store model when applying L to S_i
        M = []
        
        for i in range(self.m): 
            S_[i] = self.S.sample(n=self.n,replace=True)
            
            # Define X is a set of feature columns,y is label 
            X = S_[i][col].values  # Pick all values of feature columns from S_i dataframe
            y = S_[i][flag].values  # Pick label values from S_i dataframe 
            
            model = clone(self.L)
            M.append(model.fit(X,y))
        
        label = []
        S_array = self.S[col].values 
        
        for i in range(len(self.S)): 
            if not self.q: 
                # Add key from S_ into list if i is not ID row of S_ values
                k_th  = [k for k,v in S_.items() if i not in v.index]
                M_ = list(np.array(M)[k_th])
            else: 
                M_ = M 
            if self.p: 
                # If p = True that means model produce distibution probability between classes 
                # Save probability on P_j for each model in M
                P_j = [model.predict_proba(S_array[[i]]) for model in M_]
            else: 
                P_j = []
                # Create a vector
                vector = [0] * num_class 
                # Each model in M_, save predict value (for example 0 or 1 class label)
                for model in M_: 
                    vector[model.predict(S_array[[i]])] = 1 
                    P_j.append(vector)
                # Each model have a vector represent for prediction value to evaluate misclassification cost 
                # Compute P(j|x)
                P = np.array(np.mean(P_j,0)).T 
                
                # Relabel 
                label.append(np.argmin(self.C.dot(P)))
            
        X_train = self.S[col].values 
        y_train = np.array(label)
         
        new_model = clone(self.L)
        new_model = new_model.fit(X_train,y_train)
        
        return new_model
                
                
        