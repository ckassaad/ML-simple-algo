# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:10:25 2017

@author: Karim ASSAAD
language: python3
"""
#libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score as f1


class rco_training():
    def __init__(self, X1, X2, Y1, Y2, model1, model2, uld ,n_iter=2, feature_split=False):
        #Verify if data types are correct
        if((type(X1)!=type(pd.DataFrame())) | (type(X2)!=type(pd.DataFrame())) | (type(Y1)!=type(pd.Series())) | (type(Y2)!=type(pd.Series()))):
            raise NameError('Mismatch in data types X1 and X2 should be of type DataFrame and Y1 and Y2 should be of type Series')

        self.X1=X1.copy()
        self.X2=X2.copy()
        self.Y1=Y1.copy()
        self.Y2=Y2.copy()
        self.uld=uld.copy()
        self.n_iter=n_iter
        
        if(type(model1)==type([])):
            if(len(model1)==2):
                self.model1_init=model1[0]
                self.model1=model1[1]
            else:
                raise NameError('The list must contain at most two models')
        else:
            self.model1_init=model1
            self.model1=model1

        if(type(model2)==type([])):
            if(len(model2)==2):
                self.model2_init=model2[0]
                self.model2=model2[1]
            else:
                raise NameError('The list must contain at most two models')
        else:
            self.model2_init=model2
            self.model2=model2
        
        self.n=int((self.uld.shape[0]/self.n_iter))
        if(self.n==0):
            raise NameError('The number of iterations must be smaller than the number of unlabeled observations')

        self.err=list()



    def __feature_split(self):
        print('Will be Available in a another version')
        
        
    #This Function sample two sub unlabeled datasets 
    def __resampeling(self):
        x1=self.uld.sample(self.n, replace=False)
        self.uld.drop(x1.index.values, inplace=True)
        x2=x1.sample(frac=0.5, replace=False)
        x1.drop(x2.index.values, inplace=True)
        return x1,x2


    
    
    def fit(self):
        x1,x2=self.__resampeling()
        try:
            err1=f1(self.Y1,cross_val_predict(self.model1_init, self.X1, self.Y1, cv=10)) 
        except:
            err1=0
        try:
            err2=f1(self.Y2,cross_val_predict(self.model2_init, self.X2, self.Y2, cv=10)) 
        except:
            err2=0
        self.err.append([err1,err2])
        try:
            self.model1_init.fit(self.X1,self.Y1)
        except:
            self.model1_init.fit(self.X1)
        try:
            self.model2_init.fit(self.X2,self.Y2)
        except: 
            self.model2_init.fit(self.X2)
        
        
        y1=self.model1_init.predict(x1)         
        if all(np.unique(y1==[-1.,  1.])):
            y1[y1==-1.]=1
            y1[y1==1.]=0
        y2=self.model2_init.predict(x2)
        if all(np.unique(y2==[-1.,  1.])):
            y2[y2==-1.]=1
            y2[y2==1.]=0

        
        self.X1=self.X1.append(x2).reset_index(drop=True)
        self.X2=self.X2.append(x1).reset_index(drop=True)
        self.Y1=self.Y1.append(pd.Series(y2)).reset_index(drop=True)
        self.Y2=self.Y2.append(pd.Series(y1)).reset_index(drop=True)
        
        for i in range(self.n_iter-1):
            x1,x2=self.__resampeling()
            
            err1=f1(self.Y1,cross_val_predict(self.model1, self.X1, self.Y1, cv=10)) 
            err2=f1(self.Y2,cross_val_predict(self.model2, self.X2, self.Y2, cv=10))
            self.err.append([err1,err2])
            
            self.model1.fit(self.X1,self.Y1)
            self.model2.fit(self.X2,self.Y2)
            
            y1=self.model1.predict(x1)        
            y2=self.model2.predict(x2)
            
            self.X1=self.X1.append(x2).reset_index(drop=True)
            self.X2=self.X2.append(x1).reset_index(drop=True)
            self.Y1=self.Y1.append(pd.Series(y2)).reset_index(drop=True)
            self.Y2=self.Y2.append(pd.Series(y1)).reset_index(drop=True)


    def predict(self):
        print('Will be Available in a another version')


    



