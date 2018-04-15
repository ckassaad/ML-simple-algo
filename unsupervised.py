# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:57:27 2017

@author: Karim ASSAAD
language: python3
"""
import numpy as np

class k_means:
    def __init__(self,k,nbmax=1000,alpha=0.01):
        self.nbmax=nbmax
        self.alpha=alpha
        self.k = k  

    def __initialize__(self):
        self.coordonnees = np.zeros((self.k, self.ncol))
        for cpt1 in range(self.k):
            indice = np.random.randint(0,self.nrow)
            for cpt2 in range(self.ncol):
                self.coordonnees[cpt1][cpt2] =  self.desIndividus[indice][cpt2]
                
    def __show__(self):
       print("Les centres sont:")
       print(self.coordonnees)
       
    def __DistanceBetweenRowsAndCentroids__(self):
        # devra retourner les distances de chacun des individus
        # a chacun des centres -> matrice(nbDIndividus,nbDeCentres)
        Dist=np.zeros((self.nrow, self.k))        
        for i in range(self.nrow):
            distance=[]
            for j in range(self.k):
                distance.append(np.sqrt(np.sum(np.power(self.desIndividus[i,:]-self.coordonnees[j,:],2))))
            Dist[i]=distance
        print('matrice des distance:')
        print(Dist)
        return Dist
        
    def __AssigningDataToCentroids__(self):
        # devra retourner le centre le plus proche
        # pour chacun des individus -> matrice(nbDIndividus,1)
        self.Cluster=[]
        Dist=self.__DistanceBetweenRowsAndCentroids__()
        for i in range(self.nrow):
            self.Cluster.append(np.argmin(Dist[i])+1)
        print('Les clusters sont:')
        print(self.Cluster)
    
    def __Centroids__(self):
        # apres appel a __calculerLeCentreLePlusProche__
        # devra determiner les donnees affectees a chaque centre
        # avant de recalculer le centre de chaque aggregat
        for i in np.unique(self.Cluster):
            indice=np.where(np.array(self.Cluster) == i)[0]           
            for j in range(self.ncol):
                self.coordonnees[i-1,j]=np.mean(self.desIndividus[indice,j])                
    
    def fit(self,desIndividus):
        # apprendre des centres, en les initialisant, ...
        # ...
        # attention, il faudra aussi des alphas d'arret
        self.desIndividus=desIndividus
        self.ncol= desIndividus.shape[1]
        self.nrow = desIndividus.shape[0]
        self.__initialize__()
        for i in range(self.nbmax):
            self.__AssigningDataToCentroids__()
            temp=self.coordonnees.copy()
            self.__Centroids__()
            if((np.power(temp-self.coordonnees,2))<=self.alpha).all():
                break
        self.Cluster=np.array(self.Cluster)



class forme_forte(k_means):
    def __init__(self,k,N,nbmax=1000,alpha=0.01):
        self.nbmax=nbmax
        self.alpha=alpha
        self.k = k
        self.N = N

    def fit1(self,desIndividus):
        self.desIndividus=desIndividus
        self.ncol= desIndividus.shape[1]
        self.nrow = desIndividus.shape[0]
        self.__initialize__()
        C = np.zeros((self.nrow, self.N))
        for i in range(self.N):
            if(np.mod(self.N,2)):
                self.__initialiserLesCentresAleatoirement__()
            else:
                self.__initialize__()
            self.fit(desIndividus)
            C[:,i]=self.Cluster
        index_cluster=[]
        for i in range(self.nrow):
            index=[]
            for j in range(i,self.nrow):
                flatten = lambda l: [item for sublist in l for item in sublist]
                if j not in flatten(index_cluster):
                    F=C[i]-C[j]
                    if(np.all(F)==0):
                        index.append(j)
            if (index!=[]):
                index_cluster.append(index)
        print(index_cluster)
        index_cluster=np.array(index_cluster)
        self.Cluster=np.array(self.Cluster)
        for i in range(self.k):
            self.Cluster[index_cluster[i]]=i+1
        return self.Cluster
    
    
class anomaly_detecion():
    def __init__(self, X, val, yval, method='gaussian', num_iter=1000):
        self.X=X
        self.val=val
        self.yval=yval
        self.method=method
        self.num_iter=num_iter

        n=X.shape[0]        
        self.mu=np.mean(X,axis=0)
        self.sig2=np.var(X,axis=0)*(n-1)/n
        
    def gaussian(self,X):
        p=(2*np.pi*self.sig2)**(-1/2)*np.exp(-((X-self.mu)**2)/(2*self.sig2))
        return np.prod(p,axis=1)
    
    def multi_gaussian(self,X):
        p=(2*np.pi**2)*np.exp(X-self.mu)/(self.sig2)
        return np.prod(p,axis=1)
        
    def select_threshold(self):
        bestEpsilon = 0
        bestF1 = 0
    
        for e in range(0,self.num_iter):
            epsilon= np.min(self.pval,axis=0) + e*np.max(self.pval,axis=0)/self.num_iter
            integer=np.vectorize(np.int)
            predictions = integer(self.pval < epsilon)
            tp = len(predictions[(predictions==1) & (self.yval==1)])
            fp = len(predictions[(predictions==1) & (self.yval==0)])
            fn = len(predictions[(predictions==0) & (self.yval==1)])
            if(tp==0):
                precision=0
                recall=0
                F1 =0
            else:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                F1 = 2*precision*recall/(precision+recall)
            if (F1 > bestF1):
               bestF1 = F1
               bestEpsilon = epsilon    
        return bestEpsilon

    def train(self):
        if(self.method=='gaussian'):
            self.pval=self.gaussian(self.val)
        elif(self.method=='multi_gaussian'):
            self.pval=self.multi_gaussian(self.val)
        else:
            print('Parameter method must be either gaussian or multi_gaussian.')    
        self.epsilon = self.select_threshold()
    
    def predict(self,test):
        if(self.method=='gaussian'):
            p=self.gaussian(test)
        elif(self.method=='multi_gaussian'):
            p=self.multi_gaussian(test)
        else:
            print('Parameter method must be either gaussian or multi_gaussian.')  
        integer=np.vectorize(np.int)
        return integer(p < self.epsilon)