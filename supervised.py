# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:57:27 2017

@author: Karim ASSAAD
language: python3
"""
import numpy as np

class linear_regression():
    def __init__(self,x,Y,method='ne',alpha=0.01,num_iters=1500,normalisation=True,test=False):
        self.method=method
        if((method!='gd')&(method!='ne')):
            print('Error: le parametre methode peut prendre seulement 2 parametres ne (normal equation) and gd (gradient decent)')
        else:
            if(method=='gd'):
                self.num_iters=num_iters
                self.alpha=alpha
            self.m=x.shape[0]
            try:
                self.n=x.shape[1]
            except:
                self.n=1
            self.X = np.full((self.m, self.n+1), 1.0)
            self.X[:,1:] = x.reshape(self.m,self.n)
            self.Y=Y
        
    def fit(self):
        def normal_equation(self):
            #teta = inverse(X'*X)*X'*Y
            theta=np.linalg.inv(self.X.transpose().dot(self.X)).dot(self.X.transpose().dot(self.Y))
            return theta

        def gradient_descent(self):       
            def cost_function(self, theta):
                h=self.X.dot(theta);
                sqrErrors=(h-y)**2;
                J=(1/(2*self.m)) * sum(sqrErrors)
                return J
    
            theta = np.zeros([self.n+1,1])
            self.cost_history=[]
            for iter in range(self.num_iters):            
                self.delta=(self.X.transpose().dot(self.X).dot(theta)-self.X.transpose().dot(self.Y))/self.m
                theta=theta-self.alpha*self.delta
                self.cost_history.append(cost_function(self, theta))
            return theta


        if(self.method=='ne'):
            self.theta=normal_equation(self)
        elif(self.method=='gd'):
            self.theta=gradient_descent(self)
        else:
            print('Error2: le parametre methode peut prendre seulement 2 parametres ne (normal equation) and gd (gradient decent)')

    
    def predict(self,x):
        try:
            m=x.shape[0]
            try:
                n=x.shape[1]
            except:
                n=1
            X = np.full((m, n+1), 1.0)
            X[:,1:] = x.reshape(m,n)
            pred=X.dot(self.theta)
            return pred        
        except: 
            print('Error3: le parametre methode peut prendre seulement 2 parametres ne (normal equation) and gd (gradient decent)')




class logistic_regression():        
    def __init__(self,x,Y,threshold=0.5, alpha=0.01,num_iters=1500,normalisation=True,test=False,reg_lambda=0):
        self.threshold=threshold
        self.num_iters=num_iters
        self.alpha=alpha
        self.m=x.shape[0]
        try:
            self.n=x.shape[1]
        except:
            self.n=1
        self.X = np.full((self.m, self.n+1), 1.0)
        self.X[:,1:] = x.reshape(self.m,self.n)
        self.Y=Y
        self.reg_lambda=reg_lambda


    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

        
    def fit(self):
        def gradient_descent(self):       
            def cost_function(self, theta):
                h=self.sigmoid(self.X.dot(theta))
                logisf = (-y)*np.log(h)-(1-y)*np.log(1-h)
                J=((1/self.m)*sum(logisf))+(self.reg_lambda/(2*self.m))*sum(np.power(theta[1:len(theta)],2))
                return J    
            theta = np.zeros([self.n+1,1])
            self.cost_history=[]
            for iter in range(self.num_iters):            
                h=self.sigmoid(self.X.dot(theta))
                self.delta=(self.X.transpose().dot(h)-self.X.transpose().dot(self.Y))/self.m
                theta=theta-self.alpha*self.delta
                self.cost_history.append(cost_function(self, theta))
            return theta
        self.theta=gradient_descent(self)

    
    def predict(self,x):
        m=x.shape[0]
        try:
            n=x.shape[1]
        except:
            n=1
        X = np.full((m, n+1), 1.0)
        X[:,1:] = x.reshape(m,n)
        pred=self.sigmoid(X.dot(self.theta))
        pred[pred>=self.threshold]=1
        pred[pred<self.threshold]=0
        return pred        
