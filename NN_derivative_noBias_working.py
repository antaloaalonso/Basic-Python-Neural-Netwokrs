from datetime import datetime
startTime = datetime.now()
import numpy as np
import matplotlib.pyplot as plt

#define the neural network
number_of_neurons = 10
number_of_inputs = 1
number_of_output_neurons = 1
number_of_training_sets =  1000

speed=20
X=np.random.rand(number_of_training_sets,number_of_inputs)
X=X/np.max(X)

Y=np.sin(X*12)
#Y=3*X
Y=1/(1+np.exp(-Y))



class Neural_Network(object):
    def __init__(self):
        # here we will initialize the weights
        self.Win = np.random.randn(number_of_inputs, number_of_neurons)
        self.Wout = np.random.randn(number_of_neurons, number_of_output_neurons)
#        self.WinNew=self.Win
#        self.WoutNew=self.Wout


    def yHat_calculator(self,X):
        #this will produce an estiamte of the output based on the inputs and the weights
        self.act1 = np.dot(X,self.Win)
        self.sig_act1 = self.sigmoid(self.act1)
        self.act2 = np.dot(self.sig_act1,self.Wout)
        self.yHat = self.sigmoid(self.act2)
        return self.yHat
    
#    def yHat_calculatorNew(self,X):
#        #this will produce an estiamte of the output based on the inputs and the weights
#        self.act1 = np.dot(X,self.WinNew)
#        self.sig_act1 = self.sigmoid(self.act1)
#        self.act2 = np.dot(self.sig_act1,self.WoutNew)
#        self.yHat = self.sigmoid(self.act2)
#        return self.yHat


    def cost_calculator(self,Y):
        self.yHat=self.yHat_calculator(X)
        self.cost = (sum(((Y-self.yHat)**2)/2))/number_of_training_sets 
        return self.cost
        
    def slope_calculator(self,X,Y):
       self.dcostdwoutpartial=(np.multiply((-Y+self.yHat),self.sigmoid_derivada(self.act2)))
       self.dcostdwout=np.dot(self.sig_act1.T,self.dcostdwoutpartial)
       self.dcostdwout=self.dcostdwout/number_of_training_sets
        
       self.dcostdwinpartial=np.dot(self.dcostdwoutpartial,self.Wout.T)*self.sigmoid_derivada(self.act1)
       self.dcostdwin=np.dot(X.T,self.dcostdwinpartial)
       self.dcostdwin=self.dcostdwin/number_of_training_sets
           
        
       return (self.dcostdwout,self.dcostdwin)
        
        
        
        

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivada(self,x):
        return np.exp(-x)/((1+np.exp(-x))**2)



class Trainer(object):
    def __init__(self,N):
        self.N=N
        self.yHat=self.N.yHat_calculator(X)
        (self.slopewout,self.slopewin)=self.N.slope_calculator(X,Y)


    def train(self):
        for q in range (1000):
 
            self.N.Win = self.N.Win - (self.slopewin)*speed
            (self.slopewout,self.slopewin)=self.N.slope_calculator(X,Y)
                       
            self.N.Wout = self.N.Wout - (self.slopewout) * speed
            (self.slopewout,self.slopewin)=self.N.slope_calculator(X,Y)
     
            self.cost=self.N.cost_calculator(Y)
            plt.plot(q,self.cost,'r.')

        plt.show()
        self.q=q

NN=Neural_Network()
T=Trainer(NN)
T.train()

for w in range(1000):
    NewX=w/1000
    #YNew=3*NewX
    YNew=np.sin(NewX*12)
    #YNew=1/(1+np.exp(-YNew))
    YY= np.log(NN.yHat_calculator(NewX)/(1-NN.yHat_calculator(NewX)))
    plt.plot(NewX,YY,'c.')
    plt.plot(NewX,YNew,'r.')
plt.show()

print(' Expected result=',YNew,'\n','Result from the NN=',YY)
Win=NN.Win
Wout=NN.Wout
print (' Time elapsed=',datetime.now() - startTime)
print (' NUMERO OF ITERATIONS=', T.q+1)
print(' Number_of_neurons=',number_of_neurons,'\n','Number_of_inputs=',number_of_inputs,'\n',
'Number_of_output_neurons=',number_of_output_neurons,'\n','Number_of_training_sets=',number_of_training_sets,'\n','Speed=',speed)
print(' Cost=',T.cost)
