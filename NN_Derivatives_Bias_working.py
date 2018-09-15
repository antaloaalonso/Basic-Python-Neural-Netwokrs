# -*- coding: utf-8 -*-
from datetime import datetime #import datetime to time how long it takes to run the script
import numpy as np #import numpy library to do the math
import matplotlib.pyplot as plt #import matplotlib library to make the different plots

#define the neural network
startTime = datetime.now() #starts startTime variable to the time the script start running
number_of_neurons = 10 #set the number of neurons in the hidden layer
number_of_inputs = 1 #set the number of input neurons
number_of_output_neurons = 1 #set the number of output neurons
number_of_training_sets =  1000 #set the number of training sets (sets of inputs 
#we will use to train the net). If we have one input neuron a set of 1000 training sets will consist of 1000 numbers)

speed=20 #the speed is a value that influences the size of the steps you are taking to try to approach the output value when training the net

#generate the training set of data

X=np.random.rand(number_of_training_sets,number_of_inputs) #this generates random inputs according to the number of input neurons and training sets
X=X/np.max(X) #this normalizes the inputs so the maximum input is 1

Y=3*X #this is a simple liner function to generate the outputs for the training sets
#Y=np.sin(X*12) # this is a complex function to generate the outputs for the training sets
Y=1/(1+np.exp(-Y)) #this is a sigmoid function to set the outputs in the training set in a range from 0 to 1

#This is the class corresponding to the main functions of the neural_network
class Neural_Network(object):
    def __init__(self):
        # here we will initialize the weights
        self.Win = np.random.randn(number_of_inputs, number_of_neurons) #this sets the weight for the input neurons to a random initial number
        self.Wout = np.random.randn(number_of_neurons, number_of_output_neurons) #this sets the weight for the output neuron to a random initial number
        self.Wbin = np.random.randn(1,number_of_neurons)#this sets the weight for the bias in the input neurons to a random initial number
        self.Wbout = np.random.randn(1,number_of_output_neurons)#this sets the weight for the bias of the output neuron to a random initial number


    def yHat_calculator(self,X):
        #this will produce an estimate of the output based on the inputs and the weights
        self.act1 = (np.dot(X,self.Win))+(self.Wbin) #this calculates the act values for the input neurons
        self.sig_act1 = self.sigmoid(self.act1) #this calculates the sigmoid of the act values for the input neurons
        self.act2 = (np.dot(self.sig_act1,self.Wout))+(self.Wbout) #this calculates the act values for the output neurons
        self.yHat = self.sigmoid(self.act2) #this calculates the sigmoid of the act values for the output neuron

        return self.yHat


    def cost_calculator(self,Y): #this defines what the cost function is and calculates the cost value based on the values in the training 
        #set and the outputs predicted by the NN
        #(the cost function gives you the difference between the real output from the train set and the predicted by the NN)
        self.yHat=self.yHat_calculator(X)
        self.cost = (sum(((Y-self.yHat)**2)/2))/number_of_training_sets 
        return self.cost
        
    def slope_calculator(self,X,Y):#this calculates the slope of the cost as a function of the different weights.
        #The outputs are the slope of the cost as a function of each of the weights
        self.dcostdwoutpartial=(np.multiply((-Y+self.yHat),self.sigmoid_derivada(self.act2)))
        self.dcostdwout=np.dot(self.sig_act1.T,self.dcostdwoutpartial)
        self.dcostdwout=self.dcostdwout/number_of_training_sets
              
        self.dcostdwinpartial=np.dot(self.dcostdwoutpartial,self.Wout.T)*self.sigmoid_derivada(self.act1)
        self.dcostdwin=np.dot(X.T,self.dcostdwinpartial)
        self.dcostdwin=self.dcostdwin/number_of_training_sets
        
        self.dcostdWbout=np.mean(self.dcostdwoutpartial)
        self.dcostdWbout=self.dcostdWbout
        
        Xb=np.ones(X.shape)
        self.dcostdWbin=np.dot(Xb.T,self.dcostdwinpartial)
        self.dcostdWbin=self.dcostdWbin/number_of_training_sets

        return (self.dcostdwout,self.dcostdwin,self.dcostdWbout,self.dcostdWbin)
        
        
        
        

    def sigmoid(self,x):#This calculates the sigmoid of an input
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivada(self,x):#This calculates the slope of a sigmoid function
        return np.exp(-x)/((1+np.exp(-x))**2)

#This is the class to train the net
class Trainer(object):
    def __init__(self,N):
        self.N=N
        self.yHat=self.N.yHat_calculator(X)
        (self.slopewout,self.slopewin,self.slopewbout,self.slopewbin)=self.N.slope_calculator(X,Y)


    def train(self):
        for q in range (1000):#this is the number of iterations we will use to adjust the weights so the cost is minimized
            
                    
            self.N.Win = self.N.Win - (self.slopewin)*speed #this modifies the weights of the inputs using the information from the 
            #slope of the cost function with respect to the Win and the constant speed (size of the steps you want to take)
            (self.slopewout,self.slopewin,self.slopewbout,self.slopewbin)=self.N.slope_calculator(X,Y)# this calculates the slopes of the cost function
            #as a function of each one of the weights
                    
            self.N.Wbin = self.N.Wbin - (self.slopewbin)*speed
            (self.slopewout,self.slopewin,self.slopewbout,self.slopewbin)=self.N.slope_calculator(X,Y)
                                        
            self.N.Wout = self.N.Wout - (self.slopewout) * speed
            (self.slopewout,self.slopewin,self.slopewbout,self.slopewbin)=self.N.slope_calculator(X,Y)
                    #
                    
            self.N.Wbout = self.N.Wbout - (self.slopewbout)*speed
            (self.slopewout,self.slopewin,self.slopewbout,self.slopewbin)=self.N.slope_calculator(X,Y)
                    
       
            self.cost=self.N.cost_calculator(Y)# this calculates the new cost after we have adjusted all the weights according to the corresponding 
            #cost function slopes
            plt.plot(q,self.cost,'r.')#this just plots the cost in each iteration
           
        plt.show()
        self.q=q







NN=Neural_Network()
T=Trainer(NN)
T.train()

for w in range(1000):#I use this to plot the values from the training set against the values obtained with the trained NN
    NewX=w/1000
    #YNew=np.sin(NewX*12)
    YNew=3*NewX
   # YNew=1/(1+np.exp(-YNew))
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
