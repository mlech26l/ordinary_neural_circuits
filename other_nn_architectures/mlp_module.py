"""
"""
import numpy as np

class MLP():

    def __init__(self,
                 input_dimension,
                 hidden_size,
                 output_dimension,
                 activation_function = np.tanh,
                 ):
        self.input_dimension = input_dimension
        self.hidden_size = hidden_size
        self.output_dimension = output_dimension

        self.initialize_weights()

        self.activation_function = activation_function

    ''' Sets the internal state (c and y) to 0 '''
    def reset_state(self):
        pass

    ''' Initialize all kernel weights to 0.5 and all bias weights to 0 '''
    def initialize_weights(self):
        # Input weights
        self.W_1 = np.zeros([self.hidden_size,self.input_dimension])
        self.W_2 = np.zeros([self.output_dimension,self.hidden_size])

        self.b_1 = np.zeros([self.hidden_size])
        self.b_2 = np.zeros([self.output_dimension])

    def step(self,x):
        x = np.tanh(np.dot(self.W_1,x)+self.b_1)
        x = np.dot(self.W_2,x)+self.b_2
        return x
        
    '''
    Initializes all weights randomly, sampled from a zero mean normal
    distribution with standard deviation of 1.
    Inputs:
        @seed: Seed value for the RNG, if set to None the np.random RNG is used
    '''
    def add_noise(self,scale=0.1,seed=None):
        # Input weights
        if(seed is None):
            rnd = np.random
        else:
            rnd = np.random.RandomState(seed)

        self.W_backup = [np.copy(self.W_1),np.copy(self.W_2)]
        self.b_backup = [np.copy(self.b_1),np.copy(self.b_2)]

        self.W_1 += rnd.normal(scale=scale,size=[self.hidden_size,self.input_dimension])
        self.W_2 += rnd.normal(scale=scale,size=[self.output_dimension,self.hidden_size])
        # Bias weights
        self.b_1 += rnd.normal(scale=scale,size=self.hidden_size)
        self.b_2 += rnd.normal(scale=scale,size=self.output_dimension)


    def undo_noise(self):
        self.W_1,self.W_2 = self.W_backup
        self.b_1,self.b_2 = self.b_backup

    def save(self,path):
        np.savez(path,
                 W_1=self.W_1,W_2=self.W_2,
                 b_1=self.b_1,b_2=self.b_2)

def load(path):
    npzfile = np.load(path)
    W_1 = npzfile['W_1']
    W_2 = npzfile['W_2']
    input_dimension = W_1.shape[1]
    hidden_size = W_1.shape[0]
    output_dimension = W_2.shape[0]

    lstm = MLP(input_dimension,hidden_size,output_dimension)
    lstm.W_1 =npzfile['W_1']
    lstm.W_2 =npzfile['W_2']
    lstm.b_1 =npzfile['b_1']
    lstm.b_2 =npzfile['b_2']
    return lstm
