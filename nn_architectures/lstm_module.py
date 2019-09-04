"""
Vanilla LSTM implementation based on https://arxiv.org/pdf/1503.04069.pdf
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class VanillaLSTM():

    '''
    Creates a new LSTM module
    Inputs:
        @input_dimension: number of input channels
        @block_size: number of output channels
        @input_activation_function: Activation function to be applied to the
        input gate, usually tanh
        @output_activation_function: Activation function to be applied to the
        output, usually tanh
    '''
    def __init__(self,
                 input_dimension,
                 block_size,
                 output_dimension,
                 input_activation_function = np.tanh,
                 output_activation_function = np.tanh):
        self.input_dimension = input_dimension
        self.block_size = block_size
        self.output_dimension = output_dimension

        self.initialize_state()
        self.initialize_weights()

        self.input_activation_function = input_activation_function
        self.output_activation_function = output_activation_function

    ''' Sets the internal state (c and y) to 1 '''
    def initialize_state(self):
        self.c = np.ones(self.block_size)
        self.y = np.ones(self.block_size)

    ''' Sets the internal state (c and y) to 0 '''
    def reset_state(self):
        self.c = np.zeros(self.block_size)
        self.y = np.zeros(self.block_size)

    ''' Initialize all kernel weights to 0.5 and all bias weights to 0 '''
    def initialize_weights(self):
        # Input weights
        self.P_out = 0.5*np.ones([self.block_size,self.output_dimension])

        self.W_z = 0.5*np.ones([self.block_size,self.input_dimension])
        self.W_i = 0.5*np.ones([self.block_size,self.input_dimension])
        self.W_f = 0.5*np.ones([self.block_size,self.input_dimension])
        self.W_o = 0.5*np.ones([self.block_size,self.input_dimension])

        # Recurrent weights
        self.R_z = 0.5*np.ones([self.block_size,self.block_size])
        self.R_i = 0.5*np.ones([self.block_size,self.block_size])
        self.R_f = 0.5*np.ones([self.block_size,self.block_size])
        self.R_o = 0.5*np.ones([self.block_size,self.block_size])

        # Bias weights
        self.b_z = np.zeros(self.block_size)
        self.b_i = np.zeros(self.block_size)
        self.b_f = np.zeros(self.block_size)
        self.b_o = np.zeros(self.block_size)

    '''
    Performs one forward step of the LSTM block
    Inputs:
        @x: numpy array of size input_dimension, Input values
    '''
    def step(self,x):
        # Block Input

        # single values must be converted to a 1-D array
        if(not x is np.ndarray):
            x = np.asarray([x])
        x = x.flatten()

        z_bar = np.dot(self.W_z,x) \
                + np.dot(self.R_z,self.y) \
                + self.b_z
        z = self.input_activation_function(z_bar)

        # Input Gate
        i_bar = np.dot(self.W_i,x) \
                + np.dot(self.R_i,self.y) \
                + self.b_i
        i = sigmoid(i_bar)

        # Forget Gate
        f_bar = np.dot(self.W_f,x) \
                + np.dot(self.R_f,self.y) \
                + self.b_f
        f = sigmoid(f_bar)

        # Recurrent Cell
        c_next = z*i + self.c * f

        # Output Gate
        o_bar = np.dot(self.W_o,x) \
                + np.dot(self.R_o,self.y) \
                + self.b_o
        o = sigmoid(o_bar)

        # Block Output
        y_next = self.output_activation_function(c_next) * o

        # Delay
        self.c = c_next
        self.y = y_next

        out = np.dot(self.y,self.P_out)
        # Set members, to access internal values also form outsize the class
        self.block_input = z
        self.forget_gate = f
        self.input_gate = i
        self.output_gate = o
        self.block_output = y_next

        # Return internal state and output
        return out
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

        self.P_backup = self.P_out
        self.W_backup = [self.W_z,self.W_i,self.W_f,self.W_o]
        self.R_backup = [self.R_z,self.R_i,self.R_f,self.R_o]
        self.b_backup = [self.b_z,self.b_i,self.b_f,self.b_o]

        self.P_out += rnd.normal(scale=scale,size=[self.block_size,self.output_dimension])

        self.W_z += rnd.normal(scale=scale,size=[self.block_size,self.input_dimension])
        self.W_i += rnd.normal(scale=scale,size=[self.block_size,self.input_dimension])
        self.W_f += rnd.normal(scale=scale,size=[self.block_size,self.input_dimension])
        self.W_o += rnd.normal(scale=scale,size=[self.block_size,self.input_dimension])

        # Recurrent weights
        self.R_z += rnd.normal(scale=scale,size=[self.block_size,self.block_size])
        self.R_i += rnd.normal(scale=scale,size=[self.block_size,self.block_size])
        self.R_f += rnd.normal(scale=scale,size=[self.block_size,self.block_size])
        self.R_o += rnd.normal(scale=scale,size=[self.block_size,self.block_size])

        # Bias weights
        self.b_z += rnd.normal(scale=scale,size=self.block_size)
        self.b_i += rnd.normal(scale=scale,size=self.block_size)
        self.b_f += rnd.normal(scale=scale,size=self.block_size)
        self.b_o += rnd.normal(scale=scale,size=self.block_size)

    def undo_noise(self):
        self.W_z,self.W_i,self.W_f,self.W_o = self.W_backup
        self.R_z,self.R_i,self.R_f,self.R_o = self.R_backup
        self.b_z,self.b_i,self.b_f,self.b_o = self.b_backup
        self.P_out = self.P_backup

    def feed_weights(self,weights):
        weight_fields = ['W_z', 'W_i', 'W_f', 'W_o', 'R_z', 'R_i', 'R_f', 'R_o', 'b_z', 'b_i', 'b_f', 'b_o','P_out']
        for (i,weight) in enumerate(weights):
            setattr(self, weight_fields[i], weight)

    def print_matrix_shapes(self):
        print('LSTM matrix shapes:')
        print('W_z: ',str(self.W_z.shape))
        print('W_i: ',str(self.W_i.shape))
        print('W_f: ',str(self.W_f.shape))
        print('W_o: ',str(self.W_o.shape))
        print('R_z: ',str(self.R_z.shape))
        print('R_i: ',str(self.R_i.shape))
        print('R_f: ',str(self.R_f.shape))
        print('R_o: ',str(self.R_o.shape))
        print('b_z: ',str(self.b_z.shape))
        print('b_i: ',str(self.b_i.shape))
        print('b_f: ',str(self.b_f.shape))
        print('b_o: ',str(self.b_o.shape))
        print('P_out: ',str(self.P_out.shape))
    '''
    Saves the weights of the LSTM block in a npz file
    Inputs:
        @path: Path where to store the file
    '''
    def save(self,path):
        np.savez(path,
                 W_z=self.W_z,W_i=self.W_i,W_f=self.W_f,W_o=self.W_o,
                 R_z=self.R_z,R_i=self.R_i,R_f=self.R_f,R_o=self.R_o,
                 b_z=self.b_z,b_i=self.b_i,b_f=self.b_f,b_o=self.b_o,
                 P_out = self.P_out)


'''
Loads an LSTM block from a npz file
Inputs:
    @path: Path where to load the file
Outputs:
    @lstm: Restored LSTM block, input dimension and block size is also
    restored from the file
'''
def load(path):
    npzfile = np.load(path)
    W_z = npzfile['W_z']
    block_size = W_z.shape[0]
    input_dimension = W_z.shape[1]

    lstm = VanillaLSTM(input_dimension,block_size)
    lstm.W_z =npzfile['W_z']
    lstm.W_i =npzfile['W_i']
    lstm.W_f =npzfile['W_f']
    lstm.W_o =npzfile['W_o']

    lstm.R_z =npzfile['R_z']
    lstm.R_i =npzfile['R_i']
    lstm.R_f =npzfile['R_f']
    lstm.R_o =npzfile['R_o']

    lstm.b_z =npzfile['b_z']
    lstm.b_i =npzfile['b_i']
    lstm.b_f =npzfile['b_f']
    lstm.b_o =npzfile['b_o']
    lstm.P_out =npzfile['P_out']
    return lstm
