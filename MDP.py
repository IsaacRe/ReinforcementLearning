"""
Classes and helper methods for MDP actor-critic architectures
"""

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.nn import relu
from os import listdir

hidden_size = 512

""" 
Abstract Class for MDP's in continuous action space.
Uses fully-connected NN to approximate action-value function
"""
class MDP:
    __metaclass__ = ABCMeta

    def __init__(self, s_dim, a_dim, hidden_sizes,
                 num_layers=2,
                 weights_path='models/'):

        self._critic = Critic(num_layers, hidden_sizes,
                                        weights_path + 'critic.pkl')
        self._actor = Actor
        super().__init__()
        
    def action_value(self, s, a):
        return self._critic(s, a)

    def 

"""
Defines architecture for action-deciding FA implementation for actor in continuous action space
"""
class Contiuous_Actor:
    
    """
    Initialize architecture for action decision process
    """   
    def __init__(self, s_dim, a_dim, hidden_sizes,
                 num_layers=2,
                 weights_path='models/'):

        self._critic = Critic(num_layers, hidden_sizes,
                                        weights_path + 'critic.pkl')
        self._actor = Actor
        super().__init__()
        

    def forward(self):
        pass

"""
Initialize architecture for action-value approximation
    a_dim : # of dimensions in action space for continuous,
            # of possible actions for discrete
"""   
class Critic:

    def __init__(self, s_dim, a_dim, hidden_sizes, 
                 num_layers=2,
                 weights_dir='models/',
                 weights_filename='critic.pkl',
                 restart=False):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.num_layers = num_layers
        if len(hidden_sizes) == 1 and num_layers != 1:
            hidden_sizes = [hiddens_sizes for i in range(num_layers)]
        self.hidden_sizes = hidden_sizes
        self.weights_path = weights_dir + weights_filename

        self.weights = {'hiddens': []}

        if not restart and weights_filename in listdir(weights_path):
            # load weights from self.weights_path
            print('Loading critic weights from {}'.format(self.weights_path))
            pass

        self.input = tf.placeholder('float', [None, self.s_dim + self.a_dim], name='input')
        hidden_output, self.weights['hiddens'] = add_fc(self, self.input, relu, 
                                                        self.hidden_sizes, self.num_layers,
                                                        self.weights['hiddens'])
        if not self.weights.has_key('output'):
            self.weights['output'] = (tf.Variable(tf.random_normal([self.hidden_sizes[-1], 1])),
                                      tf.Variable(tf.random_normal([1])))
        self.prediction = fc(hidden_output, self.weights['output'], relu)

"""
Add fully connected layers to a passed network
    weights : a list containing weights in order of correspinding layers
    biases : a list containing biases in order of corresponding layers
"""
def add_fc(prev_layer, nonlin=relu, hidden_sizes=512, num_layers=2, weights=None):
    if type(hidden_sizes) == list:
        num_layers = len(hidden_sizes)
    else:
        hidden_sizes = [hidden_sizes]
    if len(hidden_sizes) == 1 and num_layers != 1:
        hidden_sizes = [hiddens_sizes for i in range(num_layers)]
    if weights == None:
        # Initialize weights
        weights = [(tf.Variable(tf.random_normal([hidden_sizes[i-1],hidden_sizes[i]])),
                    tf.Variable(tf.random_normal([hidden_sizes[i]]))) for i in range(1, num_layers)]
        weights.insert(0,(tf.Variable(tf.random_normal([in_dims,hidden_sizes[0]]))),
                          tf.Variable(tf.random_normal([hidden_sizes[0]]))) 

    # attach layers
    for i in range(num_layers):
        prev_layer = fc(prev_layer, weights[i], nonlin)

    return prev_layer, (weights, biases)
            
"""
Implements a fully connected layer with the given non_linearity
    w : either (h, b) or h , where h is the weight and b is the bias
"""
def fc(x, w, nonlin=lambda arg: pass):
    if type(w) != tuple:
        return nonlin(tf.matmul(x, w))
    else:
        return nonlin(tf.add(tf.matmul(x, w[0]), w[1]))
