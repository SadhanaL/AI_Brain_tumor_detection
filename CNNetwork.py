# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:20:05 2022

@author: Sadhana L
"""
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(asctime)s %(message)s',)

class CNNetworkError(IOError):
    """
    CNN Error class.

        - Base class for CNN Errors.
        - Inherits from IOError Class.¨
    """
    
class CNNetwork():
    def __init__(self, img):
        self._img = img         # a sample image from the dataset to initialize values in correct shape.
        # Network control variables
        # self._kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])  #accuracy:66
        # self._kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #accuracy:64
        self._kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) #accuracy:65
        self._conv_bias = -2
        
        self._input_layer_shape = int(((self._img.shape[1] - 2)/2)**2)
        # Uncomment for sigmoid
        #self._input_weights = np.random.uniform(-1,1,self._input_layer_shape)
        #np.random.shuffle(self._input_weights)
        #self._input_bias = 1
        ##
        #Uncomment for Softmax
        self._input_weights_c1 = np.random.randn(self._input_layer_shape)/self._input_layer_shape
        self._input_weights_c2 = np.random.randn(self._input_layer_shape)/self._input_layer_shape #to reduce the variance
        # self._input_weights_c1 = np.random.uniform(-1,1,self._input_layer_shape)
        # self._input_weights_c2 = np.random.uniform(-1,1,self._input_layer_shape)
        np.random.shuffle(self._input_weights_c1)
        np.random.shuffle(self._input_weights_c2) 
        self._input_bias_c1 = -2 #-0.5
        self._input_bias_c2 = 0.8 #1
        ##
        self._feat_map = np.zeros((self._img.shape[0]-2,self._img.shape[1]-2))
        self._rectified_feat_map = None
        self._max_pool = None
        self._max_pool_index_map = None
        self._input_layer = None
        self._output = None
        self._output_loss = None
        self._input_layer_err = None
        self._max_pool_err = None
        self._conv_err = None
        
        self._learning_rate = 0.1
        self._weighted_sum_c1 = None
        self._weighted_sum_c2 = None
    
    def calc_dot_product(self, _matrix1, _matrix2):
        """
        Generates dot product between two matrices of same size. i.e., given data 
        and the kernel.

        Parameters
        ----------
        _matrix1 : np array
            Array 1 of size N. The chosen.
        _matrix2 : np array
            Array 2 of size N.

        Returns
        -------
        dot_product : The 
            DESCRIPTION.

        """
        assert(_matrix1.shape == _matrix2.shape) # Assering if both arrays are of same size.
        
        _dot_product=0                      # initializing the result 
        
        # Loops for selecting one element at a time. We use the shape of array 1 as
        # a reference.
        for i in range(0,_matrix1.shape[0]):
            for j in range(0, _matrix1.shape[1]):
                # Performing element by element mulitplication
                _dot_product += _matrix1[i,j] * _matrix2[i,j]
        return _dot_product
    
    def calc_2Dactivation_Relu(self, _data):
        """
        Calculate ReLu function output on the given 2D matrix.

        Parameters
        ----------
        _data : np array
            2D matrix.

        Returns
        -------
        result : np array
            DESCRIPTION.

        """
        _result = np.zeros((_data.shape[0],_data.shape[1]))
        for i in range(0,_data.shape[0]):
            for j in range(0,_data.shape[1]):
                if (_data[i,j]<=0):
                    _result[i,j] = 0
                else:
                    _result[i,j] = _data[i,j]
        return _result
    
    def calc_1Drelu(self, _data):
        """
        Calculate Relu function output for the given data.

        Parameters
        ----------
        _data : TYPE
            DESCRIPTION.

        Returns
        -------
        _data : TYPE
            DESCRIPTION.

        """
        if _data <= 0:
            _data = 0
        else:
            _data = _data
        return _data
    
    def calc_max_pool(self, _data): 
        """
        Generate Max pool matrix from the given 2D matrix

        Parameters
        ----------
        _data : np array
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        _result = np.zeros((int(_data.shape[0]/2),int(_data.shape[1]/2)))
        self._max_pool_index_map = list()
        for i in range(0,_data.shape[0]-1,2):
            for j in range(0,_data.shape[1]-1,2):
                # pooling with a 2x2 max_pool window with a stride of 2
                _max_pool_window = np.array([[_data[i,j],_data[i,j+1]],
                                             [_data[i+1,j],_data[i+1,j+1]]])
                _max_val = np.max(_max_pool_window)
                _max_ind = np.unravel_index(_max_pool_window.argmax(), 
                                            _max_pool_window.shape)
                _result[int(i/2),int(j/2)] = _max_val #storing the max value in the pooled matrix
                self._max_pool_index_map.append([_max_ind[0] + i,
                                                 _max_ind[1] + j])
        return _result
        
    def calc_conv_err(self):
        _conv_err = np.zeros((self._rectified_feat_map.shape[0], 
                             self._rectified_feat_map.shape[1]))
        k = 0                   # Iterator for max pool index map
        for i in range(0, self._max_pool_err.shape[0]):
            for j in range(0, self._max_pool_err.shape[0]):
                _conv_err[self._max_pool_index_map[k][0],
                          self._max_pool_index_map[k][1]] = self._max_pool_err[i,j] #backpropagate the maxpool error to the conv layer
                k += 1
        return _conv_err
    
    def calc_weighted_sum(self, _inputs, _weights):
        """
        Calculate weighted sum from the set of given inputs and weights.

        Parameters
        ----------
        _inputs : np array
            1D .
        _weights : TYPE
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        _result = 0
        for i in range(0, _inputs.shape[0]):
            _result += _inputs[i]*_weights[i]
        return _result
    
    def calc_sigmoid(self, _data):
        """
        Calculate sigmoid function output for the given data

        Parameters
        ----------
        _data : int or float
            DESCRIPTION.

        Returns
        -------
        _result : float
            DESCRIPTION.

        """
        _result = (1/(1+np.exp(-_data)))
        return _result
    
    def calc_softmax(self, _data):
        """
        

        Parameters
        ----------
        _data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        return np.exp(_data)/sum(np.exp(_data))
    
    def forward_prop(self, _image, _target_class):
        self._target_class = _target_class
        for i in range (1,_image.shape[0]-1): #creating a 3x3 window to generate a feature map
            for j in range (1, _image.shape[1]-1):
                win=np.array([[_image[i-1,j-1],_image[i-1,j],_image[i-1,j+1]],
                              [_image[i,j-1],_image[i,j],_image[i,j+1]],
                              [_image[i+1,j-1],_image[i+1,j],_image[i+1,j+1]]])
                self._feat_map[i-1,j-1] = self.calc_dot_product(win, self._kernel) + self._conv_bias
        self._rectified_feat_map = self.calc_2Dactivation_Relu(self._feat_map)
        
        self._max_pool = self.calc_max_pool(self._rectified_feat_map)
        self._input_layer = self._max_pool.flatten()
        
        
        
        ## Uncomment for Sigmoid
        # _weighted_sum = self.calc_weighted_sum(self._input_layer, 
        #                                       self._input_weights) + self._input_bias
        # #network_sigmoid = calc_sigmoid(Weighted_sum)
        # #output = (output_weight *network_sigmoid) + output_bias   
        # # Removing the hidden layer
        # self._output = self.calc_sigmoid(_weighted_sum)
        # self._output_loss = self._output*(1-self._output)*(_target_val-self._output) 
        ## 
        ## Uncomment for Softmax
        self._weighted_sum_c1 = self.calc_weighted_sum(self._input_layer, 
                                                       self._input_weights_c1) + self._input_bias_c1
        self._weighted_sum_c2 = self.calc_weighted_sum(self._input_layer, 
                                                       self._input_weights_c2) + self._input_bias_c2
        self._output = self.calc_softmax(np.array([self._weighted_sum_c1,
                                                   self._weighted_sum_c2]))
        # print("\n WS1: \n {}".format(_weighted_sum_c1))
        # print("\n WS2: \n {}".format(_weighted_sum_c2))
        # print("\n Output: \n {}".format(self._output))
        
        
        self._output_loss = -np.log(self._output[_target_class])
        ##
        if np.argmax(self._output) == _target_class:
            _is_correct = 1
        else:
            _is_correct = 0
        self._output_loss_gradient = np.zeros(2) #dl/dout = 0 for i != c
        self._output_loss_gradient[_target_class] = -1/self._output[_target_class] #dl/dout for i = c
        return self._output, self._output_loss, _is_correct
    
    def back_propagate(self):
        for i, self._output_loss_gradient in enumerate(self._output_loss_gradient):
            if self._output_loss_gradient == 0: 
                continue
            weighted_sum_exp = np.exp(np.array([self._weighted_sum_c1,
                                                self._weighted_sum_c2]))
            Sum_of_exp = np.sum(weighted_sum_exp)
            
        
            dOut_dt = -weighted_sum_exp[i]*weighted_sum_exp/(Sum_of_exp**2) #dout/dtk for k!= c
            dOut_dt[i] = weighted_sum_exp[i]*(Sum_of_exp-weighted_sum_exp[i])/(Sum_of_exp**2) #dout/dtc for k = c
            # weighted_sum(t) = input * weights + bias
            weight_gradient = self._input_layer #dt/dw
            bias_gradient = 1 # dt/db
            input_gradient = np.array([self._input_weights_c1,
                                       self._input_weights_c2]).T # dt/dinput
            
            dL_dWeighted_sum = self._output_loss_gradient*dOut_dt #dL/dt = dL/dout*dout/dt
            #newaxis adds an extra dimension to dL/dt and dt/dw as both are 1d array as we are trying to get 2d array for matrix mul
            dL_dW = weight_gradient[np.newaxis].T @ dL_dWeighted_sum[np.newaxis] #dL/dw = dL/dt*dt/dw
            dL_dB = dL_dWeighted_sum*bias_gradient # dL/db = dL/dt * dt/db
            self._input_layer_err = input_gradient @ dL_dWeighted_sum #dL/dInput = dL/dt * dt/dInput
            
            del_weights = (self._learning_rate*dL_dW).T
            self._input_weights_c1 -= del_weights[0]
            self._input_weights_c2 -= del_weights[1]
            
            del_biases = self._learning_rate*dL_dB
            self._input_bias_c1 -= del_biases[0]
            self._input_bias_c2 -= del_biases[1]
            

        self._max_pool_err = np.reshape(self._input_layer_err, self._max_pool.shape)
        self._conv_err = self.calc_conv_err()
        
        # Uncomment for Sigmoid
        # self._input_layer_err = np.zeros(self._input_layer.shape[0])
        # del_input_bias = self._learning_rate * self._output_loss
        # self._input_bias += del_input_bias
        # for i in range(0,self._input_weights.shape[0]):
        #     del_input_weight = self._learning_rate * self._output_loss * self._input_layer[i]
        #     self._input_weights[i] += del_input_weight
        #     # Calculating error on the input layer
        #     self._input_layer_err[i] = self._input_layer[i] * (1-self._input_layer[i]) * self._output_loss * self._input_weights[i]
        #     self._max_pool_err = np.reshape(self._input_layer_err, self._max_pool.shape)
        #     self._conv_err = self.calc_conv_err()
        ##
    