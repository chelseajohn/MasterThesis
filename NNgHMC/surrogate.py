import numpy as np
import torch



class ShallowNet:
    """
    Shallow random projection neural network.

    The neural network has two sets of weights and biases: W1, b1, W2.
    W1 and b1 are randomized while W2 are trained with penalized regression.

    Attributes
        dim: dimension of input vector
        size: size of hidden layer
        W1, b1, W2: neural network parameters
    """

    def __init__(self, dim, size):
        self.dim = dim
        self.size = size
        self.W1 = (np.random.normal(0, 1, size * dim)).reshape((size, dim))
        self.b1 = np.random.normal(0, 1, size)
        self.W2 = np.zeros((size, 1))

    def __hidden(self, X):
        """
        Forward pass input to hidden layer.

        Args
            X: 2d numpy array (n x p) of input matrix

        Returns
            2d numpy array of hidden layer output
        """
        return(np.log(1.0 + np.exp(np.dot(X, np.transpose(self.W1)) + self.b1)))

    def train(self, X, y, penalty):
        """
        Train output weights with penalty.

        Args
            X: 2d numpy array (n x p) of training points
            y: 1d numpy array (n) of target vector
            penalty: weight of L2 penalty
        """
        H = self.__hidden(X)
        hat = np.dot(np.transpose(H), H) + np.identity(self.size) * penalty
        weights = np.dot(np.linalg.inv(hat), np.dot(np.transpose(H), y))
        self.W2 = weights

    def gradient(self, x):
        """
        Calculate gradient of neural network.

        Args
            x: 1d numpy array of input vector

        Returns
            1d numpy array of gradient vector
        """
        hidden_grad = 1.0 / (1.0 + np.exp(-(np.dot(self.W1, x) + self.b1)))
        return((np.dot(np.transpose(self.W2) * hidden_grad, self.W1)).reshape(self.dim))


class NeuralGrad:
    """
    Neural network gradient approximator in Keras.

    Attributes:
        network: pre-trained Keras model
        preprocessor: sklearn preprocessor 
    """

    def __init__(self, network, preprocessor):
        self.network = network
        self.preprocessor = preprocessor

    def gradient(self, x):
        """
        Predict gradient with neural network.

        Args
            x: 1d numpy array of input vector

        Returns
            1d numpy array of gradient vector
        """
        if x.shape[0] != 1:
            x = x.reshape(1, x.shape[0])

        x_scaled = self.preprocessor.transform(x)
        return(self.network.predict(x_scaled)[0])
    
    
class NeuralGrad_pytorch:
    """
    Neural network gradient approximator in Pytorch

    Attributes:
        network: pre-trained pytorch model
        preprocessor: sklearn preprocessor 
    """

    def __init__(self, network):
        self.network = network
      

    def gradient(self, x):
        """
        Predict gradient with neural network.

        Args
            x: 1d numpy array of input vector

        Returns
            1d numpy array of gradient vector
        """
        if x.shape[0] != 1:
            x = x.reshape(1, x.shape[0])
        xx = torch.from_numpy(x).double()
       
        return(self.network(xx)[0].detach().numpy())
    
    
