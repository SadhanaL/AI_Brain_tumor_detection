import logging
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format='(%(threadName)-9s) %(asctime)s %(message)s',
)

class CNNetwork:
    """
    Minimal CNN-like classifier:
      3x3 conv + ReLU + 2x2 maxpool + linear softmax (two classes).
    """

    def __init__(self, img_or_shape: np.ndarray | tuple) -> None:
        """
        Initialize the CNNetwork with input geometry and randomly initialized linear classifier weights.

        Parameters
        ----------
        img_or_shape : np.ndarray or tuple
            If a tuple (H, W) is given, a dummy image with that shape is used to establish internal dimensions.
            If a numpy array is given, its shape is used.

        Returns
        -------
        None
        """
        try:
            if isinstance(img_or_shape, tuple):
                H, W = map(int, img_or_shape)
                self._img = np.zeros((H, W), dtype=np.float64)
            else:
                self._img = np.array(img_or_shape, dtype=np.float64, copy=False)

            # Input image dimensions
            self.input_h, self.input_w = int(self._img.shape[0]), int(self._img.shape[1])

            # Network control variables (same as your provided code) 
            self._kernel = np.array([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]], dtype=np.float64)  
            self._conv_bias = -2.0

            # Feature map after valid 3x3 conv is (H-2, W-2), after 2x2 pool stride 2:
            pooled_h = (self._img.shape[0] - 2) // 2
            pooled_w = (self._img.shape[1] - 2) // 2
            self._input_layer_shape = int(pooled_h * pooled_w)

            # Softmax branch weights (two classes)
            rng = np.random.default_rng(seed=42) # reproducible
            self._input_weights_c1 = rng.standard_normal(self._input_layer_shape) / self._input_layer_shape
            self._input_weights_c2 = rng.standard_normal(self._input_layer_shape) / self._input_layer_shape
            rng.shuffle(self._input_weights_c1)
            rng.shuffle(self._input_weights_c2)
            self._input_bias_c1 = -2.0
            self._input_bias_c2 = 0.8

            # Working buffers
            self._feat_map = np.zeros((self._img.shape[0] - 2, self._img.shape[1] - 2), dtype=np.float64)
            self._rectified_feat_map = None
            self._max_pool = None
            self._max_pool_index_map = None
            self._input_layer = None
            self._output = None
            self._output_loss = None
            self._output_loss_gradient = None  # dl/dOut (only target class nonzero)
            self._input_layer_err = None
            self._max_pool_err = None
            self._conv_err = None

            self._learning_rate = 0.1
            self._weighted_sum_c1 = None
            self._weighted_sum_c2 = None
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CNNetwork: {e}") from e

    def calc_dot_product(self, _matrix1: np.ndarray, _matrix2: np.ndarray) -> float:
        """
        Compute the element-wise dot product (sum of products) between two matrices.

        Parameters
        ----------
        _matrix1 : np.ndarray
            First input matrix of shape (M, N).
        _matrix2 : np.ndarray
            Second input matrix of shape (M, N).

        Returns
        -------
        float
            The scalar dot product result.
        """
        try:
            assert _matrix1.shape == _matrix2.shape
            _dot_product = 0.0
            for i in range(_matrix1.shape[0]):
                for j in range(_matrix1.shape[1]):
                    _dot_product += _matrix1[i, j] * _matrix2[i, j]
            return _dot_product
        except Exception as e:
            raise ValueError(f"Error in calc_dot_product: {e}") from e

    def calc_2Dactivation_Relu(self, _data: np.ndarray) -> np.ndarray:
        """
        Apply element-wise ReLU activation.

        Parameters
        ----------
        _data : np.ndarray
            Input 2D array of shape (H, W).

        Returns
        -------
        np.ndarray
            Output 2D array of shape (H, W) with negative values set to zero.
        """
        try:
            _result = np.zeros((_data.shape[0], _data.shape[1]), dtype=np.float64)
            for i in range(_data.shape[0]):
                for j in range(_data.shape[1]):
                    if (_data[i,j]<=0):
                        _result[i,j] = 0
                    else:
                        _result[i,j] = _data[i,j]
            return _result
        except Exception as e:
            raise ValueError(f"Error in calc_2Dactivation_Relu: {e}") from e

    def calc_max_pool(self, _data: np.ndarray) -> np.ndarray:
        """
        Perform 2x2 max pooling with stride 2.

        Parameters
        ----------
        _data : np.ndarray
            Input 2D array of shape (H, W).

        Returns
        -------
        np.ndarray
            Output pooled 2D array of shape (H//2, W//2).
        """
        try:
            _result = np.zeros((int(_data.shape[0] / 2), int(_data.shape[1] / 2)), dtype=np.float64)
            self._max_pool_index_map = []
            for i in range(0, _data.shape[0] - 1, 2):
                for j in range(0, _data.shape[1] - 1, 2):
                    _max_pool_window = np.array([[ _data[i, j],     _data[i, j + 1]],
                                                [ _data[i + 1, j], _data[i + 1, j + 1] ]], dtype=np.float64)
                    _max_val = np.max(_max_pool_window)
                    _max_ind = np.unravel_index(_max_pool_window.argmax(), _max_pool_window.shape)
                    _result[int(i / 2), int(j / 2)] = _max_val
                    self._max_pool_index_map.append([_max_ind[0] + i, _max_ind[1] + j])
            return _result
        except Exception as e:
            raise ValueError(f"Error in calc_max_pool: {e}") from e

    def calc_conv_err(self) -> np.ndarray:
        """
        Map pooled error values back to their originating pre-pooled positions using the recorded argmax indices from 2x2 max-pooling.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            2D array with the same shape as the rectified feature map, containing propagated errors.
        """
        try:
            _conv_err = np.zeros((self._rectified_feat_map.shape[0], self._rectified_feat_map.shape[1]), dtype=np.float64)
            k = 0
            for i in range(self._max_pool_err.shape[0]):
                for j in range(self._max_pool_err.shape[1]):  
                    ii, jj = self._max_pool_index_map[k]
                    _conv_err[ii, jj] = self._max_pool_err[i, j]
                    k += 1
            return _conv_err
        except Exception as e:
            raise ValueError(f"Error in calc_conv_err: {e}") from e

    def calc_weighted_sum(self, _inputs: np.ndarray, _weights: np.ndarray) -> float:
        """
        Compute the weighted sum (dot product) of two 1D arrays.

        Parameters
        ----------
        _inputs : np.ndarray
            Input 1D array.
        _weights : np.ndarray
            Weights 1D array.

        Returns
        -------
        float
            Scalar weighted sum.
        """
        try:
            _result = 0.0
            for i in range(_inputs.shape[0]):
                _result += _inputs[i] * _weights[i]
            return _result
        except Exception as e:
            raise ValueError(f"Error in calc_weighted_sum: {e}") from e

    def calc_softmax(self, _data: np.ndarray) -> np.ndarray:
        """
        Compute a numerically stable softmax over the logits.

        Parameters
        ----------
        _data : np.ndarray
            1D array of logits.

        Returns
        -------
        np.ndarray
            1D array of probabilities that sum to 1.
        """
        try:
            z = _data.astype(np.float64)
            z = z - np.max(z)
            e = np.exp(z)
            return e / np.sum(e)
        except Exception as e:
            raise ValueError(f"Error in calc_softmax: {e}") from e

    def forward_prop(self, _image: np.ndarray, _target_class: int = None):
        """
        Perform the forward pass: convolution, ReLU, max-pool, linear, softmax.
        If _target_class is None, returns probabilities.
        If _target_class is given, returns (probs, loss, is_correct, grad_out).

        Parameters
        ----------
        _image : np.ndarray
            Input image array of shape (H, W).
        _target_class : int or None, optional
            Target class index for training, or None for inference.

        Returns
        -------
        np.ndarray
            If _target_class is None: predicted class probabilities of shape (2,).
        tuple[np.ndarray, float, int, np.ndarray]
            If _target_class is not None: (probs, loss, is_correct, grad_out).
        """
        try:
            print("Inside forward_prop")
            img = _image.astype(np.float64, copy=False)

            # Conv (valid 3x3) + bias 
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    win=np.array([[_image[i-1,j-1],_image[i-1,j],_image[i-1,j+1]],
                                [_image[i,j-1],_image[i,j],_image[i,j+1]],
                                [_image[i+1,j-1],_image[i+1,j],_image[i+1,j+1]]],
                                dtype=np.float64)
                    self._feat_map[i - 1, j - 1] = self.calc_dot_product(win, self._kernel) + self._conv_bias

            self._rectified_feat_map = self.calc_2Dactivation_Relu(self._feat_map)
            self._max_pool = self.calc_max_pool(self._rectified_feat_map)
            self._input_layer = self._max_pool.flatten()

            # Linear (two logits) + softmax
            self._weighted_sum_c1 = self.calc_weighted_sum(self._input_layer, self._input_weights_c1) + self._input_bias_c1
            self._weighted_sum_c2 = self.calc_weighted_sum(self._input_layer, self._input_weights_c2) + self._input_bias_c2
            self._output = self.calc_softmax(np.array([self._weighted_sum_c1, self._weighted_sum_c2], dtype=np.float64))

            if _target_class is None:
                # Inference mode
                return self._output

            # Training mode
            self._target_class = int(_target_class)
            eps = 1e-12
            p_t = max(float(self._output[self._target_class]), eps)
            self._output_loss = -np.log(p_t)

            _is_correct = int(np.argmax(self._output) == self._target_class)

            # dl/dOut
            self._output_loss_gradient = np.zeros(2, dtype=np.float64)
            self._output_loss_gradient[self._target_class] = -1.0 / p_t

            # grad_out (probs - one_hot) – back_propagate ignores it
            grad_out = self._output.copy()
            grad_out[self._target_class] -= 1.0

            return self._output, float(self._output_loss), _is_correct, grad_out
        except Exception as e:
            raise RuntimeError(f"Error in forward_prop: {e}") from e

    def back_propagate(self, grad_out=None):
        """
        Backpropagate through the final linear-softmax layer.
        Updates the linear layer weights and biases with SGD.

        Parameters
        ----------
        grad_out : None or np.ndarray, optional
            Gradient of the loss with respect to logits.

        Returns
        -------
        None
        """
        try:
            print ("Inside back_propagate")
            if grad_out is not None:
                dL_dt = grad_out  # p - one_hot
            else:
                # For each class i, compute dOut/dt and propagate
                weighted_sum_exp = np.exp(np.array([self._weighted_sum_c1, self._weighted_sum_c2], dtype=np.float64))
                Sum_of_exp = np.sum(weighted_sum_exp)

                # Jacobian row for the target step.
                dOut_dt = -weighted_sum_exp[:, None] * weighted_sum_exp[None, :] / (Sum_of_exp ** 2)
                # replace diagonal with the correct expression
                for i in range(2):
                    dOut_dt[i, i] = weighted_sum_exp[i] * (Sum_of_exp - weighted_sum_exp[i]) / (Sum_of_exp ** 2)

                # dL/dt = dL/dOut @ dOut/dt  (dL/dz= dL/dp*dp/dz)
                dL_dOut = self._output_loss_gradient  
                dL_dt = dL_dOut @ dOut_dt             

            # weighted_sum(t) = input · weights + bias
            weight_gradient = self._input_layer           # dt/dw
            bias_gradient = 1.0                           # dt/db
            input_gradient = np.vstack([self._input_weights_c1, self._input_weights_c2]).T  # dt/dinput 

            dL_dW = np.outer(dL_dt, weight_gradient) # dL/dW = (dL/dt) * (dt/dw)
            dL_dB = dL_dt * bias_gradient
            self._input_layer_err = input_gradient @ dL_dt  

            # SGD update
            del_weights = (self._learning_rate * dL_dW)
            self._input_weights_c1 -= del_weights[0]
            self._input_weights_c2 -= del_weights[1]

            del_biases = self._learning_rate * dL_dB
            self._input_bias_c1 -= del_biases[0]
            self._input_bias_c2 -= del_biases[1]

            # Back-map error shapes for potential conv backprop 
            self._max_pool_err = np.reshape(self._input_layer_err, self._max_pool.shape)
            self._conv_err = self.calc_conv_err()
        except Exception as e:
            raise RuntimeError(f"Error in back_propagate: {e}") from e

    def save(self, path: str):
        """
        Save model parameters to a compressed `.npz` archive.

        Parameters
        ----------
        path : str
            Filesystem path (including filename) where the `.npz` should be written.

        Returns
        -------
        None
        """
        try:
            np.savez_compressed(
                path,
                kernel=self._kernel.astype(np.float64),
                conv_bias=np.array(self._conv_bias, dtype=np.float64),
                w1=self._input_weights_c1.astype(np.float64),
                w2=self._input_weights_c2.astype(np.float64),
                b1=np.array(self._input_bias_c1, dtype=np.float64),
                b2=np.array(self._input_bias_c2, dtype=np.float64),
                input_h=np.array(self.input_h, dtype=np.int64),
                input_w=np.array(self.input_w, dtype=np.int64),
            )
        except Exception as e:
            raise RuntimeError(f"Error saving model to '{path}': {e}") from e

    @classmethod
    def load(cls, path: str) -> "CNNetwork":
        """
        Load model parameters from a `.npz` archive and return a reconstructed CNNetwork instance.

        Parameters
        ----------
        path : str
            Filesystem path to the `.npz` archive containing saved parameters.

        Returns
        -------
        CNNetwork
            A CNNetwork instance whose parameters are populated from the file.
        """
        try:
            data = np.load(path, allow_pickle=True)
            net = cls(img_or_shape=(int(data["input_h"]), int(data["input_w"])))
            net._kernel = data["kernel"].astype(np.float64)
            net._conv_bias = float(data["conv_bias"])
            net._input_weights_c1 = data["w1"].astype(np.float64)
            net._input_weights_c2 = data["w2"].astype(np.float64)
            net._input_bias_c1 = float(data["b1"])
            net._input_bias_c2 = float(data["b2"])
            return net
        except Exception as e:
            raise RuntimeError(f"Error loading model from '{path}': {e}") from e