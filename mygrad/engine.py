import numpy as np
from typing import Optional
import sys; sys.path.append("build")
import bten

# -------- autograd wrapper --------

class AGTensor:
    """
    Minimal autograd wrapper over bten.TensorF (and/or) bten.TensorU32.
    Forward ops: +, *, -, @, relu, mean, cross_entropy_loss
    non-differentiable ops: ==, argmax
    Backward: autodiff via backward() call.

    data is tensor that AGTensor wraps. It can be either bten.TensorF or bten.TensorU32 or a numpy array.
    If data is a numpy array, it will be converted to bten.TensorF or bten.TensorU32 based on its dtype.

    children is the set of AGTensors that this tensor's gradients should propagate to. Alternatively, you can 
    think of these children as the parents used to compute this AGTensor.

    op is the operation that produced this tensor, for debugging purposes only.

    requires_grad is a boolean flag indicating whether to track gradients for this tensor.

    is_cuda is an optional boolean flag indicating whether to store the data on GPU or CPU if data is a numpy array.
    """
    def __init__(self, data : Union[np.ndarray, bten.TensorT, bten.TensorU32], children=(), op='', requires_grad=True, is_cuda: Optional[bool]=None):
        # Lab-2: add your code to complete initialization of self.data and self.grad by replacing None with appropriate values.
        # You do not need to change other member variables' initialization.
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

        self.data = None
        self.grad = None
        

    @property
    def shape(self):
         return self.data.shape

    @property
    def is_cuda(self): 
        return self.data.is_cuda

    @property
    def T(self):
        """
        Transpose of a 2D tensor. This function is only partially filled. 
        You need to add the backward() function.
        """
        out = AGTensor(self.data.transpose(), (self,), 'T', requires_grad=(is_grad_enabled() and self.requires_grad))
        if out.requires_grad:
            def _backward():
                # Lab-2: add your code to compute self.grad (dIn) given out.grad (dOut).
                pass
            out._backward = _backward
        return out

    def __add__(self, other: Union[AGTensor, float, int]):
        """
        Elementwise addition with broadcasting: self + other
        other can be an AGTensor, float, or int.
        """
        # Lab-2: add your code here
        pass

    def __sub__(self, other: AGTensor):
        """Elementwise subtraction with broadcasting: self - other"""
        #Lab-2: add your code here
        pass

    def __mul__(self, other : Union[AGTensor, float, int]):
        """
        Elementwise multiplication with broadcasting: self * other
        other can be an AGTensor, float, or int.
        """
        # Lab-2: add your code here
        pass

    def __matmul__(self, other: AGTensor):
        """Matrix multiplication: self @ other"""
        # Lab-2: add your code here
        pass

    def relu(self):
        """
        Elementwise ReLU. 
        Forward: y = max(0, x)
        Backward: uses self.data.relu_back for d/dx.
        """
        # Lab-2: add your code here
        pass

    def sum(self, axis: Optional[int] = None):
        """
        Sum reduction. If axis is None, sum to scalar (1x1 tensor).
        If axis=0 or 1, reduce over that axis and keepdims.
        """
        #Lab-2: add your code here
        pass
    def mean(self):
        """Mean of all elements, returning a scalar AGTensor(1x1 tensor)."""
        # Lab-2: add your code here
        # Hint: It can be implemented using AGTensor's sum and * operation and therefore no need for separate backward logic.
        pass

    def cross_entropy_loss(self, targets: np.ndarray):
        """
        Calculate cross-entropy loss between self (logit Tensor) and targets (integer label tensor). 
        Returns a scalar AGTensor (1x1).
        """
        # Lab-2: add your code here. You should use self.data.cross_entropy_loss...
        pass
        
    def argmax(self):
        """
        Per-row argmax, returns an AGTensor of shape (N,1).
        This function is non-differentiable.
        """
        # Lab-2: add your code here"""
        pass
   
    def __eq__(self, other: Union[AGTensor, np.ndarray]):
        """
        Elementwise equality (with broadcasting).
        Returns a AGTensor with the broadcasted output shape whose elements are 1 (True) or 0 (False) 
        at the location where the correspond elements in self and other are equal.
        other can be an AGTensor or a NumPy array.
        This function is non-differentiable.
        """
        # Lab-2: add your code here
        pass

  
    #since I re-defined __eq__, I need to re-define __hash__
    __hash__ = object.__hash__


    # ----- backprop driver -----
    def backward(self, grad=None):
        """
        Backpropagate the gradient of the loss through the saved computation graph.
        Call this function on a scalar AGTensor (i.e., shape (1,1)).
        The optional grad argument is the initial gradient to be backpropagated.
        If grad is None, it defaults to a tensor of ones with the same shape as self.
        """
        # Lab-2: add your code here
        pass
         
    # Convenience: numpy view for debugging
    def numpy(self):
        return self.data.to_numpy()

    def __repr__(self):
        dev = "cuda" if self.is_cuda else "cpu"
        return f"Tensor(shape={self.data.shape}, device={dev}, op='{self._op}')"
