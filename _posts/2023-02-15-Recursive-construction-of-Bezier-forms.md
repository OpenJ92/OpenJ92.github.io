---
layout: post
title:  "Recursive construction of Bezier Forms"
categories: jekyll update
---

<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

Here we'll describe a method for forming Bezier Forms through a series of 'convolutions' over n-dimensional arrays. With a clear mathematical 
notation and construction, we'll move to implementation in Python and Rust.

#### Notation

We'll be working with a collection of entities whose representations are defined here.

- **Multidimensional Array**

n-dimensional containers will be labeled $\theta\_{abc \ldots n}$ where each subscript correspond to an additional dimension.
For example, the symbol $\theta\_{abc}$ represents the 3-dimensional container. Supposing that $a, b, c \in [0,1]$, then we'll
have constructed an object with eight components. Accessing an individual component will require resolution of all subscripts
to a specific sequence of integers. ie $\theta\_{000}$. 

An incomplete sequence of integers will corespond to what's known as a slice. That is to say, if we're to write $\theta^{a..}\_{.bc}$ 
we will resolve a (3-1)-dimensional container with free variables $b$ and $c$. In our example, there would be two such slices in
the item, $\theta^{0..}\_{.bc}$ and $\theta^{1..}\_{.bc}$.

Scalar multiplication and addition between identically size arrays are needed for this construction. Much like vector and matrix algebra, 
it will be defined as a broadcast multiplication and componentwise addition. 

- **Lambda Functions**

These 'on the fly' functions will appear throughout the construction as we combine slices of multi-dimensional arrays together
into their new form. They will appear as $\lambda \theta\_i \theta\_{j, k}: \lambda t: \theta\_{i} + \theta\_{i}t + \theta\_{j, k}x\_{i}$ 
with the standard notation for scalars, vectors and matrices. 

- **Convolution Operator**

A convolution operator denoted $C\_{m}$ is a motion from an array of one dimensionality to another $\theta_{\Pi a} \rightarrow \theta\_{\Pi b}$. 
For our purpose, the operation will reduce the dimensionality of the array along the specified index $C\_{\phi} \theta\_{\phi \Pi a} = \theta\_{\Pi a}$.
A composition of convolutions $C\_{m} \circ C\_{n}$ will be denoted by $C\_{mn}$

#### Construction

Firstly, we'll define a function $\beta$ which accepts a multidimensional array $\theta\_{\Pi b\_{k}}$, a vector of dimensions $b\_{i}$ and a vector from 
the domain $v\_{i}$. 

$$ 
\tag{1.1}
\beta (\theta_{\Pi b_{k}}, b_{i}, v_{i}) = C_{\Pi b_{i}} \theta_{\Pi b_{k}} (v_{i})
\label{eq:1p1}
$$

$$
\tag{1.2}
\beta (\theta_{\Pi b_{k}}, b_{i}, v_{i}) = \theta_{\frac{\Pi b_{k}}{\Pi b_{i}}} (v_{i})
\label{eq:1p2}
$$

A motion from $\eqref{eq:1p1}$ to $\eqref{eq:1p2}$ come through repeatedly consuming a convolutional operator and vector component on either side of our
multidimensional array. Their combination reduces the dimension of the array as mentioned above. The process continues until all components are consumed.

We now define the implementation of our operator $C\_{b\_i}$. It itself is a composition of operations over the array. Firstly, we'll define $\lambda_i$ 
which combines adjacent slices of $\theta$ into a singular slice. 

$$
\tag{1.3}
 \lambda_i t \theta_{i} \theta_{i+1} : (1 - t)\theta_{i} + t\theta_{i+1}
$$

A map of this function over the given dimension will procude a new array that's been reduced by 1 in the given dimension. 

$$
\tag{1.4}
\lambda_i t \triangleright \theta
$$

We wish to repeat this map until one component remains. Which indeed resolves to our desired convolutional operator. 

$$
\tag{2.0}
\lambda_i t \triangleright \lambda_i t \triangleright \lambda_i t \triangleright \dots \lambda_i t \triangleright \theta = \lambda_i t \overset{\|i\|}{\triangleright} \theta = C_{i}
$$


#### Implementation (Python)
```python
from src.typeclass.VFF import VFF

from numpy import array, concatenate, stack

class Bezier(VFF):
    def __init__(self
                , shape_in: array
                , shape_out: array
                , control_points: array
                , callparam=lambda t:t
                ):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.control_points = control_points

    @classmethod
    def make_closed(self
                   , _spine: array
                   , _loop: array
                   , _control_points: array
                   ):
        pass

    @classmethod
    def make_random(self):
        pass

    @classmethod
    def make_random_closed(self):
        pass

    def __call__(self, t):
        return self.evaluate(t)

    def evaluate(self, t):
        t = self.callparam(t)
        convolve = lambda t, c1, c2: (1-t)*c1 + t*c2
        a = [
                self.shape_in.reshape(self.control_points.shape[0],1),
                self.control_points,
                self.shape_out.reshape(self.control_points.shape[0],1)
            ]
        temp_control = concatenate(a, axis=1)
        while temp_control.shape[1] > 1:
            A = [
                    convolve(t, temp_control[:, i], temp_control[:, i+1])
                    for i
                    in range(temp_control.shape[1] - 1)
                ]
            temp_control = stack(A, axis = 1)
        return tuple(temp_control.T.reshape(self.control_points.shape[0]))

    def _evaluate(self, t, arr):
        pass

    def split(self, t):
        return [ Bezier( self.shape_in
                       , self.shape_out
                       , self.control_points
                       , callparam=lambda nt: t*nt
                       )
               , Bezier( self.shape_in
                       , self.shape_out
                       , self.control_points
                       , callparam=lambda nt: (1-t)*nt+t
                       )
               ]
```

