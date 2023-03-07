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
\lambda_i t \triangleright \theta^{.i.}_{...}
$$

We wish to repeat this map until one component remains. Which indeed resolves to our desired convolutional operator. 

$$
\tag{2.0}
\lambda_i t \triangleright \lambda_i t \triangleright \lambda_i t \triangleright \dots \lambda_i t \triangleright \theta^{.i.}_{...} = \lambda_i t \overset{\|i\|}{\triangleright} \theta^{.i.}_{...} = C_{i} \theta^{.i.}_{...} = \theta^{..}_{..}
$$


#### Implementation (Python)
```python
from src.typeclass.VFF import VFF

from numpy import array, array_split, squeeze

class Bezier(VFF):
    def __init__(self, control_points, collapse_axes, callparam=lambda t:t):
        self.control_points = control_points
        self.collapse_axes = collapse_axes
        self.callparam = callparam

    def __call__(self, ts):
        ## Extract domain value and axis indicator.
        t, *ts = ts
        m, *ms = self.collapse_axes

        ## Along the given axis, gather sub-arrays from control points
        scp = array_split(self.control_points, self.control_points.shape[m], m)

        ## condense sub-arrays with convolution 
        while len(scp) > 1: 
            scp = [self.convolve(t,p,c) for p, c in zip(scp, scp[1:])]

        ## collect result of above computation and remove the condensed axis
        retv, *_ = scp
        retv = squeeze(retv, axis=m)

        ## recur computation if there're more axes to compress
        if ms:
            ms = list(map(lambda x: x if x < m else x - 1, ms))
            retv = Bezier(retv, ms, self.callparam).__call__(ts)

        return retv

    def convolve(self, t, slice_one, slice_two):
        return (1-t)*slice_one + t*slice_two
```

