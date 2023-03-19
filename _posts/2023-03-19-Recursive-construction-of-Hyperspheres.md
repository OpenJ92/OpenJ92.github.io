---
layout: post
title:  "Recursive construction of hyper-spheres"
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

Here we describe an algorithm that enables one to sample hyper-spheres of any dimension. 

$$\begin{align}
\xi_i & = cos(\theta_i)\xi_{i-1} \triangledown sin(\theta_i) \\
\xi_1 & = 1
\end{align}$$

#### Implementation (Python)
```python
from src.typeclass.VFF import VFF

from math import sin, cos
from numpy import hstack, array

class Sphere(VFF):
    def __call__(self, ts):
        return self.recursive_call(array([1]), ts)

    def recursive_call(self, arr, ts):
        t, *ts = ts
        arr = hstack((cos(t)*arr, sin(t)))
        if ts: return self.recursive_call(arr, ts)
        else : return arr
```

