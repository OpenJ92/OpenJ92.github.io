---
layout: post
title:  "Recursive construction of Hyperspheres"
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

#### Implementation (Python)
```python
from src.typeclass.VFF import VFF
from src.VFF.blade import Blade

from math import sin, cos
from numpy import hstack, array

class Sphere(VFF):
    def __call__(self, ts):
        cach = { t : { sin: sin(t), cos: cos(t) } for t in ts }
        return self.recursive_call(cach, array([1]), ts)

    def recursive_call(self, cach, arr, ts):
        t, *ts = ts
        arr = hstack((cach[t][cos]*arr, cach[t][sin]))
        if ts: return self.recursive_call(cach, arr, ts)
        else : return arr

```

