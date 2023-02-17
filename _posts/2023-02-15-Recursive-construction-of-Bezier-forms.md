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

- **Lambda Functions**

These 'on the fly' functions will appear throughout the construction as we combine slices of multi-dimensional arrays together
into their new form. They will appear as $\lambda \theta\_i \theta\_{j, k}: \lambda t: \theta\_{i} + \theta\_{i}t + \theta\_{j, k}x\_{i}$ 
with the standard notation for scalars, vectors and matrices. 

- **Convolution Operator**
	* $C\_{m} \theta\_{lmn} \rightarrow \theta\_{ln}$

#### Construction
