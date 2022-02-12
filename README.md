# C/Ginv Version 0.1
Continuation/Generalized-inverse method

## Introduction

**`C/Ginv`** is a nonlinear model predictive controller whose nonlinear model predictive control (NMPC) problem DOES NOT employ cost functions, therefore having NO WEIGHT PARAMETERS TO TUNE and computationally light. With a 2.3GHz 2019 Macbook Pro, a computation time less than 1ms can be achieved with C/Ginv about certain systems, with Python language.

The purpose of the Python codes in this github repository is to provide simulation examples of C/Ginv so that people can experience the easy-to-use handiness and the computation speed of this method.

<!--
The codes DO NOT put emphasis on taking full advantage of Python language for speed, but rather concentrates on the readability for easy translation to different programing languages such as C, C++, etx. Therefore, one maybe able to implement a faster code than the ones found in this repository.
-->

## Notes 
* Works with almost any state equations of the form dx/dt=f(t,x,u).
* Quick rise time and high convergence rate.
* Constraints are not considered yet in Version 0.
* Uses only numpy and no other special packages.


## Installation

1. Clone or download CGinv.py and TestCGinv_*.py files.
2. Make sure CGinv.py file is in your working directory.

 
## Requirements

* Python ver. 3 (Python ver.2 not confirmed)
* numpy

## Getting Started (Python ver.3)

1. Put CGinv.py in the same directory with TestCGinv*.py files
``` Shell or Command line
>> python [filename].py
```

## Citing C/Ginv


#coming soon.


## License

C/Ginv is distributed under the BSD 2-Clause "Simplified" License.
