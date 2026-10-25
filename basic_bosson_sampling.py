from collections import Counter
import gzip
import pickle #Serialization 
import time

import random

import perceval as pcvl
from perceval.algorithm import Sampler

n = 14       #number of photons at the input
m = 60       #number of modes
N = 5000000  #number of samplings

Unitary_60 = pcvl.Matrix.random_unitary(m) #creates a random unitary of dimension 60
mzi = (pcvl.BS() // (0, pcvl.PS(phi=pcvl.Parameter("φ_a")))
       // pcvl.BS() // (1, pcvl.PS(phi=pcvl.Parameter("φ_b"))))
pcvl.pdisplay(mzi)