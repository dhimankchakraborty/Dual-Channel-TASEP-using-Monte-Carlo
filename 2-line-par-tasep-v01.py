import numpy as np
import numpy.random as npr
import random as rn
import matplotlib.pyplot as plt
from functions import *




L = 10
alpha = 0.3
beta = 0.6
w = 0
initial_filing = 0.1
mc_step_no = 100
therm_step_no = mc_step_no

state = initial_state_generator(L, initial_filing)


state = thermelization(L, therm_step_no, state, alpha, beta, w)

_ = simulation(L, mc_step_no, state, alpha, beta, w)
print("compilation done")
# print(density_arr)

L = 500
mc_step_no = 10**5
therm_step_no = mc_step_no

state = initial_state_generator(L, initial_filing)

state = thermelization(L, therm_step_no, state, alpha, beta, w)

print("thermelization done")

average_density = simulation(L, mc_step_no, state, alpha, beta, w)



print(np.round(average_density, 3))
