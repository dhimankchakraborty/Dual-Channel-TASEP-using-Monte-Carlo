import numpy as np
import numpy.random as npr
import random as rn
from numba import jit, njit, prange




def initial_state_generator(N, initial_filing):
    state = np.zeros((2, N), dtype=np.int8)
    pos_arr_1 = list(range(N))
    pos_arr_2 = list(range(N))
    initial_filing_no = int(N * initial_filing)

    rn.shuffle(pos_arr_1)
    rn.shuffle(pos_arr_2)
    # print(pos_arr)

    for i in range(initial_filing_no):
        j1 = pos_arr_1.pop(0)
        state[0][j1] = 1
        j2 = pos_arr_2.pop(0)
        state[1][j2] = 1

    return state



@njit
def thermelization(L, therm_step_no, state, alpha, beta, w):

    for k in prange(therm_step_no):
        for l in prange(2 * L):

            # print(k, l)

            # new_state = state.copy()

            i = npr.randint(2)
            j = npr.randint(L)
            # print(i, j)

            if j == 0 and state[i][j] == 0:
                r = npr.random()
                if r < alpha:
                    state[i][j] = 1
            
            elif j == L - 1 and state[i][j] == 1:

                if state[i - 1][j] == 1:
                    r = npr.random()
                    if r < beta:
                        state[i][j] = 0
                
                else:
                    r = npr.random()
                    if r < (beta * (1 - w)):
                        state[i][j] = 0
                    else:
                        state[i][j] = 0
                        state[i -1][j] = 1
            
            elif state[i][j] == 1:

                if state[i - 1][j] == 1 and state[i][j + 1] == 0:
                    state[i][j] = 0
                    state[i][j + 1] = 1
                
                elif state[i - 1][j] == 0:
                    r = npr.random()
                    if r < w:
                        state[i][j] = 0
                        state[i - 1][j] = 1
                    
                    elif state[i][j + 1] == 0:
                        state[i][j] = 0
                        state[i][j + 1] = 1

    return state



@njit
def simulation(L, mc_step_no, state, alpha, beta, w):

    density_arr = []
    for k in prange(mc_step_no):
        for l in prange(2 * L):
            # print(k, l)

            # new_state = state.copy()

            i = npr.randint(2)
            j = npr.randint(L)
            # print(i, j)

            if j == 0 and state[i][j] == 0:
                r = npr.random()
                if r < alpha:
                    state[i][j] = 1
            
            elif j == L - 1 and state[i][j] == 1:

                if state[i - 1][j] == 1:
                    r = npr.random()
                    if r < beta:
                        state[i][j] = 0
                
                else:
                    r = npr.random()
                    if r < (beta * (1 - w)):
                        state[i][j] = 0
                    else:
                        state[i][j] = 0
                        state[i -1][j] = 1
            
            elif state[i][j] == 1:

                if state[i - 1][j] == 1 and state[i][j + 1] == 0:
                    state[i][j] = 0
                    state[i][j + 1] = 1
                
                elif state[i - 1][j] == 0:
                    r = npr.random()
                    if r < w:
                        state[i][j] = 0
                        state[i - 1][j] = 1
                    
                    elif state[i][j + 1] == 0:
                        state[i][j] = 0
                        state[i][j + 1] = 1

        if (k + 1) % 10 == 0:
            density = (state[0].sum() + state[1].sum()) / (2 * L)
            density_arr.append(density)

    average_density = np.array(density_arr).mean()

    return average_density
