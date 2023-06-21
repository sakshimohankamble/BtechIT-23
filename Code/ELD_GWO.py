import random
import math    
import copy    
import sys
import pandas as pd
import numpy as np
from decimal import Decimal

# loss coefficient method or B-Coefficient
def loss_coefficient_method(position, B, Bi0, B00):
    Pl1 = float(0.0)
    Pl2 = float(0.0)
    
    for i in range(len(position)):
        for j in range(len(position)):
            Pl1 += B[i][j]*position[i]*position[j]
    
    for i in range(len(position)):
        Pl2 += Bi0[i]*position[i]
    
    Pl = Pl1 + Pl2 + B00
    return Pl

def Economic_Load_Dispatch(position, coefficient_matrix):
    fitness_value = float(0.0)

    for i in range(len(position)):
        fitness_value += coefficient_matrix[i][0]*math.pow(position[i],2) + coefficient_matrix[i][1]*position[i]+ coefficient_matrix[i][2]

    return fitness_value   

# wolf class
class wolf:
    def __init__(self, fitness, dim, minx, maxx, seed, B, Bi0, B00, Pd, coefficient_matrix, q1):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]
        sumP = float(0.0)

        for i in range(dim-1):
            self.position[i] = ((maxx[i] - minx[i]) * self.rnd.random() + minx[i])
            sumP+=self.position[i]

        self.position[dim-1] = Pd - sumP

        for i in reversed(range(1,dim)):
            if self.position[i] < minx[i]:
                self.position[i-1] += (self.position[i]-minx[i])
                self.position[i] = minx[i]

        loss = loss_coefficient_method(self.position,B, Bi0, B00)/float(dim)

        for i in range(dim):
            self.position[i] += q1*loss

        #print(self.position)

        self.fitness = fitness(self.position, coefficient_matrix) # curr fitness

# grey wolf optimization (GWO)
def gwo(fitness, max_iter, n, dim, minx, maxx, B, Bi0, B00, Pd, coefficient_matrix,q1):
    rnd = random.Random()
 
    # create n random wolves
    population = [ wolf(fitness, dim, minx, maxx, i, B, Bi0, B00, Pd, coefficient_matrix,q1) for i in range(n)]
    
    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key = lambda temp: temp.fitness)
 
    # best 3 solutions will be called as
    # alpha, beta and gaama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
 
 
    # main loop of gwo
    Iter = 0
    while Iter < max_iter:
 
        # after every 10 iterations
        # print iteration number and best fitness value so far
        #if Iter % 10 == 0 and Iter > 1:
            #print("Iter = " + str(Iter) + " best fitness = %.6f" % alpha_wolf.fitness)
 
        # linearly decreased from 2 to 0
        a = 2*(1 - Iter/max_iter)
 
        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
              2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2*rnd.random(), 2*rnd.random()
 
            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            
            for j in range(dim-1):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                  C1 - alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                  C2 -  beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                  C3 - gamma_wolf.position[j] - population[i].position[j])
                
                if X1[j] > maxx[j]:
                    X1[j] = maxx[j]
                elif X1[j] < minx[j]:
                    X1[j] = minx[j]
                    
                if X2[j] > maxx[j]:
                    X2[j] = maxx[j]   
                elif X2[j] < minx[j]:
                    X2[j] = minx[j]
                
                if X3[j] > maxx[j]:
                    X3[j] = maxx[j]  
                elif X3[j] < minx[j]:
                    X3[j] = minx[j]
                
                
                Xnew[j]+= X1[j] + X2[j] + X3[j]
            
            sumP = float(0.0)
            
            for j in range(dim-1):
                Xnew[j]/= 3.0
                sumP += Xnew[j]
            
            # determination of how much power should the last unit generate
            Xnew[dim-1] = Pd - sumP  
            
            for i in reversed(range(1,dim)):
                if Xnew[i] < minx[i]:
                    Xnew[i-1] += (Xnew[i]-minx[i])
                    Xnew[i] = minx[i]
            
            sum_loss = loss_coefficient_method(Xnew,B, Bi0, B00)
            # calculation avarege of loss transmission
            loss = sum_loss/float(dim)
            
            sumPowerofUnits = float(0.0)
            
            # avarege of loss transmission added to each unit 
            for j in range(dim):
                Xnew[j] += q1*loss  
                sumPowerofUnits += Xnew[j]
             
            # fitness calculation of new solution
            fnew = fitness(Xnew, coefficient_matrix)
    
            # greedy selection
            if fnew < population[i].fitness and (sum_loss + Pd) == sumPowerofUnits:
                population[i].position = Xnew
                population[i].fitness = fnew
                 
        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key = lambda temp: temp.fitness)
 
        # best 3 solutions will be called as
        # alpha, beta and gaama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
        
        #print(alpha_wolf.fitness)
         
        Iter+= 1
    # end-while
 
    # returning the best solution
    return alpha_wolf.position
  
if __name__ == '__main__':
    # Specifying number of iteration and population
    max_iter = 500
    population = 20
    q1 = 1.0
    
    # CASE1
    B1 = 0.01 * np.array([[0.0218, 0.0093, 0.0028],
                          [0.0093, 0.0228, 0.0017],
                          [0.0028, 0.0017, 0.0179]])

    B1i0 = 0.001 * np.array([0.3, 3.1, 1.5])

    B100 = 0.030523

    # a, b ,c
    coefficient_matrix1 = [[0.008, 7, 200],
                          [0.009, 6.3, 180],
                          [0.007, 6.8, 140]]

    Pd1 = 150

    minx1 = [10, 10, 10]
    maxx1 = [85, 80, 70]

    dim1 = 3

    fitness_list1 = []
    solution_list1 = []
    loss_list1 = []

    for i in range(20):

        fitness = gwo(Economic_Load_Dispatch, max_iter, population, dim1, minx1, maxx1, B1, B1i0, B100, Pd1, coefficient_matrix1, q1)
        solution_list1.append(fitness)
        fitness_list1.append(Economic_Load_Dispatch(fitness, coefficient_matrix1))
        loss_list1.append(loss_coefficient_method(fitness, B1, B1i0, B100))
    
    print("============================ CASE1 ============================")
    print("\nMin                : ",  '%.20e' % Decimal(min(fitness_list1)))
    print("Avarage            : ", '%.20e' % Decimal(np.mean(fitness_list1)))
    print("Standard Deviation : ", '%.20e' % Decimal(np.std(fitness_list1)))
    
    num = np.array(fitness_list1)
    reshaped = num.reshape(20,1)
    df = pd.DataFrame(reshaped, columns=["Economic Load Dispatch"])
    df.to_csv('case1.csv')
    
    # CASE2
    B2 = 0.0001*np.array([[0.14, 0.17, 0.15, 0.19, 0.26, 0.22],
                          [0.17, 0.60, 0.13, 0.16, 0.15, 0.20],
                          [0.15, 0.13, 0.65, 0.17, 0.24, 0.19],
                          [0.19, 0.16, 0.17, 0.71, 0.30, 0.25],
                          [0.26, 0.15, 0.24, 0.30, 0.69, 0.32],
                          [0.22, 0.20, 0.19, 0.25, 0.32, 0.85]])

    B2i0 = 0.001*np.array([-0.3908, -0.1297, 0.7047, 0.0591, 0.2161, -0.6635])

    B200 = 0.056

    # a, b ,c
    coefficient_matrix2 = [[0.007, 7, 240],
                           [0.005, 10, 200],
                           [0.009, 8.5, 220],
                           [0.009, 11, 200],
                           [0.008, 10.5, 220],
                           [0.0075, 12, 120]]
    Pd2 = 700

    minx2 = [100, 50, 80, 50, 50, 50]
    maxx2 = [500, 200, 300, 150, 200, 120]
    dim2 = 6

    fitness_list2 = []
    solution_list2 = []
    loss_list2 = []

    for i in range(20):

        fitness = gwo(Economic_Load_Dispatch, max_iter, population, dim2, minx2, maxx2, B2, B2i0, B200, Pd2, coefficient_matrix2, q1)
        solution_list2.append(fitness)
        fitness_list2.append(Economic_Load_Dispatch(fitness, coefficient_matrix2))
        loss_list2.append(loss_coefficient_method(fitness, B2, B2i0, B200))
    
    print("\n============================ CASE2 ============================")
    print("\nMin                : ",  '%.20e' % Decimal(min(fitness_list2)))
    print("Avarage            : ", '%.20e' % Decimal(np.mean(fitness_list2)))
    print("Standard Deviation : ", '%.20e' % Decimal(np.std(fitness_list2)))
    
    num = np.array(fitness_list2)
    reshaped = num.reshape(20,1)
    df = pd.DataFrame(reshaped, columns=["Economic Load Dispatch"])
    df.to_csv('case2.csv')