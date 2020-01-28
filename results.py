import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import seaborn as sns

import grid
import ants

'''
This function collects the data for the different ratios of ants. It runs the
simulation 100 times for every ratio from 10:0 to 0:10 workers:searchers.
'''

def make_data(grid, strength, fade):
    # initialize the data list and start values of worker and search ants
    data_cost = []
    data_board = []

    search = 1
    work = 10

    # run the test for 10 different ratios
    for i in range(10):

        total_cost = 0
        total_board = 0

        # determine the amount of boards and the cost of 100 iterations
        for i in range(100):
            # make the environment for the simulation
            world = Grid([grid, strength, fade, search, work])
            world.setNestLocation((14,3))
            world.setFoodSource((2,1), 6)
            world.setFoodSource((11,18), 6)
            #world.setFoodSource((8,8), 6)


            print(i)
            print("------------------------")

            cost, board = world.simulation()
            if cost == 0 and board == 0:
                continue
            total_cost += cost
            total_board += board


        # add the average to a the data list
        data_cost.append(total_cost/100)
        data_board.append(total_board/100)

        # change the ratio of work:search ants for next iteration
        search += 1
        work -= 1

    return data_cost, data_board


'''
This function will plot the standard deviations of 2 populations. The
input is two arrays containing tuples.
'''
def plot_distributions(pop1, pop2, term1, term2):
    # plot the histograms and distibution of the data
    sns.distplot(pop1)
    sns.distplot(pop2)

    # add labels
    plt.title('Distribution of cost per worker:searcher ants')
    plt.xlabel('Ratio workers:searchers (%)')
    plt.legend(labels=['Population ' + term1,'Population ' + term2])
    plt.ylabel('Cost')
    plt.show()


'''
This function returns the t-test value and the p-value between two populations.
'''
def ttest_pvalue(pop1, pop2):
    ttest = stats.ttest_ind(pop1, pop2)
    return ttest[0], ttest[1]



#plot_distributions(a, b, 'with 1 food source', 'with 2 food source')


#print(ttest_pvalue(a,b))
