import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import choice
import copy
from matplotlib import cm
import matplotlib
import matplotlib.image as mpimg
import matplotlib.animation as animation
from math import *
import queue
from scipy import stats
import seaborn as sns
from drawnow import drawnow, figure
import time
from statistics import stdev

from scipy.stats import norm

cost = [35.1629999999999, 65.93599999999932, 60.98299999999982, 63.50499999999972, 58.5399999999997, 81.97399999999936, 71.29199999999955, 64.08199999999961, 69.51199999999966]

cost2 = [62.30099999999966, 98.12499999999922, 110.58899999999875, 102.2749999999997, 102.4589999999991, 88.06799999999899, 67.72599999999944, 70.39599999999965, 70.13599999999973]
cost3 = [97.85099999999964, 109.52099999999947, 105.75499999999964, 110.6309999999996, 115.39099999999937, 89.23599999999898, 79.57199999999912, 67.2249999999997, 65.99899999999984]






'''
This function will plot the standard deviations of 2 populations. The
input is two arrays containing tuples.
'''
def plot_distributions(pop1, pop2, term1, term2, titel):
    # plot the histograms and distibution of the data
    sns.distplot(pop1)
    sns.distplot(pop2)

    # add labels
    plt.title('Distribution of cost per worker:searcher ants')
    plt.xlabel('Ratio workers:searchers (%)')
    plt.legend(labels=['Population ' + term1,'Population ' + term2])
    plt.ylabel('Cost')
    plt.savefig(titel)
    plt.show()


'''
This function returns the t-test value and the p-value between two populations.
'''
def ttest_pvalue(pop1, pop2):
    ttest = stats.ttest_ind(pop1, pop2)
    return ttest[0], ttest[1]



#food_t, food_p = ttest_pvalue(cost, cost1)
strength_t, strength_p = ttest_pvalue(cost, cost2)
fade_t, fade_p = ttest_pvalue(cost, cost3)

# print(food_t)
# print(food_p)
#
# print(strength_t)
# print(strength_p)
#
# print(fade_t)
# print(fade_p)

# print("The t-value for food:" + food_t + "The p-value for food" + food_p)
# print("The t-value for strength:" + strength_t + "The p-value for strength" + strength_p)
# print("The t-value for fade:" + fade_t + "The p-value for fade" + fade_p)
#
#plot_distributions(cost, cost1, 'with one food source', 'with two food sources', 'diff_food_source.png')
# plot_distributions(cost, cost2, 'strength of 0.1', 'strength of 0.2', 'diff_pher_strength.png')
# plot_distributions(cost, cost3, 'fade of 0.005', 'fade of 0.01', ' diff_pher_fade.png')



def scatter_boiii(cost, term, titel):
    x = list(range(1,10))
    plt.scatter(x, cost)
    plt.plot(x,cost, color = 'orange')
    plt.title('Distribution of cost per worker:searcher ants')
    plt.xlabel('Ratio workers:searchers (%)')
    plt.legend(labels=['Population ' + term])#,'Population ' + term2])
    plt.ylabel('Cost')
    plt.savefig(titel)
    plt.show()

scatter_boiii(cost, 'Baseline', 'baseline.png')
#scatter_boiii(cost1, 'Two food sources', 'two_food_boisss.png')
scatter_boiii(cost2, 'Higher pheromone strength', 'high_on_pheromone.png')
scatter_boiii(cost3, 'Higher fading rate of pheromone', 'fading.png')
