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

cost = [29.40199999999977, 55.60799999999975, 69.1879999999993, 75.95399999999943, 58.23599999999997, 63.727999999999454, 76.64599999999909, 68.73999999999944, 62.928000000000104]
boards = [228.82, 228.34, 212.74, 212.9, 166.2, 155.38, 170.0, 143.44, 131.72]


cost1= [17.429999999999776, 56.44799999999994, 98.45400000000005, 139.56199999999998, 138.83799999999997, 166.746, 179.166, 175.09200000000004, 200.0]
boards1 = [191.68, 221.46, 308.56, 381.6, 370.5, 433.26, 462.02, 450.34, 500.0]

cost2  = [54.284999999999734, 86.56299999999963, 111.38799999999938, 111.48699999999948, 105.5279999999994, 84.54799999999904, 76.93799999999926, 74.60799999999944, 71.62399999999963]
board3 = [280.78, 272.1, 301.39, 288.02, 260.46, 197.49, 169.48, 159.96, 145.97]

cost3 = [98.54199999999966, 131.28999999999922, 118.39799999999954, 113.78599999999955, 118.16799999999925, 72.99399999999932, 71.05799999999925, 65.36599999999984, 68.95199999999971]
boards3 = [331.5, 317.92, 286.78, 269.96, 262.72, 173.46, 158.68, 142.34, 139.98]





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



food_t, food_p = ttest_pvalue(cost, cost1)
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
scatter_boiii(cost1, 'Two food sources', 'two_food_boisss.png')
scatter_boiii(cost2, 'Higher pheromone strength', 'high_on_pheromone.png')
scatter_boiii(cost3, 'Higher fading rate of pheromone', 'fading.png')
