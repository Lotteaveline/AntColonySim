import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import seaborn as sns

from grid import Grid
from ants import Ant

'''
This function collects the data for the different ratios of ants. It runs the
simulation 100 times for every ratio from 10:0 to 0:10 workers:searchers.
'''
def make_data(grid, strength, fade, food_sources):
    # initialize the data list and start values of worker and search ants
    data_cost = []
    data_board = []

    search = 1
    work = 9

    # run the test for 10 different ratios
    for i in range(9):

        total_cost = 0
        total_board = 0

        # determine the amount of boards and the cost of 100 iterations

        amount_iterations = 100
        for i in range(amount_iterations):
            # make the environment for the simulation
            world = Grid([grid, strength, fade, search, work])

            for food_source in food_sources:
                world.setFoodSource(food_source, 6)

            world.setNestLocation((14,3))

            #world.setFoodSource((8,8), 6)


            print(i)
            print("------------------------")

            cost, board = world.simulation()
            if cost == 0 and board == 0:
                cost = 200
                board = 500
            total_cost += cost
            total_board += board


        # add the average to a the data list
        data_cost.append(total_cost/amount_iterations)
        data_board.append(total_board/amount_iterations)

        # change the ratio of work:search ants for next iteration
        search += 1
        work -= 1

    return data_cost, data_board



'''
This function will scatter the points of the cost against the ratio
searcher:worker ants.
'''

def scatter(cost, term, titel):
    x = list(range(1,10))
    plt.scatter(x, cost)
    plt.yscale('linear')
    plt.plot(x,cost, color = 'orange')
    plt.title('Distribution of cost per worker:searcher ants')
    plt.xlabel('Amount of search ants (10 ants in total)')
    plt.legend(labels=['Population ' + term])
    plt.ylabel('Cost')
    plt.savefig(titel)
    plt.clf()

'''
This function returns the t-test value and the p-value between two populations.
'''
def ttest_pvalue(pop1, pop2):
    ttest = stats.ttest_ind(pop1, pop2)
    return ttest[0], ttest[1]


'''
The following code makes the simulation for the Antcolony. It is either possible
to only show the visual representation, or to simulate, collect data and show
the graphs.
'''

print("Welcome to the Antcolony sim.")
print("To run the simulation, collect data and show graphs, \
        enter: run simulation")
print("To see a visualisation of one run of the simulation, \
        enter: show visual ")

correct_input = False

while not correct_input:
    option = input()

    # for the visualisation of the grids over times
    if option == 'show visual':
        world = Grid([25, 0.1, 0.005, 1, 2])
        world.setNestLocation((14,3))
        world.setFoodSource((2,1), 6)
        world.visualSimulation()

        correct_input = True

    # for the collection of data and shows graphs
    elif option == 'run simulation':
        param = []
        food_sources = [(11,18)]
        food_sources2 = [(11, 18), (2, 1)]
        '''
        # this collects the baseline data and puts it in txt file
        cost, boards = make_data(25, 0.1, 0.005, food_sources)
        with open("cost.txt", "w") as output:
            output.write(str(cost))
            output.write(str(boards))

        # this collects the two food sources data and puts it in txt file
        cost1, boards1 = make_data(25, 0.1, 0.005, food_sources2)
        with open("cost1.txt", "w") as output:
            output.write(str(cost1))
            output.write(str(boards1))

        # this collects the higher pheromone strength data and puts in txt file
        cost2, boards2 = make_data(25, 0.2, 0.005, food_sources)
        with open("cost2.txt", "w") as output:
            output.write(str(cost2))
            output.write(str(boards2))
        '''
        # this collects the higher pheromone fade data and puts it in txt file
        #cost3, boards3 = make_data(25, 0.1, 0.01, food_sources)
        #with open("cost3.txt", "w") as output:
        #    output.write(str(cost3))
        #    output.write(str(boards3))


        correct_input = True
        cost = [38.80399999999983, 50.41299999999981, 70.52199999999976, 68.55299999999961, 70.16199999999985, 70.88499999999952, 66.54699999999983, 67.13799999999974, 63.55299999999992]
        cost1 = [19.110999999999777, 70.06799999999981, 102.92300000000002, 133.673, 139.988, 147.0680000000001, 138.53100000000012, 142.91300000000012, 124.54000000000022]
        cost2 = [65.8379999999995, 101.05799999999925, 102.88199999999928, 101.46599999999917, 115.79899999999866, 92.93499999999878, 86.93799999999878, 73.54599999999961, 70.94599999999967]
        cost3 = [81.62699999999944, 100.0979999999997, 121.99199999999965, 98.54699999999951, 99.0779999999993, 84.91599999999893, 65.85799999999955, 67.36899999999983, 64.46299999999987]
        
        scatter(cost, 'baseline fade', 'diff_pher_phade.png')
        scatter(cost1, 'two food sources fade', 'diff_food_sources.png')
        scatter(cost2, 'stronger pheromone', 'diff_pher_strength.png')
        scatter(cost3, 'faster pheromone fade', 'diff_pher_phade.png')

        # calculate the t-value and the p-value for every situation
        food_t, food_p = ttest_pvalue(cost, cost1)
        strength_t, strength_p = ttest_pvalue(cost, cost2)
        fade_t, fade_p = ttest_pvalue(cost, cost3)

        # print the t-value and p-value
        print("The t-value for food sources: " + str(food_t) + "\nThe p-value for food sources: " + str(food_p))
        print("The t-value for strength: " + str(strength_t) + "\nThe p-value for strength: " \
                                 + str(strength_p))
        print("The t-value for fade: " + str(fade_t) + "\nThe p-value for fade: " + str(fade_p))

    else:
        print("Incorrect input. Please enter either 'run simulation' or \
            'show visual'")
