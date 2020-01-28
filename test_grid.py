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


class Ant:

    def __init__(self, kind, create_location):
        self.location = create_location
        self.origin = create_location
        self.prev_steps = []

        # queue to keep track of the steps an ant needs to make
        self.nextSteps = queue.Queue()
        self.kind = kind

    # returns string of type of ant, string with w is worker, s is searcher
    def getKind(self):
        return self.kind

    # returns location of an ant
    def getLocation(self):
        return self.location

    # returns the previous step of an ant
    def getPrevLocations(self):
        return self.prev_steps

    # changes the origin of an ant
    def changeOrigin(self, coord):
        self.prev_steps = []
        self.origin = coord

    # returns the origin of an ant
    def getOrigin(self):
        return self.origin

    # adds a next step to the queue of next steps
    def addNextStep(self, nextCoord):
        self.nextSteps.put(nextCoord)

    # the location of the ant changes with a new step from the queue
    def setStep(self):
        step = self.nextSteps.get()
        self.prev_steps.append(self.location)

        if len(self.prev_steps) > 20:
            self.prev_steps.pop(0)

        self.location = step

    # returns an empty queue of steps of an ant
    def noNextSteps(self):
        return self.nextSteps.empty()

    '''
    This function finds the shortest path from a origin to the targetcoordinate.
    It is used when a searcher ant finds a food location (origin) and has to
    return to the nest (target location).
    There are three possibilities:
    1 ant is on the same x or y coordinate, in which case it can walk back in a
    straight line.
    2 the x-distance is the same as the y-distance, in which case the ant can
    walk back in diagonal line.
    3 the ant should walk back in a partly diagonal, partly straight line.
    '''
    def findShortestPath(self, targetCoord, origin):
        x = origin[0]
        y = origin[1]

        x_new = x
        y_new = y

        x_target = targetCoord[0]
        y_target = targetCoord[1]

        dx = x_target - x
        dy = y_target - y

        l = []

        while x != x_target or y != y_target:

            # posibility 1
            if x == x_target:
                y_new += dy / np.abs(dy)
                self.nextSteps.put((x_new, y_new))
                y = y_new
                continue

            if np.abs(dx) == np.abs(dy):
                x_new += dx / np.abs(dx)
                y_new += dx / np.abs(dy)
                self.nextSteps.put((x_new, y_new))
                x = x_new
                y = y_new
                continue

            # possibility 2
            if abs(dx) > abs(dy):
                while y != y_target:
                    x_new += dx / np.abs(dx)
                    y_new += dy / np.abs(dy)
                    self.nextSteps.put((x_new, y_new))
                    x = x_new
                    y = y_new
                x_new += dx / np.abs(dx)
                self.nextSteps.put((x_new, y_new))
                x = x_new

            # possibility 3
            else:
                while x != x_target:
                    x_new += dx / np.abs(dx)
                    y_new += dy / np.abs(dy)
                    self.nextSteps.put((x_new, y_new))
                    x = x_new
                    y = y_new

                y_new += dy / np.abs(dy)
                self.nextSteps.put((x_new, y_new))

                y = y_new



class Grid:

    # para = [Grid size, pheromone strength, pheromone fade, n_search, n_work]
    def __init__(self, para):
        # Create board of specifc board size
        self.grid = np.zeros((para[0], para[0]), dtype=float)
        self.grid_no_ants = self.grid
        self.grid_size = para[0]
        self.nest_value = 25
        self.food_source_value = 1000
        self.ant_value = 5
        self.pheromone_strength = para[1]
        self.pheromone_fade = para[2]
        self.n_search = para[3]
        self.n_work = para[4]
        self.ants = []

        self.nest_location = None
        self.food_location = {}
        self.total_cost = 0
        self.cost_per_step = 0.1
        self.return_1 = False
        self.total_found_food_value = 0
        self.found_food_sources = []
        self.return_reward = 1
        self.food_reward = 2*self.cost_per_step * (para[0]-1) +1
        self.return_reward = 0
        self.amt_steps = 0


    # Get neighbour xy coordinates as tuple
    def check_neighbours(self, coord):
        # Initalize variables
        neighbours = []

        # List of relative position for all possible neighbours of a coordinate
        adj = [(-1, 1),(0, 1),(1, 1),(-1, 0),(1, 0),(-1, -1),(0,-1),(1,-1)]

        # Create list of all possible nieghbours
        for n in adj:
            neighbours.append(tuple(np.subtract(coord, n)))

        # In case of corners and edge pieces: find all neighbours that exist but
        # are not valid for the specific board
        neigh = []
        for n_tup in neighbours:
            if n_tup[0] < 0 or n_tup[1] < 0:
                neigh.append(n_tup)
            elif n_tup[0] >= self.grid_size or n_tup[1] >= self.grid_size:
                neigh.append(n_tup)

        # Remove the found none valid neighbours
        for i in neigh:
            neighbours.remove(i)

        # Return list of neighbours as tuple
        return neighbours

    # change value at specific cell, returns new grid
    def change_cell_value(self, coord, new_value):
        self.grid[int(coord[1])][int(coord[0])] = new_value
        return self.grid

    # return all the ants on the board
    def get_ants(self):
        return self.ants

    # returns the value of a cell
    def get_kind(self, coord):
        x = int(coord[0])
        y = int(coord[1])
        return self.grid[y, x]

    # returns the pheromone value of a cell
    def get_pheromone(self, coord):
        x = int(coord[0])
        y = int(coord[1])
        return self.grid_no_ants[y, x]

    # returns the distance between a coordinate and the ants its origin
    def origin_distance(self, coord, origin):
        val = np.square(coord[0]-origin[0])+np.square(coord[1]-origin[1])
        return np.sqrt(val)

    # sets the new value of a cell
    def setKind(self, coord, value):
        x = int(coord[0])
        y = int(coord[1])
        ant = self.ant_value
        nest = self.nest_value
        food = self.food_source_value
        if value is not ant and value is not nest and value < food:
            return "Invalid value"
        self.grid[y, x] = value

    # set the coordinates and nest value as a dict
    def setNestLocation(self, coord):
        x = int(coord[0])
        y = int(coord[1])
        self.nest_location = (x,y)
        self.setKind(coord, self.nest_value)

    # set the coordinates and foodsource value as dict
    def setFoodSource(self, coord, amtFood):
        x = int(coord[0])
        y = int(coord[1])
        self.setKind(coord, self.food_source_value)
        self.food_location[coord] = amtFood

    # adds pheromone to a cell, when an ant leaves it
    def addPheromone(self, ant):
        coord = ant.getLocation()
        x = int(coord[0])
        y = int(coord[1])

        strength = self.pheromone_strength
        nest_loc = self.nest_location

        # if the ant is returning from food: add more pheromone than normal
        if ant.getOrigin() in self.food_location.keys():
                strength += 0.43

        # if the coordinate is not the food nor nest location add pheromone
        # with a maximum of 1.00
        if coord not in self.food_location.keys() and coord != nest_loc:
            self.grid[y, x] += strength
            self.grid_no_ants[y, x] += strength
            if self.grid_no_ants[y,x] >= 1:
                self.grid[y,x] = 1.00
                self.grid_no_ants[y, x] = 1.00

    '''
    This function calculates the new value of a cell where pheromone is located.
    It uses the reaction-diffusion-type simulation so the cells around a
    pheromone containing cell also have influence on how long it stays.
    '''
    def pheromoneFade(self, coord):
        # The cell must already have pheromones
        x = int(coord[0])
        y = int(coord[1])

        surroundPhero = 0

        # get the values of the neighbour cells, if they are only pheromones,
        # add them to surrounding pheromones
        for i in self.check_neighbours(coord):
            if 0 <= self.grid_no_ants[i[1], i[0]] <= 1:
                surroundPhero += self.grid_no_ants[i[1], i[0]]

        #reaction-diffusion-type simulation (Moore neighbours)
        fading = self.pheromone_fade
        phero_loc = self.grid_no_ants[coord[1], coord[0]]
        new_value = phero_loc * (1 - 8*fading) + fading*surroundPhero

        self.grid[y, x] = new_value
        self.grid_no_ants[y, x] = new_value

        if self.get_kind(coord) <= 0:
            self.grid[y,x] = 0
            self.grid_no_ants[y, x] = 0

    # if an ant retreives food, the total food value goes down by 1
    def retrieveFood(self, coord):
        # The cell must be of the kind food source
        x = int(coord[0])
        y = int(coord[1])
        self.grid[y, x] -= self.food_value;

    '''
    This function returns a list of the possible steps an ant can take
    '''
    def possible_steps_list(self, ant):
        coord = ant.getLocation()
        origin = ant.getOrigin()

        pn = self.check_neighbours(coord)
        dist = self.origin_distance(coord, origin)

        best_neigh = []
        for n in pn:
            distance = self.origin_distance(n, origin)
            if n not in ant.getPrevLocations():
                best_neigh.append(n)

        return best_neigh

    '''
    This function returns the decision of which step an ant takes. The input is
    the ant from which the location and origin can be retrieved. This function
    must be used only on a worker ant.
    '''
    def decide_step_worker(self, ant):
        coord = ant.getLocation()
        origin = ant.getOrigin()
        possible_steps = self.possible_steps_list(ant)
        pos_step = []
        pos_step_sum = 0

        # go over all the possible steps an ant can make
        for p in possible_steps:

            # add every possible step with pheromone to the possible step list
            if 0 < self.get_pheromone(p) <= 1:
                pos_step.append(p)
                pos_step_sum += self.get_pheromone(p)**2

            # food location is the only possible step, when next to is
            if p in self.food_location.keys():
                return p

            # when returning from food and ta possible step is the nest location
            # return the nest location as only possible step
            if p == self.nest_location and origin in self.food_location.keys():
                return p

        probability_distribution = []

        # ????
        for pos in pos_step:
            prob_dis = (self.get_pheromone(pos))**2/pos_step_sum
            probability_distribution.append(prob_dis)

        if possible_steps == []:
            return random.choice(self.check_neighbours(coord))

        if possible_steps == []:
            return random.choice(self.check_neighbours(coord))

        if pos_step == []:
            pos_step = possible_steps
            prob_value = 1/len(possible_steps)
            probability_distribution = [prob_value]*len(possible_steps)

        [index] = choice(range(len(pos_step)), 1, p=probability_distribution)

        return pos_step[index]


    '''
    This function returns the step a search ant will make. It
    '''
    def decide_step_search(self, ant):
        coord = ant.getLocation()
        origin = ant.getOrigin()
        possible_steps = self.possible_steps_list(ant)

        coord = ant.getLocation()
        origin = ant.getOrigin()

        pos_step = []
        p_food_n = []
        p_moore = []

        # go over all the possible steps
        for p in possible_steps:

            #retunr only the food location if it is next to it
            if p in self.food_location.keys():
                return p

            # adds the location next to food to the possible food steps ??
            if p in self.food_neighbours():
                p_food_n.append(p)

            # ???
            if p in self.food_moore_neigh():
                p_moore.append(p)

            # ?? kunnen we dit op een manier binnen de 80 krijgen??
            if self.get_kind(p) == 0 or self.ant_value < self.get_kind(p) < self.ant_value+self.pheromone_strength:
                pos_step.append(p)

        # if the possible steps to food is empty make a random choice
        if p_food_n != []:
            return random.choice(p_food_n)

        elif p_moore != []:
            return random.choice(p_moore)

        elif pos_step != []:
            return random.choice(pos_step)

        elif origin != self.nest_location and self.next_location in self.check_neighbours(coord):
            return self.next_location

        else:
            return random.choice(self.check_neighbours(coord))

    # returns list of all neighbours of a food location, all neighbours for
    # every food location are returned in one list
    def food_neighbours(self):
        food_neigh = []
        for food_loc in self.food_location.keys():
            food_neigh.append(self.check_neighbours(food_loc))
        return sum(food_neigh, [])

    # ???
    def food_moore_neigh(self):
        moore = []
        for n in self.food_neighbours():
            moore.append(self.check_neighbours(n))
        return sum(moore, [])

    # adds onse work ant
    def add_work_ant(self):
        nest = self.nest_location
        ant = Ant('w', nest)
        self.ants.append(ant)

    # adds one search ant
    def add_search_ant(self):
        nest = self.nest_location
        ant = Ant('s', nest)
        self.ants.append(ant)

    # adss food location as dictionary with coordinates and a food value
    def add_food_location(self, coord, amount):
        self.food_location[coord] = amount
        self.grid[int(coord[1]), int(coord[0])] += self.food_source_value

    # this function checks if an ant has returned to the nest and if so, the ant
    # gets removed
    def return_of_ant(self, ant):
        found_food_source = ant.getOrigin()
        if ant.getKind() == 's':

            # return of the first search ant, used to release worker ants
            if not self.return_1:
                self.return_1 = True

            if found_food_source not in self.found_food_sources:
                self.found_food_sources.append(found_food_source)

        # Mag dit weg???
        #if found_food_source not in self.found_food_sources:
        #    self.total_found_food_value += self.food_location[found_food_source]

        self.ants.remove(ant)


    # releases the ants to the grid from the nest location
    def release_ant(self, type_ant):
        if type_ant == 's':
            self.n_search -= 1
            self.add_search_ant()

        elif type_ant == 'w':
            self.n_work -= 1
            self.add_work_ant()

    '''
    update board
    First, we update the pheromone values when they fade each step. This is
    done for both the grid with and the grid without ants. Then, we remove the
    old ants by setting the grid equal to the grid_no_ants. Then we add the
    ants on their new locations, by first determining the location of an ant
    after it sets a step, and then setting this location of the grid to
    ant_value
    '''
    def renew_board(self):
        self.amt_steps += 1
        self.grid = copy.deepcopy(self.grid_no_ants)

        for i in range(len(self.grid_no_ants)):
            for j in range(len(self.grid_no_ants)):
                curr_cell = (i,j)
                cell_value = self.grid_no_ants[j,i]

                # If cell is empty
                if cell_value == 0:
                    continue

                # If cell is empty but contains a pheromone
                if 0 < cell_value <= 1:
                    self.pheromoneFade(curr_cell)

                if (i,j) in self.food_location and self.food_location[(i,j)] == 0:
                    self.grid_no_ants[j,i] = 0
                    self.grid[j,i] = 0
                    self.food_location.pop((i,j))

                    if (i,j) in self.found_food_sources:
                        self.found_food_sources.remove((i,j))


        # determine new location of each ant, set this location as ant_value
        ants_copy = self.ants.copy()
        for ant in ants_copy:

            # add pheromone on the current location before the ants sets a step
            ant_location = ant.getLocation()
            self.addPheromone(ant)

            # if the queue of ant is empty, decide a new step
            if ant.noNextSteps():

                # ant is on food source
                if ant_location in self.food_location.keys():
                    ant.changeOrigin(ant_location)
                    self.total_cost -= self.food_reward
                    self.food_location[ant_location] =  self.food_location[ant_location] - 1

                    # check if search ant
                    if ant.getKind() == 's':
                        ant.findShortestPath(self.nest_location, ant.getOrigin())

                    # check if worker ant
                    if ant.getKind() == 'w':
                        new_cell = self.decide_step_worker(ant)
                        ant.addNextStep(new_cell)

                # ant is on nest
                elif ant_location == self.nest_location:
                    if ant.getKind() == 's' and ant.getOrigin() == self.nest_location:
                        new_cell = self.decide_step_search(ant)
                        ant.addNextStep(new_cell)

                    if ant.getKind() == 'w' and ant.getOrigin() == self.nest_location:
                        new_cell = self.decide_step_worker(ant)
                        ant.addNextStep(new_cell)


                # ant is somewhere on the board that is not a nest or food source
                else:
                    # Check if search ant
                    if ant.getKind() == 's':
                        new_cell = self.decide_step_search(ant)
                        ant.addNextStep(new_cell)

                    # Check if worker ant
                    if ant.getKind() == 'w':
                        new_cell = self.decide_step_worker(ant)
                        ant.addNextStep(new_cell)

            ant.setStep()
            self.total_cost += self.cost_per_step
            new_location = ant.getLocation()

            # If after the step the ant is back at the nest location,
            # the ant has done its job and can be removed
            if new_location == self.nest_location and ant.getOrigin() != self.nest_location:
                #self.total_cost -= self.return_reward
                self.return_of_ant(ant)
                continue

            x_new_loc = int(new_location[0])
            y_new_loc = int(new_location[1])

            self.grid[y_new_loc, x_new_loc] += self.ant_value



    '''
    The following function finalizes the complete visualisation of the
    simulation. It renews the board for every time step and it then draws the
    new board. It also decides when search and work ants will be released on the
    board. It returns the total cost it took for the ants to empty the complete
    food source and it returns the amount of board renewals.
    '''
    def visualSimulation(self):
        cnt = 0
        iterations_counter = 0
        while True:
            # while not all the search ants have been released, release them
            while self.n_search != 0:
                self.release_ant('s')
                self.renew_board()
                drawnow(self.showGrid)
                plt.pause(0.0001)

            # if no search ant has returned, just keep updating the board
            while self.return_1 == False:
                self.renew_board()
                drawnow(self.showGrid)
                plt.pause(0.0001)

            # as long as the food location is not empty and ???
            while self.food_location != {}:
                # if not all the work ants have been released, release them
                while self.n_work != 0:
                    cnt += 1

                    if len(self.found_food_sources) == 0:
                        break

                    if cnt % 5 == 0:
                        self.release_ant('w')

                    self.renew_board()
                    drawnow(self.showGrid)
                    plt.pause(0.0001)

                #if len(self.ants) != 0:
                 #   break



                # kep on renewing the board
                if len(self.found_food_sources) == 0:
                    break

                iterations_counter += 1
                if iterations_counter == 400:
                    print("Too many iterations")
                    return 0, 0

                self.renew_board()
                drawnow(self.showGrid)
                plt.pause(0.0001)

            break
        print("Total cost: ", self.total_cost)
        print("Amount of board renewals: ", self.amt_steps)

        return self.total_cost, self.amt_steps

        self.showGrid()
        plt.show()

    '''
    The following function finalizes the complete simulation. It renews the
    board for every time step. It also decides when search and work ants will be
    released on the board. It returns the total cost it took for the ants to
    empty the complete food source and it retunrs the amount of board renewals.
    This function is used for the collection of data and is fast for testing.
    '''
    def simulation(self):
        print("Search ants: ", self.n_search)
        print("Work ants:" , self.n_work)
        cnt = 0
        iterations_counter = 0
        while True:
            # while not all the search ants have been released, release them
            while self.n_search != 0:
                self.release_ant('s')
                self.renew_board()

            # if no search ant has returned, just keep updating the board
            while self.return_1 == False:
                self.renew_board()

            # as long as the food location is not empty and ???
            while self.food_location != {}:

                # if not all the work ants have been released, release them
                while self.n_work != 0:
                    cnt += 1
                    if cnt % 5 == 0:
                        self.release_ant('w')

                    self.renew_board()

                #if len(self.ants) != 0:
                #    break
                
                # kep on renewing the board
                if len(self.found_food_sources) == 0:
                    break

                iterations_counter += 1
                if iterations_counter == 400:
                    print("Too many iterations")
                    return 0, 0

                self.renew_board()

            break
        print("Total cost: ", self.total_cost)
        print("Amount of board renewals: ", self.amt_steps)

        return self.total_cost, self.amt_steps

        self.showGrid()
        plt.show()

    '''
    This function gives the coloring of the grid. It also plots the complete
    grid with the corresponding colors for every unit of the simulation.
    '''
    def showGrid(self):
        # Amount of different colors for pheromones
        pheromone_amount = 80

        # Blue gradient for pheromones: the darker the blue, the more pheromones
        pher_colors = cm.Blues(np.linspace(0, 1.00001, num=pheromone_amount)).tolist()
        pher_bounds = np.linspace(0,1.1,num=pheromone_amount+1).tolist()

        # Other color: nothing, ant, nest and food source
        other_colors = ['white', 'black', 'white', 'peru', 'white', 'forestgreen']

        # Boundaries for the values of the other colors
        other_bounds = [1.00001, self.ant_value - 0.01, 11.00001, self.nest_value - 0.00001, self.nest_value + 0.0001, 99.99999, 100.0]
        total_colors = sum([pher_colors, other_colors], [])
        total_bounds = sum([pher_bounds, other_bounds], [])



        # If the value is 0, the cell is white
        total_colors[0] = 'white'

        # Create a cmap of all the colors
        cmap = mpl.colors.ListedColormap(total_colors)
        norm = mpl.colors.BoundaryNorm(total_bounds, cmap.N, clip=True)
        img = plt.imshow(self.grid,interpolation='nearest', cmap=cmap, norm=norm)

        ax = plt.gca()

        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True);
        ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True);

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)
        plt.xticks([], [])
        plt.yticks([], [])


'''
This function collects the data for the different ratios of ants. It runs the
simulation 100 times for every ratio from 10:0 to 0:10 workers:searchers. '''

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

<<<<<<< HEAD
#print(make_data(25, 0.1, 0.005))
=======
# print(make_data(25, 0.1, 0.005))
>>>>>>> 6ed2437be62976de10c67a204a75ff52b965a047


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


<<<<<<< HEAD
print("Welcome to the Antcolony sim.")
print("To run the simulation, collect data and show graphs, enter: run simulation")
print("To see a visualisation of one run of the simulation, enter: show visual ")

correct_input = False


while not correct_input:
    option = input()

    if option == 'show visual':
        world = Grid([25, 0.1, 0.005, 2, 8])
        world.setNestLocation((14,3))
        world.setFoodSource((2,1), 6)
        world.setFoodSource((11,18), 6)
        world.visualSimulation()

        correct_input = True

    elif option == 'run simulation':
        make_data(25, 0.1, 0.005)

        correct_input = True
        
    else:
        print("Incorrect input. Please enter either 'run simulation' or 'show visual'")



=======

a = [15.812000000000086, 24.11600000000023, 32.53600000000034, 41.981000000000385, 40.81100000000044, 40.42100000000045, 46.508000000000436, 49.846000000000394, 47.188000000000486, 50.36700000000042]

b =[10.54400000000005, 15.386000000000056, 15.502000000000088, 20.938000000000105, 22.395000000000167, 22.613000000000135, 23.635000000000126, 24.231000000000144, 24.222000000000165, 23.390000000000146]

plot_distributions(a, b, 'with 1 food source', 'with 2 food source')


print(ttest_pvalue(a,b))

#
# world = Grid([25, 0.1, 0.005, 8, 15])
# print("uuuuh")
# world.setNestLocation((14,3))
# world.setFoodSource((2,1), 6)
# world.setFoodSource((11,18), 6)
# world.setFoodSource((8,8), 6)
#
#
#
#
# world.showGrid()
# world.simulation()
>>>>>>> 6ed2437be62976de10c67a204a75ff52b965a047
