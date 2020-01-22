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
#from pylab import *
from drawnow import drawnow, figure

matplotlib.rc('animation', html='html5')

class Ant:
    
    def __init__(self, kind, create_location):
        self.location = create_location
        self.origin = create_location
        self.prev_steps = []
        # Queue to keep track of the steps an ant needs to make
        self.nextSteps = queue.Queue()
        self.kind = kind
    
    def getKind(self):
        return self.kind
        
    def changeLocation(self, coord):
        self.location = coord
        
    def getLocation(self):
        return self.location

    def getPrevLocations(self):
        return self.prev_steps
        
    def changeOrigin(self, coord):
        self.prev_steps = []
        self.origin = coord
        
    def getOrigin(self):
        return self.origin
        
    def addNextStep(self, nextCoord):
        self.nextSteps.put(nextCoord)
    
    # nextSteps.get removes the first element from the queue and returns it.
    # Therefore, when a step gets made it gets automatically removed from the queue
    def setStep(self):
        step = self.nextSteps.get()
        self.prev_steps.append(self.location)
        
        if len(self.prev_steps) > 20:
            self.prev_steps.pop(0)
        
        self.location = step
        
    def noNextSteps(self):
        return self.nextSteps.empty()

        
    def findShortestPath(self, targetCoord, origin):
        x = origin[0]
        y = origin[1]

        x_new = x
        y_new = y

        x_target = targetCoord[0]
        y_target = targetCoord[1]

        dx = x_target - x
        dy = y_target - y

        # Three possibilities: ant is on the same x or y coordinate, in which case it
        # can walk back in a straight line.
        # Or the x-distance is the same as the y-distance, in whcih case the ant can 
        # walk back in diagonal line.
        # Else, the ant should walk back in a partly diagonal, partly straight line.
        l = []

        while x != x_target or y != y_target:
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
    
    # para([Grid size, pheromone strength, pheromone fade, n_search, n_work])
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
        

    # Get neighbour xy coordinates as tuple  
    def check_neighbours(self, coord):
        # Initalize variables
        neighbours = []

        # List of relative position for all possible neighbours of a coordinate
        adj = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0,-1), (1,-1)]

        # Create list of all possible nieghbours
        for n in adj:
            neighbours.append(tuple(np.subtract(coord, n)))

        # In case of corners and edge pieces: find all neighbours that exist but are not valid for the specific 
        # board
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
    
    def get_ants(self):
        return self.ants
    
    # return which value a cell has
    def get_kind(self, coord):
        x = int(coord[0])
        y = int(coord[1])
        return self.grid[y, x]
    
    # returns the distance between a coordinate and the ants its origin
    def origin_distance(self, coord, origin):
        val = np.square(coord[0]-origin[0])+np.square(coord[1]-origin[1])
        return np.sqrt(val)
    
    # sets the new value of a cell
    def setKind(self, coord, value):
        x = int(coord[0])
        y = int(coord[1])
        if value is not 5 and value is not 25 and value < 100:
            return "Invalid value"
        self.grid[y, x] = value
        
    def setNestLocation(self, coord):
        x = int(coord[0])
        y = int(coord[1])
        self.nest_location = (x,y)
        self.setKind(coord, self.nest_value)

    def setFoodSource(self, coord, amtFood):
        x = int(coord[0])
        y = int(coord[1])
        self.setKind(coord, self.food_source_value)
        self.food_location[coord] = amtFood
        print(self.food_location)

    def addPheromone(self, ant):
        coord = ant.getLocation()
        x = int(coord[0])
        y = int(coord[1])

        strength = self.pheromone_strength

        if ant.getOrigin() in self.food_location.keys():
            strength += 0.2

        if coord not in self.food_location.keys() and coord != self.nest_location:
            self.grid[y, x] += strength
            self.grid_no_ants[y, x] += strength
            if self.grid_no_ants[y,x] >= 1:
                self.grid[y,x] = 1.00
                self.grid_no_ants[y, x] = 1.00    

    def pheromoneFade(self, coord):
        # The cell must already have pheromones
        x = int(coord[0])
        y = int(coord[1])
        
        surroundPhero = 0
        
        # get the values of the neighbour cells, if they are only pheromones, add them to surrounding pheromones
        for i in self.check_neighbours(coord):
            if 0 <= self.get_kind(i) <= 1:
                surroundPhero += self.get_kind(i) 
    
        #reaction-diffusion-type simulation (Moore neighbours)
        new_value = self.get_kind(coord) * (1 - 8*self.pheromone_fade) + self.pheromone_fade*surroundPhero

        self.grid[y, x] = new_value
        self.grid_no_ants[y, x] = new_value    
        
        if self.get_kind(coord) <= 0:
            self.grid[y,x] = 0
            self.grid[y,x] = 0

    def retrieveFood(self, coord):
        # The cell must be of the kind food source
        x = int(coord[0])
        y = int(coord[1])
        self.grid[y, x] -= self.food_value;
    
        
    def possible_steps_list(self, ant):
        coord = ant.getLocation()
        origin = ant.getOrigin()

        pn = self.check_neighbours(coord)
        dist = self.origin_distance(coord, origin)
        
        best_neigh = []
        for n in pn:
            distance = self.origin_distance(n, origin)
            #if distance > dist:
            if n not in ant.getPrevLocations():
                best_neigh.append(n)

        return best_neigh

    
    def decide_step_worker(self, ant):
        coord = ant.getLocation()
        origin = ant.getOrigin()
        possible_steps = self.possible_steps_list(ant)
        pos_step = []
        pos_step_sum = 0
        
        
        for p in possible_steps:
            # if a possible step contains pheromone, add it to the possible steps list
            if 0 < self.get_kind(p) <= 1:
                pos_step.append(p)
                pos_step_sum += self.get_kind(p)**2
                
            # if a food location is a possible step: return this as only possible step
            if p in self.food_location.keys():
                return p
            
            # if a food location is a possible step: return this as only possible step
            if p == self.nest_location and origin in self.food_location.keys():
                return p
            
                
        probability_distribution = []
        
        for pos in pos_step:
            self.get_kind(pos)
            probability_distribution.append(((self.get_kind(pos))**2/pos_step_sum))
            
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
    def decide_step_worker(self, ant):
        coord = ant.getLocation()
        origin = ant.getOrigin()
        largest_value = 0
        step_to_take = 0
        possible_steps = self.possible_steps_list(ant)

        for step in possible_steps:
            x = int(step[0])
            y = int(step[1])
            value = self.grid_no_ants[y, x]
            if value > largest_value:
                largest_value = value
                step_to_take = step

        if step_to_take == 0:
            step_to_take = random.choice(possible_steps)

        return step_to_take
    '''
    
    def decide_step_search(self, ant):
        possible_steps = self.possible_steps_list(coord, origin)
        
        pos_step = []
        p_food_n = []
        for p in possible_steps:
            if p in self.food_location.keys():
                return p
            if p in self.food_neighbours():
                p_food_n.append(p)
            if self.get_kind(p) == 0 or self.ant_value < self.get_kind(p) < self.ant_value+self.pheromone_strength:
                pos_step.append(p)
        
        if p_food_n != []:
            return random.choice(p_food_n)
        
        elif pos_step != []:
            return random.choice(pos_step)

        elif origin != self.nest_location and self.next_location in self.check_neighbours(coord):
            return self.next_location
            
        else:
            return random.choice(self.check_neighbours(coord))

    # Returns list of all neighbours of a food location, all neigh for every food loc are returned in one list
    def food_neighbours(self):
        food_neigh = []
        for food_loc in self.food_location.keys():
            food_neigh.append(self.check_neighbours(food_loc))
        return sum(food_neigh, [])
        
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
        
    def add_food_location(self, coord, amount):
        self.food_location[coord] = amount
        self.grid[int(coord[1]), int(coord[0])] += self.food_source_value

        
    def return_of_ant(self, ant):
        if ant.getKind() == 's':

            if not self.return_1:
                self.return_1 = True

        found_food_source = ant.getOrigin()
            
        if found_food_source not in self.found_food_sources:
            self.found_food_sources.append(found_food_source)
            self.total_found_food_value += self.food_location[found_food_source]
            
        self.ants.remove(ant)

                    
    # release of the ants
    def release_ant(self, type_ant):
        if type_ant == 's':
            self.n_search -= 1
            self.add_search_ant()
            
        elif type_ant == 'w':
            self.n_work -= 1
            self.add_work_ant()
        

                        

    # update board
    # First, we update the pheromone values when they fade each step. This is done for both the grid
    # with and the grid without ants.
    # Then, we remove the old ants by setting the grid equal to the grid_no_ants
    # Then we add the ants on their new locations, by first determining the location of an ant after it sets
    # a step, and then setting this location of the grid to ant_value
    def renew_board(self):
                
        self.grid = copy.deepcopy(self.grid_no_ants)
        
        for i in range(len(self.grid_no_ants)):
            for j in range(len(self.grid_no_ants)):
                curr_cell = (i,j)
                cell_value = self.grid_no_ants[j,i]
                
                # If cell is empty
                if cell_value == 0:
                    continue
      
                # If cell is empty but contains a pheromone (hier gaat het goed met nest_value)
                if 0 < cell_value <= 1:
                    self.pheromoneFade(curr_cell)

        
        # Determine the new location of each ant and set this new location on the board to ant_value
        ants_copy = self.ants.copy()
        for ant in ants_copy:
            
            # Add pheromone on the current location before the ants sets a step
            ant_location = ant.getLocation()
            self.addPheromone(ant)
            
            # If the ant has no next steps in its queue, decide which next step it should take
            if ant.noNextSteps():
                
                # Ant is on food source
                if ant_location in self.food_location.keys():
                    ant.changeOrigin(ant_location)
                    self.total_cost -= self.food_reward
                    self.food_location[ant_location] =  self.food_location[ant_location] - 1
                    
                    # Check if search ant
                    if ant.getKind() == 's':
                        ant.findShortestPath(self.nest_location, ant.getOrigin())
                    
                    # Check if worker ant
                    if ant.getKind() == 'w':
                        new_cell = self.decide_step_worker(ant)
                        ant.addNextStep(new_cell)
                
                # Ant is on nest
                elif ant_location == self.nest_location:
                    if ant.getKind() == 's' and ant.getOrigin() == self.nest_location:
                        new_cell = self.decide_step_search(ant)
                        ant.addNextStep(new_cell)

                    if ant.getKind() == 'w' and ant.getOrigin() == self.nest_location:
                        new_cell = self.decide_step_worker(ant)
                        ant.addNextStep(new_cell)
                        
                  
                # Ant is somewhere on the board that is not a nest or food source
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
                self.total_cost -= self.return_reward
                self.return_of_ant(ant)
                continue
                
            x_new_loc = int(new_location[0])
            y_new_loc = int(new_location[1])
            
            self.grid[y_new_loc, x_new_loc] += self.ant_value
            
    # this is the simulation function where the complete sotry of the
    def simulation(self):
        cnt = 0
        while True:
            cnt += 1
            if cnt > 1500:
                break

            while self.n_search != 0:
                self.release_ant('s')
                self.renew_board()
                drawnow(self.showGrid)  # Call drawnow to update our live graph
                plt.pause(0.0001)
        
            # if no search ant has returned, just keep updating the board
            while self.return_1 == False: 
                self.renew_board()
                drawnow(self.showGrid)  # Call drawnow to update our live graph
                plt.pause(0.0001)
                

            while self.total_found_food_value != 0 and len(self.ants) != 0:
                while self.n_work != 0:
                    self.release_ant('w')
                    self.renew_board()
                    drawnow(self.showGrid)  # Call drawnow to update our live graph
                    plt.pause(0.0001)
    #                 print(self.grid)

                # kep on renewing the board 
                self.renew_board()
                drawnow(self.showGrid)  # Call drawnow to update our live graph
                plt.pause(0.0001)

       	    break
            
    def showGrid(self):
        # Amount of different colors for pheromones
        pheromone_amount = 80
        
        # Blue gradient for pheromones: the darker the blue, the more pheromones
        pher_colors = cm.Blues(np.linspace(0, 1.00001, num=pheromone_amount)).tolist()
        pher_bounds = np.linspace(0,1.1,num=pheromone_amount+1).tolist()
        
        # Other color: nothing, ant, nest and food source
        other_colors = ['white', 'black', 'white', 'peru', 'white', 'forestgreen']
        
        # Boundaries for the values of the other colors
        other_bounds = [1.00001, self.ant_value - 0.001, 11.00001, self.nest_value - 0.001, self.nest_value + 0.01, 99.99999, 100.0]
        total_colors = sum([pher_colors, other_colors], [])
        total_bounds = sum([pher_bounds, other_bounds], [])
        
        # If the value is 0, the cell is white
        total_colors[0] = 'white'
        
        # Create a cmap of all the colors
        cmap = mpl.colors.ListedColormap(total_colors)
        norm = mpl.colors.BoundaryNorm(total_bounds, cmap.N)
        img = plt.imshow(self.grid,interpolation='nearest', cmap=cmap, norm=norm)
        
        ax = plt.gca()
        
        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True);
        ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True);

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)
        
        plt.xticks([], [])
        plt.yticks([], [])
        #plt.show()
        
#     def visualize_grid_plot(self, i):
#         self.showGrid()
#         imshow()
            
        

        

world = Grid([20, 0.7, 0.01, 2, 3])
world.setNestLocation((14,3))
world.setFoodSource((2,1), 20)
world.setFoodSource((11,18), 20)

# world.simulation()

world.showGrid()
world.simulation()

#for i in range(50):
#     world.renew_board()
#     #print(world.grid)
#     world.showGrid()

#for k in range(50):
#    drawnow(world)