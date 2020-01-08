import numpy as np
import matplotlib.pyplot as plt

class gridCell:

	def __init__(self, parameters):
		# Parameters is a list of the following:
		# pheromone strength, pheromone fade, food value

		self.value = 0
		self.neighbours = []
		self.pheromone_strength = parameters[0]
		self.phereomone_fade = parameters[1]
		self.food_value = parameters[2]

		self.ant_value = 5
		self.food_source_value = 100
		self.nest_value = 10


	def getKind(self, coord):
		x = coord[0]
		y = coord[1]
		return self.grid[y, x]

	def setKind(self, coord, value);
		x = coord[0]
		y = coord[1]
		if value is not 5 and value is not 10 and value < 100:
			return "Invalid value"
		self.grid[y, x] = value


	def antArrives(self, coord):
		x = coord[0]
		y = coord[1]
		self.grid[y, x] += self.ant_value

	def antLeaves(self, coord):
		x = coord[0]
		y = coord[1]
		self.grid[y, x] -= self.ant_value
		self.addPheromone(coord)

	def addPheromone(self, coord):
		x = coord[0]
		y = coord[1]
		self.grid[y, x] += pheromone_strength

	def pheromoneFade(self, coord):
		# The cell must already have pheromones
		x = coord[0]
		y = coord[1]
		self.grid[y, x] -= phereomone_fade;

	def retrieveFood(self, coord):
		# The cell must be of the kind food source
		x = coord[0]
		y = coord[1]
		self.grid[y, x] -= self.food_value;



def antSetStep(self, coord, nextCoord):
	x = coord[0]
	y = coord[1]
	x_next = nextCoord[0]
	y_next = nextCoord[1]
	self.grid[y_next, x_next] += self.ant_value
	self.grid[y, x] -= self.ant_value
	self.grid[y, x] += self.pheromone_strength



def routeBack(self, coord, origin):
	x = coord[0]
	y = coord[1]

	x_new = x
	y_new = y

	x_orig = origin[0]
	y_orig = origin[1]

	dx = x - x_orig
	dy = y - y_orig

	# Three possibilities: ant is on the same x or y coordinate, in which case it
	# can walk back in a straight line.
	# Or the x-distance is the same as the y-distance, in whcih case the ant can 
	# walk back in diagonal line.
	# Else, the ant should walk back in a partly diagonal, partly straight line.

	while x != x_orig and y != y_orig:
		if x == x_orig:
			x_new += dx / np.abs(dx)
			self.antSetStep((x, y), (x_new, y_new))
			x = x_new
			continue

		if np.abs(dx) == np.abs(dy):
			x_new += dx / np.abs(dx)
			y_new += dx / np.abs(dy)
			self.antSetStep((x, y), (x_new, y_new))
			x = x_new
			y = y_new
			continue

		while y != y_orig:
			x_new += dx / np.abs(dx)
			y_new += dx / np.abs(dy)
			self.antSetStep((x, y), (x_new, y_new))
			x = x_new
			y = y_new

		y_new += dy / np.abs(dy)
		self.antSetStep((x, y), (x_new, y_new))
		y = y_new



# Pheromones faden moet veeeel langzamer


def showGrid(self):
	cmap = mpl.colors.ListedColormap(['white', 'deepskyblue', 'white',
		'black', 'white', 'peru', 'white', 'forestgreen'])
	bounds = [0, 0.000001, 1, 4.999, 5, 6, 9.99999, 10, 10.001, 100]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	img = pyplot.imshow(self.grid,interpolation='nearest',
                    cmap=cmap, norm=norm)
	plt.show()


world  = Grid(10)

print(world.grid)

