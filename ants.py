import numpy as np
import random
from numpy.random import choice
import copy
import queue



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
