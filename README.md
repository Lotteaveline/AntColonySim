## NAME
### Ant Colony Simulation

## DESCRIPTION
This project simulates the way searcher ants search food and worker ants collect
food for an ant colony. It is an agent-based model where the agents are the ants.
There are different variables that can be changed to test the way this simulation
works. The grid size, pheromone strength, pheromone fade, amount of worker and
searcher ants can all be altered.

## VISUALS
[Baseline plot](https://i.imgur.com/uQWCm3s.png)
[Two food sources plot](https://i.imgur.com/NQHeDR3.png)
[Pheromone Fade plot](https://i.imgur.com/BvvDoiM.png)
[Pheromone Strength plot](https://i.imgur.com/9AOTok7.png)

## INSTALLATION
This project uses the standard scientific python packages. It also uses the
following packages. To install these, enter 'pip3 install <packagename>' in the 
terminale.

Packages:
queue
drawnow
copy
seaborn

## USAGE
To run our code run "antcolony.py" as python3 script and choose one of the options:
* run simulation - run the simulation, collect the data and plot the graphs
* show visual - visualizes the grids of the way the ants walk
(WARNING: run simulation will take at least 3 hours to fully run)


## Files
There are three necessary code files:

ants.py
This contains the class of an ant.

grid.py
This contains the class of the grid set-up and rules it should follow every timestap.

antcolony.py
Runs the program. Creates a visualisation or collects data and plots it.

Other files are:
costs.txt, cost1.txt, cost2.txt, cost3.txt.
These contain the calculated results. cost.txt is for the baseline and the others are
for the changed amount of food sources, the changed pheromone strength and the changed 
rate at which the pheromones fade.

The grid_pheromone.png file contains a screenshot of the visualized ant simulation.
The other .png files contain the plots of each cost.


## CONTRIBUTION
We are open to contributions and if you want to do so, you can email us on the
following mail address: antcolonysim@gmail.com

## AUTHORS AND ACKNOWLEDGEMENT
Nienke Duetz, Ghislaine van den Boogerd and Lotte Bottema have made this project
possible.
