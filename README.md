## NAME
### Ant Colony Simulation

## DESCRIPTION
This project simulates the way searcher ants search food and worker ants collect
food for an ant colony. It is an agent-based model where the agents are the ants.
There are different variables that can be changed to test the way this simulation
works. The grid size, pheromone strength, pheromone fade, amount of worker and
searcher ants can all be altered.

## VISUALS
[Grid visualisation](/grid_pheromone.png)


# INSTALLATION
This project uses matplotlib, numpy, etc. etc. #standard packages
But it also needs a package which is less commonly used namely DrawNow. This
package draws the grid on top of the others, so it will look like a simulation.

# USAGE
There are three files.

ants.py
This contains the class of an ant.

grid.py
This contains the class of the grid set-up and rules it should follow every timestap.

antcolony.py
To run our code run "antcolony.py" as python3 script and choose one of the options:
$ python3 antcolony.py    
* run simulation - run the simulation, collect the data and plot the graphs
* show visual - visualizes the grids of the way the ants walk

# CONTRIBUTION
We are open to contributions and if you want to do so, you can email us on the
following mail adres: lotteavelien@live.nl

# AUTHORS AND ACKNOWLEDGEMENT
Nienke Duetz, Ghislaine van den Boogerd and Lotte Bottema have made this project
possible.
