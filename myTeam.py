# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from util import *
from game import Directions
from game import Actions
import game
from copy import deepcopy
import numpy as np


class MCTS:
    def __init__(self, isRed):
        self.states_played = {};
        self.states_won = {};
        self.red = isRed;
        self.exploration_c = 2.0;
        self.time_limit = 0.9;

        
    def next_move(self, gameState):
        # find whos turn it is
        turn = (gameState.data._agentMoved + 1) % gameState.getNumAgents();       

        # get legal moves  
        moves = gameState.getLegalActions(turn);

        # if trivial (this should never happen)
        if(len(moves) == 0):
            return None;
        elif(len(moves) == 1):
            return moves[0];

        t0 = time.time();
        while (time.time() - t0 < self.time_limit):
            simulate();

        

    def simulate(self):
        return None;

    
class EnemyTracker: ######### TODO: Keep track of eaten food, and assign it to most likely agent
    # counter object for keeping track of enemy positions
    # self.tracker : dict of counters
    # self.enemy_idxs = [];

    def init(self, gameState, isRed):    
        self.red = isRed;
        self.first_update = True;
        self.enemy_idxs = [];
        if(isRed): # enemy blue
            self.enemy_idxs = gameState.getBlueTeamIndices();
            self.old_food_state = gameState.getRedFood();
        else:
            self.enemy_idxs = gameState.getRedTeamIndices();
            self.old_food_state = gameState.getBlueFood();

        
        
        all_idxs = gameState.getRedTeamIndices();
        all_idxs.extend(gameState.getBlueTeamIndices());
        all_idxs.sort();
        self.tracker = [None]*len(all_idxs);
        

        for i in all_idxs:
            if(i in self.enemy_idxs):                   
                self.tracker[i] = Counter();
                self.tracker[i][gameState.getInitialAgentPosition(i)] = 1.0;  


    def update(self,gameState,my_index): #{
        #check if food got eaten
        if(self.red): # enemy blue
            new_food_state = gameState.getRedFood();
        else:
            new_food_state = gameState.getBlueFood();
        
        eaten_food = [];

        for i in range(self.old_food_state.width):
            for j in range(self.old_food_state.height):
                if self.old_food_state[i][j] and not new_food_state[i][j]:
                    eaten_food.append((i,j)); 

    

        self.old_food_state = new_food_state;

        measured_dists = gameState.getAgentDistances();
        for i in self.enemy_idxs: #{
            exact_pos = gameState.getAgentPosition(i);
            if(exact_pos != None): #{
                self.tracker[i] = Counter();
                self.tracker[i][exact_pos] = 1.0;
            else:
                temp_tracker = Counter();
                for key,value in self.tracker[i].iteritems(): #{
                    if(value == 0.0):
                        continue;
                    
                    if(my_index == 0 and self.first_update):
                        self.first_update = False;    
                        temp_tracker[key] = value;
                        continue;

                    if((my_index - 1) % gameState.getNumAgents() == i):   #if this agent moved last turn, update his pos             
                        p_move = 0;
                        for direction,_ in Actions._directions.iteritems(): #{
                            pos = Actions.getSuccessor(key,direction);
                            if(not gameState.hasWall(int(pos[0]),int(pos[1]))):
                                p_move += 1;
                        #}

                        p_move = 1.0 / p_move;

                        for direction,_ in Actions._directions.iteritems(): #{
                            pos = Actions.getSuccessor(key,direction);
                            if(not gameState.hasWall(int(pos[0]),int(pos[1]))):
                                temp_tracker[pos] += p_move * value;
                        #}
                    else: #if this agent did not move last turn, pretend he moved using action STOP   
                        temp_tracker[key] = value;
                    #}
                #}
                
                for key,value in temp_tracker.iteritems(): #{
                    true_dist = manhattanDistance(key,gameState.getAgentPosition(my_index));
                    temp_tracker[key] = value * gameState.getDistanceProb(true_dist, measured_dists[i]);
                    if(key in eaten_food):
                        temp_tracker[key] = 1;
                #}
                self.tracker[i] = deepcopy(temp_tracker);
                self.tracker[i].normalize();
            #}
        #}
    #}


    def update_eaten_agent(self,gameState,index_eaten):
        self.tracker[index_eaten] = Counter();
        self.tracker[index_eaten][gameState.getInitialAgentPosition(index_eaten)] = 1.0;



tracker = EnemyTracker();


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    tracker.init(gameState,self.red);
	


    #state = gameState;
    #turn = 0;
    #t0 = time.time()
    #while True:
    #    actions = state.getLegalActions(turn);
    #    state = state.generateSuccessor(turn,np.random.choice(actions));
    #    turn = (turn + 1) % 4;

    #    if state.isOver() or state.data.timeleft == 0:
    #        break;

    #t1 = time.time()        	
    #print("time taken: ", t1-t0);


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    #n_agents = gameState.getNumAgents();

    #print (gameState)
    #for i in range(n_agents):
    #    s = "{} [{}, {}]"
    #    if gameState.getAgentPosition(i) != None:           
    #        print(s.format(i,gameState.getAgentPosition(i)[0],gameState.getAgentPosition(i)[1]))

    #s = "{} : [{}]"


    tracker.update(gameState,self.index);    
    self.displayDistributionsOverPositions(tracker.tracker);


    #print (gameState)

    ##if gameState.getAgentDistances() != None:    
    #n = len(gameState.getAgentDistances());
    #for i in range(n):
    #    #if gameState.getAgentDistances()[i] != None:
    #    print(s.format(i,gameState.getAgentDistances()[i]))

    
    #print("me:", self.index)
    
    #if self.index % 2 == 0:
    #    print("x = ",myclass.x);
    #    myclass.x += 1;

    #print(gameState)
    time.sleep(0.01)
    
    return random.choice(actions)




