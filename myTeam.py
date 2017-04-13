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
    # self.tracker : list of counters
    # self.enemy_idxs = [];

    def init(self, gameState, isRed):    
        self.red = isRed;
        self.first_update = True;
        self.enemy_idxs = [];
        self.enemy_edge = [];
        if(isRed): # enemy blue
            self.enemy_idxs = gameState.getBlueTeamIndices();
            self.old_food_state = gameState.getRedFood();
            x_edge = gameState.data.layout.width / 2;
        else:
            self.enemy_idxs = gameState.getRedTeamIndices();
            self.old_food_state = gameState.getBlueFood();
            x_edge = gameState.data.layout.width / 2 - 1;

        
        
        all_idxs = gameState.getRedTeamIndices();
        all_idxs.extend(gameState.getBlueTeamIndices());
        all_idxs.sort();
        self.tracker = [None]*len(all_idxs);
        self.carrying_food = [0]*len(self.enemy_idxs);

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

        temp_trackers = {};
        measured_dists = gameState.getAgentDistances();
        for i in self.enemy_idxs: #{
            exact_pos = gameState.getAgentPosition(i);
            if(exact_pos != None): #{
                temp_trackers[i] = Counter();
                temp_trackers[i][exact_pos] = 1.0;
            else:
                temp_trackers[i] = Counter();
                for key,value in self.tracker[i].iteritems(): #{
                    if(value == 0.0):
                        continue;
                    
                    if(my_index == 0 and self.first_update):
                        self.first_update = False;    
                        temp_trackers[i][key] = value;
                        continue;

                    if((my_index - 1) % gameState.getNumAgents() == i):   #if this agent moved last turn, update his pos             
                        p_move = np.zeros(5);
                        k = 0;
                        for direction,_ in Actions._directions.iteritems(): #{
                            pos = Actions.getSuccessor(key,direction);
                            if(not gameState.hasWall(int(pos[0]),int(pos[1]))):
                                p_move[k] += 1;
                            if(pos not in self.tracker[i]):                        
                                p_move[k] *= 2;
                            if(direction == Directions.STOP):
                                p_move[k] /= 2;
                                
                            k += 1;
                        #}

                        p_move = p_move / np.sum(p_move);

                        k = 0;
                        for direction,_ in Actions._directions.iteritems(): #{
                            pos = Actions.getSuccessor(key,direction);
                            if(not gameState.hasWall(int(pos[0]),int(pos[1]))):
                                temp_trackers[i][pos] += p_move[k] * value;
                            k += 1;
                        #}
                    else: #if this agent did not move last turn, pretend he moved using action STOP   
                        temp_trackers[i][key] = value;
                    #}
                #}
                
                for key,value in temp_trackers[i].iteritems(): #{
                    true_dist = manhattanDistance(key,gameState.getAgentPosition(my_index));
                    if(true_dist <= 5):
                        temp_trackers[i][key] = 0.0;
                    else:
                        temp_trackers[i][key] = value * gameState.getDistanceProb(true_dist, measured_dists[i]);
                    #if(key in eaten_food):
                        #temp_trackers[i][key] = 1;
                #}
                
            #}
        #}
                
        print(self.carrying_food );
        for key in eaten_food:
            if(temp_trackers[self.enemy_idxs[0]][key] > temp_trackers[self.enemy_idxs[1]][key]):
                temp_trackers[self.enemy_idxs[0]][key] = 2;
                self.carrying_food[self.enemy_idxs[0]] += 1;
            else:
                temp_trackers[self.enemy_idxs[1]][key] = 2;
                self.carrying_food[self.enemy_idxs[1]] += 1;

        for key, _ in temp_trackers.iteritems():
            self.tracker[key] = deepcopy(temp_trackers[key]);
            self.tracker[key].normalize();

        scorediff = abs(gameState.data.scoreChange);
        if(scorediff > 0):
            if(self.carrying_food[self.enemy_idxs[0]] != 0 and self.carrying_food[self.enemy_idxs[1]] == 0):
                for key, _ in self.tracker[self.enemy_idxs[0]].iteritems():
                    if(key[0] not in self.enemy_edge):
                        self.tracker[self.enemy_idxs[0]][key] = 0.0;
                self.carrying_food[self.enemy_idxs[0]] = 0;
            elif(self.carrying_food[self.enemy_idxs[0]] == 0 and self.carrying_food[self.enemy_idxs[1]] != 0):
                 for key, _ in self.tracker[self.enemy_idxs[1]].iteritems():
                    if(key[0] not in self.enemy_edge):
                        self.tracker[self.enemy_idxs[1]][key] = 0.0;
                 self.carrying_food[self.enemy_idxs[1]] = 0;
            elif(self.carrying_food[self.enemy_idxs[0]] != 0 and self.carrying_food[self.enemy_idxs[1]] != 0):
                most_likely_enemy = self.enemy_idxs[0];
                less_likely_enemy = self.enemy_idxs[1];
                max_prob = -1.0;
                for key, _ in self.tracker[self.enemy_idxs[1]].iteritems():
                    if(key[0] in self.enemy_edge):
                        if(max_prob < self.tracker[self.enemy_idxs[1]][key]):
                            max_prob = self.tracker[self.enemy_idxs[1]][key];
                            most_likely_enemy = self.enemy_idxs[1];
                            less_likely_enemy = self.enemy_idxs[0];
                for key, _ in self.tracker[self.enemy_idxs[0]].iteritems():
                    if(key[0] in self.enemy_edge):
                        if(max_prob < self.tracker[self.enemy_idxs[0]][key]):
                            max_prob = self.tracker[self.enemy_idxs[0]][key];
                            most_likely_enemy = self.enemy_idxs[0];
                            less_likely_enemy = self.enemy_idxs[1];
                
                diff = self.carrying_food[most_likely_enemy] - scorediff;       
                if(diff < 0):
                    self.carrying_food[most_likely_enemy] = 0;
                    self.carrying_food[less_likely_enemy] = max(0,self.carrying_food[less_likely_enemy] - diff);
                elif(diff > 0):
                    self.carrying_food[most_likely_enemy] = 0;
                    self.carrying_food[less_likely_enemy] += diff;
                else:
                    self.carrying_food[most_likely_enemy] = 0;

                for key, _ in self.tracker[most_likely_enemy].iteritems():
                    if(key[0] not in self.enemy_edge):
                        self.tracker[self.enemy_idxs[1]][key] = 0.0;
                   
                for key, _ in trackers.iteritems():
                    self.tracker[key].normalize();
    #}

    def estimate_enemy_pos(self, enemyIndex):
        max_prob = -1;
        pos = 0;
        for key, value in self.tracker[enemyIndex].iteritems():
            if(value > max_prob):
                max_prob = deepcopy(value);
                pos = deepcopy(key);
        return pos, max_prob;

    def update_eaten_agent(self,gameState,index_eaten):
        self.tracker[index_eaten] = Counter();
        self.tracker[index_eaten][gameState.getInitialAgentPosition(index_eaten)] = 1.0;
        self.carrying_food[index_eaten] = 0;



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
	

    self.first = False;    

    if(self.red): # enemy blue
        self.enemy_idxs = gameState.getBlueTeamIndices();
        self.enemy_edge = gameState.data.layout.width / 2;
    else:
        self.enemy_idxs = gameState.getRedTeamIndices();
        self.enemy_edge = gameState.data.layout.width / 2 - 1;


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

    my_index = self.index;
    my_pos = gameState.getAgentPosition(self.index);

    #n_agents = gameState.getNumAgents();

    #print (gameState)
    #for i in range(n_agents):
    #    s = "{} [{}, {}]"
    #    if gameState.getAgentPosition(i) != None:           
    #        print(s.format(i,gameState.getAgentPosition(i)[0],gameState.getAgentPosition(i)[1]))

    #s = "{} : [{}]"


    tracker.update(gameState,self.index);    
    self.displayDistributionsOverPositions(tracker.tracker);

    enemy0_pos, enemy0_prob = tracker.estimate_enemy_pos(self.enemy_idxs[0]);
    enemy1_pos, enemy1_prob = tracker.estimate_enemy_pos(self.enemy_idxs[1]);
    
    enemy_positions = [enemy0_pos, enemy1_pos];

    #print (gameState.getScore())

    #print(self.index)
    #if(self.index == 2 and not self.first):
    #    self.first = True;
    #    state = gameState;
    #    turn = 2;

    #    act = state.getLegalActions(turn);
    #    state = state.generateSuccessor(turn,np.random.choice(act));
    #    turn = (turn + 1) % 4;
        
    #    print(state.data.agentStates[3])

    # need to generate a AgentState
    # need to generate a Configuration
    # for each enemy agent, with the current belief


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
    #time.sleep(0.01)
    my_action = random.choice(actions);
    if(Actions.getSuccessor(my_pos,my_action) == enemy0_pos):
        tracker.update_eaten_agent(gameState,self.enemy_idxs[0]);
    if(Actions.getSuccessor(my_pos,my_action) == enemy1_pos):
        tracker.update_eaten_agent(gameState,self.enemy_idxs[1]);

    return my_action;




