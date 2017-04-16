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
from game import *
from copy import deepcopy
import numpy as np
from operator import itemgetter

#global_distancer = None;

def extract_features(gameState):
    features = Counter();
    my_idx = gameState.data._agentMoved;  
    isRed = gameState.isOnRedTeam(my_idx);
    my_pos = gameState.getAgentState(my_idx).getPosition();
    team_idx = (my_idx + 2) % 4;
    team_pos = gameState.getAgentState(team_idx).getPosition();

    if(isRed):
        home_x = gameState.data.layout.width / 2 - 1;
        enemy_idxs = gameState.getBlueTeamIndices();
        foodList = gameState.getBlueFood().asList();
    else:
        home_x = gameState.data.layout.width / 2;
        enemy_idxs = gameState.getRedTeamIndices();
        foodList = gameState.getRedFood().asList();

    # carrying food
    features['carryingFood'] = gameState.getAgentState(my_idx).numCarrying;

    # dist to nearest enemy food
    nearestFoodDist = min([global_distancer.getDistance(my_pos, food) for food in foodList]);
    features['nearestFoodDist'] = nearestFoodDist;    

    # dist to team mate
    features['distToTeam'] = global_distancer.getDistance(my_pos, team_pos);

    # dist to nearest enemies, pacmen, ghosts, carrying, and #
    nearestEnemyGhost = 9999;   
    nearestEnemyPac = 9999;
    nEnemyGhosts = 0;
    nEnemyPacs = 0;
    for c,idx in enumerate(enemy_idxs):
        s = 'distToEnemy' + str(c);
        ss = 'isPacEnemy' + str(c);
        sss = 'carryingFoodEnemy' + str(c);
        dist = global_distancer.getDistance(my_pos, gameState.getAgentState(idx).getPosition());
        features[s] = dist;
        features[sss] = gameState.getAgentState(idx).numCarrying;
        if(not gameState.data.agentStates[idx].isPacman):
            features[ss] = 0;
            nEnemyGhosts += 1;
            nearestEnemyGhost = min(nearestEnemyGhost,dist);     
        else:
            features[ss] = 1;
            nEnemyPacs += 1;
            nearestEnemyPac = min(nearestEnemyPac,dist);    

    if(nearestEnemyGhost != 9999):
        features['nearestEnemyGhost'] = nearestEnemyGhost;
    features['nEnemyGhosts'] = nEnemyGhosts;

    if(nearestEnemyPac != 9999):
        features['nearestEnemyPac'] = nearestEnemyPac;
    features['nEnemyPacs'] = nEnemyPacs;

    # score
    score = gameState.getScore();
    if(not isRed):
        score = score * -1;
    features['score'] = score;

    # dist to home
    distHome = 9999;
    distHomeEnemy0 = 9999;
    distHomeEnemy1 = 9999;
    map_height = gameState.data.layout.height;
    for y in range(map_height):
        pos = (home_x, y);
        if(gameState.hasWall(pos[0],pos[1])):
            continue;
        if(gameState.data.agentStates[my_idx].isPacman):
            distHome = min(distHome, global_distancer.getDistance(my_pos, pos));
        if(gameState.data.agentStates[enemy_idxs[0]].isPacman):
            distHomeEnemy0 = min(distHomeEnemy0, global_distancer.getDistance(gameState.getAgentState(enemy_idxs[0]).getPosition(), pos) + 1);
        if(gameState.data.agentStates[enemy_idxs[1]].isPacman):
            distHomeEnemy1 = min(distHomeEnemy1, global_distancer.getDistance(gameState.getAgentState(enemy_idxs[1]).getPosition(), pos) + 1);
    
    if(distHome != 9999):
        features['distHome'] = distHome;

    if(distHomeEnemy0 != 9999):
        features['distHomeEnemy0'] = distHomeEnemy0;

    if(distHomeEnemy1 != 9999):
        features['distHomeEnemy1'] = distHomeEnemy1;

    # Enemy scared
    enemyScared = 0;
    if(gameState.getAgentState(enemy_idxs[0]).scaredTimer > 0):
        enemyScared = 1;
    features['enemyScared'] = enemyScared;

    # team scared    
    teamScared = 0;
    if(gameState.getAgentState(my_idx).scaredTimer > 0):
        teamScared = 1;
    features['teamScared'] = teamScared;

    return features;

def team_heuristic(gameState, red):
    features = Counter();       
    map_height = gameState.data.layout.height;

    if(red):
        my_idxs = gameState.getRedTeamIndices();        
        enemy_idxs = gameState.getBlueTeamIndices();        
        home_x = gameState.data.layout.width / 2 - 1;
        enemy_food = gameState.getBlueFood().asList();
        my_food = gameState.getRedFood().asList();
        my_pills = gameState.getRedCapsules();
        enemy_pills = gameState.getBlueCapsules();
        
    else:
        my_idxs = gameState.getBlueTeamIndices();
        enemy_idxs = gameState.getRedTeamIndices();
        home_x = gameState.data.layout.width / 2;
        my_food = gameState.getBlueFood().asList();
        enemy_food = gameState.getRedFood().asList();
        my_pills = gameState.getBlueCapsules();
        enemy_pills = gameState.getRedCapsules();
    
    my_carrying = [gameState.getAgentState(i).numCarrying for i in my_idxs]
    enemy_carrying = [gameState.getAgentState(i).numCarrying for i in enemy_idxs]

    enemy_pos = [gameState.getAgentState(i).getPosition() for i in enemy_idxs];
    my_pos = [gameState.getAgentState(i).getPosition() for i in my_idxs];

    my_pac = [gameState.data.agentStates[i].isPacman for i in my_idxs];
    enemy_pac = [gameState.data.agentStates[i].isPacman for i in enemy_idxs];

    my_scared = gameState.getAgentState(my_idxs[0]).scaredTimer > 0;
    enemy_scared = gameState.getAgentState(enemy_idxs[0]).scaredTimer > 0;

    ma_ea_dists = np.array([[global_distancer.getDistance(i, j) for j in enemy_pos] for i in my_pos]);

    ma_ef_dists = np.array([[global_distancer.getDistance(i, j) for j in enemy_food] for i in my_pos]);
    ma_ep_dists = np.array([[global_distancer.getDistance(i, j) for j in enemy_pills] for i in my_pos]);

    ea_mf_dists = np.array([[global_distancer.getDistance(i, j) for j in my_food] for i in enemy_pos]);
    ea_mp_dists = np.array([[global_distancer.getDistance(i, j) for j in my_pills] for i in enemy_pos]);

    my_dist_home = [9999,9999];
    enemy_dist_home = [9999,9999];

    for y in range(map_height):
        hpos = (home_x, y);
        if(gameState.hasWall(hpos[0],hpos[1])):
            continue;

        for i,pos in enumerate(my_pos):
            if(my_pac[i]):
                my_dist_home[i] = 0;
            else:
                my_dist_home[i] = min(my_dist_home[i], global_distancer.getDistance(pos,hpos));

        for i,pos in enumerate(enemy_pos):
            if(enemy_pac[i]):
                enemy_dist_home[i] = 0;
            else:
                enemy_dist_home[i] = min(my_dist_home[i], global_distancer.getDistance(pos,hpos) + 1);



    return -np.sum(np.min(ma_ef_dists,axis=1)) + 2*np.sum(my_carrying);

def symmetric_heuristic(gameState, isRed):
    return team_heuristic(gameState,isRed) - team_heuristic(gameState,not isRed);


def shit_heuristic(next_actions_states):
    return next_actions_states[np.random.choice(len(next_actions_states))];

def less_shit_heuristic(next_actions_states):
    
    best_value = -9999999;
    best_action = None;
    best_state = None;

    for i in range(len(next_actions_states)):
        a,s = next_actions_states[i];
        weights = Counter();   
        features = extract_features(s);
        mod_features = deepcopy(features);

        # carrying food weight
        #mod_features['carryingFood'] = np.log(1 + features['carryingFood']);
        weights['carryingFood'] = 1.5;

        # nearest food
        weights['nearestFoodDist'] = -0.156;

        # enemy pac
        if(features['teamScared'] == 1):
            weights['nearestEnemyPac'] = 10;      
        else:
            weights['nearestEnemyPac'] = -1;

        # enemy 0
        weights['distToEnemy0'] = -1 * features['carryingFoodEnemy0']
        mod_features['distHomeEnemy0'] = 1.0 / (1 + features['distHomeEnemy0']);
        weights['distHomeEnemy0'] = features['carryingFoodEnemy0'] * -1;

        # enemy 1
        weights['distToEnemy1'] = -1 * features['carryingFoodEnemy1']
        mod_features['distHomeEnemy1'] = 1.0 / (1 + features['distHomeEnemy1']);
        weights['distHomeEnemy1'] = features['carryingFoodEnemy1'] * -1;


   
        weights['nEnemyPacs'] = -10000;
        
        if(features['enemyScared'] == 1):
            weights['nearestEnemyGhost'] = 0;
        else:
            mod_features['nearestEnemyGhost'] = 1.0 / (1 + features['nearestEnemyGhost'] ** 2);
            weights['nearestEnemyGhost'] = -32;


        mod_features['distToTeam'] = 1.0 / (1 + mod_features['distToTeam']);
        weights['distToTeam'] = -1;
            

        weights['score'] = 100;
        weights['distHome'] = -0.005 * features['carryingFood'] ** 2;
        weights['enemyScared'] = 1000;
        weights['teamScared'] = -1000;

        value = weights * mod_features;

        #print a;
        #print value;
        if(value > best_value):
            best_value = value;
            best_action = a;
            best_state = s;
            best_features = mod_features;
            best_weights = weights;


    #print(best_features)
    #print(best_weights)
    #print(best_action)
    #print("")

    #for key, _ in best_weights.iteritems():
    #    if(key in best_features):
    #        print(key, best_weights[key] * best_features[key]);

    #print("");
    return (best_action,best_state);

def minimax_heuristic_0(s):
    weights = Counter();   
    features = extract_features(s);
    mod_features = deepcopy(features);

    # carrying food weight
    #mod_features['carryingFood'] = np.log(1 + features['carryingFood']);
    weights['carryingFood'] = 1.5;

    # nearest food
    weights['nearestFoodDist'] = -0.156;

    # enemy pac
    if(features['teamScared'] == 1):
        weights['nearestEnemyPac'] = 10;      
    else:
        weights['nearestEnemyPac'] = -0.5;

    # enemy 0
    weights['distToEnemy0'] = -1 * features['carryingFoodEnemy0']
    mod_features['distHomeEnemy0'] = 1.0 / (1 + features['distHomeEnemy0']);
    weights['distHomeEnemy0'] = features['carryingFoodEnemy0'] * -1;

    # enemy 1
    weights['distToEnemy1'] = -1 * features['carryingFoodEnemy1']
    mod_features['distHomeEnemy1'] = 1.0 / (1 + features['distHomeEnemy1']);
    weights['distHomeEnemy1'] = features['carryingFoodEnemy1'] * -1;
   
    weights['nEnemyPacs'] = -10000;
        
    if(features['enemyScared'] == 1):
        weights['nearestEnemyGhost'] = 0;
    else:
        mod_features['nearestEnemyGhost'] = 1.0 / (1 + features['nearestEnemyGhost'] ** 2);
        weights['nearestEnemyGhost'] = -32;


    mod_features['distToTeam'] = 1.0 / (1 + mod_features['distToTeam']);
    weights['distToTeam'] = -1;
            

    weights['score'] = 100;
    weights['distHome'] = -0.005 * features['carryingFood'] ** 2;
    weights['enemyScared'] = 1000;
    weights['teamScared'] = -1000;

    return weights * mod_features;

def minimax_heuristic_retard(s):
    weights = Counter();   
    features = extract_features(s);
    mod_features = deepcopy(features);

     # enemy 0
    weights['distToEnemy0'] = -1;

    # enemy 1
    weights['distToEnemy1'] = -1;

    #for key, _ in weights.iteritems():
    #    if(key in mod_features):
    #        print(key,weights[key] * mod_features[key]);

    return weights * mod_features;

def minimax_heuristic_simple(s):
    weights = Counter();   
    features = extract_features(s);
    mod_features = deepcopy(features);

    
    
    # carrying food weight
    #mod_features['carryingFood'] = np.log(1 + features['carryingFood']);
    weights['carryingFood'] = 1.5;

    # nearest food
    weights['nearestFoodDist'] = -0.156;

    # enemy pac
    if(features['teamScared'] == 1):
        weights['nearestEnemyPac'] = 10;      
    else:
        weights['nearestEnemyPac'] = -0.5;

   
    weights['nEnemyPacs'] = -10000;
        
    if(features['enemyScared'] == 1):
        weights['nearestEnemyGhost'] = 0;
    else:
        mod_features['nearestEnemyGhost'] = 1.0 / (1 + features['nearestEnemyGhost'] ** 2);
        weights['nearestEnemyGhost'] = -32;
            

    weights['score'] = 100;
    weights['distHome'] = -0.005 * features['carryingFood'] ** 2;
    weights['enemyScared'] = 1000;
    weights['teamScared'] = -1000;

    return weights * mod_features;




class MCTS:
    def __init__(self, heuristicFcn = less_shit_heuristic):
        self.states_played = {};
        self.states_won = {};
        self.red = True;
        self.exploration_c = 2.0;
        self.time_limit = 0.9;
        self.heuristicFcn = heuristicFcn;

        
    def next_move(self, gameState):
        # find whos turn it is
        turn = (gameState.data._agentMoved + 1) % gameState.getNumAgents();       

        # get legal moves  
        actions = gameState.getLegalActions(turn);

        # if trivial (this should never happen)
        if(len(actions) == 0):
            return None;
        elif(len(actions) == 1):
            return actions[0];

        t0 = time.time();
        while (time.time() - t0 < self.time_limit):
            self.simulate(gameState); #populate the tree, record stats in states played and states won

        actions_states = [(a,gameState.generateSuccessor(turn,a)) for a in actions];    
        win_chance, action = max((self.states_won.get(s, 0) / self.states_played.get(s, 1), a) for a, s in actions_states);
        print(len(self.states_played))
        print(action,win_chance)
        return action;


    def simulate(self, gameState): # something is fucked
        visited_states = set();        
        expand = True;
        redWin = None;
        state = deepcopy(gameState);

        states_played = self.states_played;
        states_won = self.states_won;

        while True:
            turn = (state.data._agentMoved + 1) % state.getNumAgents();
            actions = state.getLegalActions(turn);

            next_actions_states = [(a,state.generateSuccessor(turn,a)) for a in actions];

            #my_action = np.random.choice(actions);
            #state = state.generateSuccessor(turn,my_action);

            if all(states_played.get(s) for a, s in next_actions_states):
                log_total = np.log(sum(states_played[s] for a, s in next_actions_states));
                value, my_action, state = max(((states_won[s] / states_played[s]) + self.exploration_c * np.sqrt(log_total / states_played[s]), a, s) for a, s in next_actions_states); # UCB1
            else:
                my_action, state = self.heuristicFcn(next_actions_states); # use some heuristic here
                #print(my_action);
                #my_action, state = np.random.choice(next_actions_states); # use some heuristic here


            if(expand and state not in self.states_played):
                expand = False;
                self.states_played[state] = 0;
                self.states_won[state] = 0;

            visited_states.add(state);
            
            if state.isOver() or state.data.timeleft == 0: # end of the run
                if (state.getScore() > 0):
                    redWin = True;
                elif(state.getScore() < 0):
                    redWin = False;
                print(redWin);
                break;

        for s in visited_states:
            if(s not in self.states_played):
                continue;
            self.states_played[s] += 1;
            if(redWin != None):
                if(self.red and redWin):
                    self.states_won[s] += 1;


        return None;

oracle = MCTS();
    
class EnemyTracker: 
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
        self.carrying_food = [0]*len(all_idxs);

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
                            if(not gameState.hasWall(int(pos[0]),int(pos[1]))): # base probability
                                p_move[k] += 1;
                            if(pos not in self.tracker[i]): # multiplier for visiting new positions                      
                                p_move[k] *= 2;
                            if(direction == Directions.STOP): # multiplier for stopping
                                p_move[k] *= 0.5;
                                
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
                #}
                
            #}
        #}
                
        #print(self.carrying_food );
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
                   
                for key, _ in temp_trackers.iteritems():
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

    def estimate_enemy_carrying(self,enemyIndex):
        return self.carrying_food[enemyIndex];

    def update_eaten_agent(self,gameState,index_eaten):
        self.tracker[index_eaten] = Counter();
        self.tracker[index_eaten][gameState.getInitialAgentPosition(index_eaten)] = 1.0;
        self.carrying_food[index_eaten] = 0;

tracker = EnemyTracker();

class SharedMemory:
    def __init__(self):
        self.pill_time = 0;
        self.red = False;
        
shared = SharedMemory();

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
    t0 = time.time();
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

    global global_distancer;
    global_distancer = deepcopy(self.distancer);


    tracker.init(gameState,self.red);
	

    self.first = False;    

    if(self.red): # enemy blue
        self.our_idxs = gameState.getRedTeamIndices();
        self.enemy_idxs = gameState.getBlueTeamIndices();
        self.enemy_edge = gameState.data.layout.width / 2;
    else:
        self.our_idxs = gameState.getBlueTeamIndices();
        self.enemy_idxs = gameState.getRedTeamIndices();
        self.enemy_edge = gameState.data.layout.width / 2 - 1;

    oracle.red = self.red;

    
    enemy0_pos = gameState.getInitialAgentPosition(self.enemy_idxs[0]);
    enemy1_pos = gameState.getInitialAgentPosition(self.enemy_idxs[1]);
    e0_config = Configuration(enemy0_pos,Directions.STOP);
    e1_config = Configuration(enemy1_pos,Directions.STOP);
    if(self.red):
        isPacman0 = enemy0_pos[0] < gameState.data.layout.width/2;
        isPacman1 = enemy1_pos[0] < gameState.data.layout.width/2;
    else:
        isPacman0 = enemy0_pos[0] >= gameState.data.layout.width/2;
        isPacman1 = enemy1_pos[0] >= gameState.data.layout.width/2;
    e0_state = AgentState(e0_config,isPacman0);
    e1_state = AgentState(e1_config,isPacman1);
    e0_state.numCarrying = 0;
    e1_state.numCarrying = 0;

    gameState.data.agentStates[self.enemy_idxs[0]] = e0_state;
    gameState.data.agentStates[self.enemy_idxs[1]] = e1_state;
    gameState.data._agentMoved = 3;

    t1 = time.time();

    time_spent = t1-t0;
    oracle.time_limit = 14.0 - time_spent;

    #oracle.next_move(gameState);
    oracle.time_limit = 0.95;
    
    shared.red = self.red;



  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    shared.pill_time = max(0, shared.pill_time - 2);
    #if(self.first and self.index == 0):        
    #    gameState.data._agentMoved = 3;
    self.first = False;

    my_index = self.index;
    my_pos = gameState.getAgentPosition(self.index);

    # UPDATE TRACKING AND GAME STATE BELIEF
    tracker.update(gameState,self.index);    
    self.displayDistributionsOverPositions(tracker.tracker);

    enemy0_pos, enemy0_prob = tracker.estimate_enemy_pos(self.enemy_idxs[0]);
    enemy1_pos, enemy1_prob = tracker.estimate_enemy_pos(self.enemy_idxs[1]);
    enemy0_carrying = tracker.estimate_enemy_carrying(self.enemy_idxs[0]);    
    enemy1_carrying = tracker.estimate_enemy_carrying(self.enemy_idxs[1]);    

    enemy_positions = [enemy0_pos, enemy1_pos];

    #Generate AgentState for enemy agents: need configuration and isPacman
    # Configuration: position and direction
    e0_config = Configuration(enemy0_pos,Directions.STOP);
    e1_config = Configuration(enemy1_pos,Directions.STOP);
    if(self.red):
        isPacman0 = enemy0_pos[0] < gameState.data.layout.width/2;
        isPacman1 = enemy1_pos[0] < gameState.data.layout.width/2;
    else:
        isPacman0 = enemy0_pos[0] >= gameState.data.layout.width/2;
        isPacman1 = enemy1_pos[0] >= gameState.data.layout.width/2;
    e0_state = AgentState(e0_config,isPacman0);
    e1_state = AgentState(e1_config,isPacman1);
    e0_state.numCarrying = enemy0_carrying;
    e1_state.numCarrying = enemy1_carrying;
    e0_state.scaredTimer = shared.pill_time;
    e1_state.scaredTimer = shared.pill_time;

    gameState.data.agentStates[self.enemy_idxs[0]] = e0_state;
    gameState.data.agentStates[self.enemy_idxs[1]] = e1_state;
    gameState.data._agentMoved = (my_index - 1) % gameState.getNumAgents(); 


    #print(gameState)
    actions_states = [(a,gameState.generateSuccessor(my_index,a)) for a in actions]; 

    actions_rewards = [(a,symmetric_heuristic(s,self.red)) for a,s in actions_states];

    #print(actions_rewards)

    #my_action = max(actions_rewards,key=itemgetter(1))[0];
    #my_action = oracle.next_move(gameState);
    #my_action, _ = less_shit_heuristic(actions_states);
    
    
    my_action = self.alphaBetaStart(gameState,4);
    #print(symmetric_heuristic(gameState,self.red))

    print my_action

    if(Actions.getSuccessor(my_pos,my_action) == enemy0_pos):
        tracker.update_eaten_agent(gameState,self.enemy_idxs[0]);
    if(Actions.getSuccessor(my_pos,my_action) == enemy1_pos):
        tracker.update_eaten_agent(gameState,self.enemy_idxs[1]);
    if(Actions.getSuccessor(my_pos,my_action) in self.getCapsules(gameState)):
        shared.pill_time = 40;

    return my_action;

  def alphaBetaStart(self,gameState,depth):
    actions = gameState.getLegalActions(self.index)
    actions_states = [(a,gameState.generateSuccessor(self.index,a)) for a in actions]; 

    bestscore = -10000000
    for a, s in actions_states:
       alpha = -1000000
       beta =   1000000
       score = self.alphaBeta(s,depth,alpha,beta)
       print(a,score)
       if score>bestscore:
            bestscore = score;
            my_action = a;

   
    return my_action;

  def alphaBeta(self, gameState, depth,alpha,beta):
    #alpha is the best choice for max, beta is the best choice for min
    agentIndex = (gameState.data._agentMoved + 1) % 4; 
    agentRed = gameState.isOnRedTeam(agentIndex);
    
    if gameState.isOver() or gameState.data.timeleft == 0:#game over,terminal state
        finalScore = gameState.getScore()
        if self.red:
            return 100000*finalScore
        else:
            return -100000*finalScore

    if depth <= 0:
        return symmetric_heuristic(gameState,self.red), None;

    actions_states = [(a, gameState.generateSuccessor(agentIndex, a)) for a in gameState.getLegalActions(agentIndex)]    


    if(agentRed == self.red):
        v = -1000000000
        for a,s in actions_states:
            evaluation = self.alphaBeta(s,depth-1,alpha,beta)[0];
            if(evaluation > v):
                best_action = a;
                v = evaluation;         
            #print('MyTeam Best: {} Eval: {}'.format(best,evaluation))
            alpha = max(alpha,v);
            if beta <= alpha:
                break #beta prune
    else:
        v = 1000000000
        for a, s in actions_states:
            evaluation = self.alphaBeta(s,depth-1,alpha,beta)[0];
            if(evaluation < v):
                best_action = a;
                v = evaluation;
            #print('Enemy Best: {} Eval: {}'.format(best, evaluation))
            beta = min(beta,v);
            if beta <= alpha:
                break  # alpha prune
    return v,best_action




