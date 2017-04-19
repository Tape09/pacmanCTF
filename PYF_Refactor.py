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


# global_distancer = None;
def minimax_heuristic_simple(s, action):
    weights = Counter();
    features = extract_features(s, action);
    mod_features = deepcopy(features);

    # carrying food weight
    # mod_features['carryingFood'] = np.log(1 + features['carryingFood']);
    weights['carryingFood'] = 1.5;

    # nearest food
    weights['nearestFoodDist'] = -0.156;

    # enemy pac
    if (features['teamScared'] == 1):
        weights['nearestEnemyPac'] = 10;
    else:
        weights['nearestEnemyPac'] = -0.5;

    weights['nEnemyPacs'] = -10000;

    if (features['enemyScared'] == 1):
        weights['nearestEnemyGhost'] = 0;
    else:
        mod_features['nearestEnemyGhost'] = 1.0 / (1 + features['nearestEnemyGhost'] ** 2);
        weights['nearestEnemyGhost'] = -32;

    weights['score'] = 100;
    weights['distHome'] = -0.005 * features['carryingFood'] ** 2;
    weights['enemyScared'] = 1000;
    weights['teamScared'] = -1000;

    return weights * mod_features;


def minimax_heuristic_retard(s, action):
    weights = Counter();
    features = extract_features(s, action);
    mod_features = deepcopy(features);
    # for key, item in features.iteritems():
    #     print("key:{} item:{}".format(key,item))
    # print("dist2e0:{}, dist2e1:{}".format(features["distToEnemy0"],features["distToEnemy1"]))
    # enemy 0
    weights['distToEnemy0'] = -1;

    # enemy 1
    weights['distToEnemy1'] = -1;

    # stop
    weights['Stop'] = -1

    return weights * mod_features;


def minimax_heuristic_0(s, action):
    weights = Counter();
    features = extract_features(s, action);
    mod_features = deepcopy(features);

    # carrying food weight
    # mod_features['carryingFood'] = np.log(1 + features['carryingFood']);
    weights['carryingFood'] = 1.5;

    # nearest food
    weights['nearestFoodDist'] = -0.156;

    # enemy pac
    if (features['teamScared'] == 1):
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

    if (features['enemyScared'] == 1):
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


def extract_features(gameState, action):
    features = Counter();
    my_idx = gameState.data._agentMoved;
    isRed = gameState.isOnRedTeam(my_idx);
    my_pos = gameState.getAgentState(my_idx).getPosition();
    team_idx = (my_idx + 2) % 4;
    team_pos = gameState.getAgentState(team_idx).getPosition();

    if (isRed):
        home_x = gameState.data.layout.width / 2 - 1;
        enemy_idxs = gameState.getBlueTeamIndices();
        foodList = gameState.getBlueFood().asList();
    else:
        home_x = gameState.data.layout.width / 2;
        enemy_idxs = gameState.getRedTeamIndices();
        foodList = gameState.getRedFood().asList();

    if action == "Stop":
        features['Stop'] = -1

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
    for c, idx in enumerate(enemy_idxs):
        s = 'distToEnemy' + str(c);
        ss = 'isPacEnemy' + str(c);
        sss = 'carryingFoodEnemy' + str(c);
        dist = global_distancer.getDistance(my_pos, gameState.getAgentState(idx).getPosition());
        features[s] = dist;
        features[sss] = gameState.getAgentState(idx).numCarrying;
        if (not gameState.data.agentStates[idx].isPacman):
            features[ss] = 0;
            nEnemyGhosts += 1;
            nearestEnemyGhost = min(nearestEnemyGhost, dist);
        else:
            features[ss] = 1;
            nEnemyPacs += 1;
            nearestEnemyPac = min(nearestEnemyPac, dist);

    if (nearestEnemyGhost != 9999):
        features['nearestEnemyGhost'] = nearestEnemyGhost;
    features['nEnemyGhosts'] = nEnemyGhosts;

    if (nearestEnemyPac != 9999):
        features['nearestEnemyPac'] = nearestEnemyPac;
    features['nEnemyPacs'] = nEnemyPacs;

    # score
    score = gameState.getScore();
    if (not isRed):
        score = score * -1;
    features['score'] = score;

    # dist to home
    distHome = 9999;
    distHomeEnemy0 = 9999;
    distHomeEnemy1 = 9999;
    map_height = gameState.data.layout.height;
    for y in range(map_height):
        pos = (home_x, y);
        if (gameState.hasWall(pos[0], pos[1])):
            continue;
        if (gameState.data.agentStates[my_idx].isPacman):
            distHome = min(distHome, global_distancer.getDistance(my_pos, pos));
        if (gameState.data.agentStates[enemy_idxs[0]].isPacman):
            distHomeEnemy0 = min(distHomeEnemy0,
                                 global_distancer.getDistance(gameState.getAgentState(enemy_idxs[0]).getPosition(),
                                                              pos) + 1);
        if (gameState.data.agentStates[enemy_idxs[1]].isPacman):
            distHomeEnemy1 = min(distHomeEnemy1,
                                 global_distancer.getDistance(gameState.getAgentState(enemy_idxs[1]).getPosition(),
                                                              pos) + 1);

    if (distHome != 9999):
        features['distHome'] = distHome;

    if (distHomeEnemy0 != 9999):
        features['distHomeEnemy0'] = distHomeEnemy0;

    if (distHomeEnemy1 != 9999):
        features['distHomeEnemy1'] = distHomeEnemy1;

    # Enemy scared
    enemyScared = 0;
    if (gameState.getAgentState(enemy_idxs[0]).scaredTimer > 0):
        enemyScared = 1;
    features['enemyScared'] = enemyScared;

    # team scared
    teamScared = 0;
    if (gameState.getAgentState(my_idx).scaredTimer > 0):
        teamScared = 1;
    features['teamScared'] = teamScared;

    return features;


def shit_heuristic(next_actions_states):
    return next_actions_states[np.random.choice(len(next_actions_states))]


class EnemyTracker:
    def init(self, gameState, isRed):
        self.red = isRed;
        self.first_update = True;
        self.enemy_idxs = [];
        self.enemy_edge = [];
        if (isRed):  # enemy blue
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
        self.tracker = [None] * len(all_idxs);
        self.carrying_food = [0] * len(all_idxs);

        for i in all_idxs:
            if (i in self.enemy_idxs):
                self.tracker[i] = Counter();
                self.tracker[i][gameState.getInitialAgentPosition(i)] = 1.0;

    def update(self, gameState, my_index):  # {
        # check if food got eaten
        if (self.red):  # enemy blue
            new_food_state = gameState.getRedFood();
        else:
            new_food_state = gameState.getBlueFood();

        eaten_food = [];

        for i in range(self.old_food_state.width):
            for j in range(self.old_food_state.height):
                if self.old_food_state[i][j] and not new_food_state[i][j]:
                    eaten_food.append((i, j));

        self.old_food_state = new_food_state;

        temp_trackers = {};
        measured_dists = gameState.getAgentDistances();
        for i in self.enemy_idxs:  # {
            exact_pos = gameState.getAgentPosition(i);
            if (exact_pos != None):  # {
                temp_trackers[i] = Counter();
                temp_trackers[i][exact_pos] = 1.0;
            else:
                temp_trackers[i] = Counter();
                for key, value in self.tracker[i].iteritems():  # {
                    if (value == 0.0):
                        continue;

                    if (my_index == 0 and self.first_update):
                        self.first_update = False;
                        temp_trackers[i][key] = value;
                        continue;

                    if ((
                            my_index - 1) % gameState.getNumAgents() == i):  # if this agent moved last turn, update his pos
                        p_move = np.zeros(5);
                        k = 0;
                        for direction, _ in Actions._directions.iteritems():  # {
                            pos = Actions.getSuccessor(key, direction);
                            if (not gameState.hasWall(int(pos[0]), int(pos[1]))):  # base probability
                                p_move[k] += 1;
                            if (pos not in self.tracker[i]):  # multiplier for visiting new positions
                                p_move[k] *= 2;
                            if (direction == Directions.STOP):  # multiplier for stopping
                                p_move[k] *= 0.5;

                            k += 1;
                        p_move = p_move / np.sum(p_move);

                        k = 0;
                        for direction, _ in Actions._directions.iteritems():
                            pos = Actions.getSuccessor(key, direction);
                            if (not gameState.hasWall(int(pos[0]), int(pos[1]))):
                                temp_trackers[i][pos] += p_move[k] * value;
                            k += 1;
                    else:  # if this agent did not move last turn, pretend he moved using action STOP
                        temp_trackers[i][key] = value;

                for key, value in temp_trackers[i].iteritems():  # {
                    true_dist = manhattanDistance(key, gameState.getAgentPosition(my_index));
                    if (true_dist <= 5):
                        temp_trackers[i][key] = 0.0;
                    else:
                        temp_trackers[i][key] = value * gameState.getDistanceProb(true_dist, measured_dists[i]);

        # print(self.carrying_food );
        for key in eaten_food:
            if (temp_trackers[self.enemy_idxs[0]][key] > temp_trackers[self.enemy_idxs[1]][key]):
                temp_trackers[self.enemy_idxs[0]][key] = 2;
                self.carrying_food[self.enemy_idxs[0]] += 1;
            else:
                temp_trackers[self.enemy_idxs[1]][key] = 2;
                self.carrying_food[self.enemy_idxs[1]] += 1;

        for key, _ in temp_trackers.iteritems():
            self.tracker[key] = deepcopy(temp_trackers[key]);
            self.tracker[key].normalize();

        scorediff = abs(gameState.data.scoreChange);
        if (scorediff > 0):
            if (self.carrying_food[self.enemy_idxs[0]] != 0 and self.carrying_food[self.enemy_idxs[1]] == 0):
                for key, _ in self.tracker[self.enemy_idxs[0]].iteritems():
                    if (key[0] not in self.enemy_edge):
                        self.tracker[self.enemy_idxs[0]][key] = 0.0;
                self.carrying_food[self.enemy_idxs[0]] = 0;
            elif (self.carrying_food[self.enemy_idxs[0]] == 0 and self.carrying_food[self.enemy_idxs[1]] != 0):
                for key, _ in self.tracker[self.enemy_idxs[1]].iteritems():
                    if (key[0] not in self.enemy_edge):
                        self.tracker[self.enemy_idxs[1]][key] = 0.0;
                self.carrying_food[self.enemy_idxs[1]] = 0;
            elif (self.carrying_food[self.enemy_idxs[0]] != 0 and self.carrying_food[self.enemy_idxs[1]] != 0):
                most_likely_enemy = self.enemy_idxs[0];
                less_likely_enemy = self.enemy_idxs[1];
                max_prob = -1.0;
                for key, _ in self.tracker[self.enemy_idxs[1]].iteritems():
                    if (key[0] in self.enemy_edge):
                        if (max_prob < self.tracker[self.enemy_idxs[1]][key]):
                            max_prob = self.tracker[self.enemy_idxs[1]][key];
                            most_likely_enemy = self.enemy_idxs[1];
                            less_likely_enemy = self.enemy_idxs[0];
                for key, _ in self.tracker[self.enemy_idxs[0]].iteritems():
                    if (key[0] in self.enemy_edge):
                        if (max_prob < self.tracker[self.enemy_idxs[0]][key]):
                            max_prob = self.tracker[self.enemy_idxs[0]][key];
                            most_likely_enemy = self.enemy_idxs[0];
                            less_likely_enemy = self.enemy_idxs[1];

                diff = self.carrying_food[most_likely_enemy] - scorediff;
                if (diff < 0):
                    self.carrying_food[most_likely_enemy] = 0;
                    self.carrying_food[less_likely_enemy] = max(0, self.carrying_food[less_likely_enemy] - diff);
                elif (diff > 0):
                    self.carrying_food[most_likely_enemy] = 0;
                    self.carrying_food[less_likely_enemy] += diff;
                else:
                    self.carrying_food[most_likely_enemy] = 0;

                for key, _ in self.tracker[most_likely_enemy].iteritems():
                    if (key[0] not in self.enemy_edge):
                        self.tracker[self.enemy_idxs[1]][key] = 0.0;

                for key, _ in temp_trackers.iteritems():
                    self.tracker[key].normalize();

    def estimate_enemy_pos(self, enemyIndex):
        max_prob = -1;
        pos = 0;
        for key, value in self.tracker[enemyIndex].iteritems():
            if (value > max_prob):
                max_prob = deepcopy(value);
                pos = deepcopy(key);
        return pos, max_prob;

    def estimate_enemy_carrying(self, enemyIndex):
        return self.carrying_food[enemyIndex];

    def update_eaten_agent(self, gameState, index_eaten):
        self.tracker[index_eaten] = Counter();
        self.tracker[index_eaten][gameState.getInitialAgentPosition(index_eaten)] = 1.0;
        self.carrying_food[index_eaten] = 0;


tracker = EnemyTracker();


class SharedMemory:
    def __init__(self):
        self.pill_time = 0;


shared = SharedMemory();


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='AlphaBetaAgent', second='AlphaBetaAgent'):
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

class AlphaBetaAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        t0 = time.time();
        CaptureAgent.registerInitialState(self, gameState)

        global global_distancer;
        global_distancer = deepcopy(self.distancer);
        tracker.init(gameState, self.red);
        self.first = False;

        if (self.red):  # enemy blue
            self.our_idxs = gameState.getRedTeamIndices()
            self.enemy_idxs = gameState.getBlueTeamIndices();
            self.enemy_edge = gameState.data.layout.width / 2;
        else:
            self.our_idxs = gameState.getBlueTeamIndices()
            self.enemy_idxs = gameState.getRedTeamIndices();
            self.enemy_edge = gameState.data.layout.width / 2 - 1;

        # oracle.red = self.red;


        enemy0_pos = gameState.getInitialAgentPosition(self.enemy_idxs[0]);
        enemy1_pos = gameState.getInitialAgentPosition(self.enemy_idxs[1]);
        e0_config = Configuration(enemy0_pos, Directions.STOP);
        e1_config = Configuration(enemy1_pos, Directions.STOP);
        if (self.red):
            isPacman0 = enemy0_pos[0] < gameState.data.layout.width / 2;
            isPacman1 = enemy1_pos[0] < gameState.data.layout.width / 2;
        else:
            isPacman0 = enemy0_pos[0] >= gameState.data.layout.width / 2;
            isPacman1 = enemy1_pos[0] >= gameState.data.layout.width / 2;
        e0_state = AgentState(e0_config, isPacman0);
        e1_state = AgentState(e1_config, isPacman1);
        e0_state.numCarrying = 0;
        e1_state.numCarrying = 0;

        gameState.data.agentStates[self.enemy_idxs[0]] = e0_state;
        gameState.data.agentStates[self.enemy_idxs[1]] = e1_state;
        gameState.data._agentMoved = 3;

        t1 = time.time();

        time_spent = t1 - t0;

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        shared.pill_time = max(0, shared.pill_time - 2);
        # if(self.first and self.index == 0):
        #    gameState.data._agentMoved = 3;
        self.first = False;

        my_index = self.index;
        my_pos = gameState.getAgentPosition(self.index);

        # UPDATE TRACKING AND GAME STATE BELIEF
        tracker.update(gameState, self.index);
        self.displayDistributionsOverPositions(tracker.tracker);

        enemy0_pos, enemy0_prob = tracker.estimate_enemy_pos(self.enemy_idxs[0]);
        enemy1_pos, enemy1_prob = tracker.estimate_enemy_pos(self.enemy_idxs[1]);
        enemy0_carrying = tracker.estimate_enemy_carrying(self.enemy_idxs[0]);
        enemy1_carrying = tracker.estimate_enemy_carrying(self.enemy_idxs[1]);

        enemy_positions = [enemy0_pos, enemy1_pos];

        # Generate AgentState for enemy agents: need configuration and isPacman
        # Configuration: position and direction
        e0_config = Configuration(enemy0_pos, Directions.STOP);
        e1_config = Configuration(enemy1_pos, Directions.STOP);
        if (self.red):
            isPacman0 = enemy0_pos[0] < gameState.data.layout.width / 2;
            isPacman1 = enemy1_pos[0] < gameState.data.layout.width / 2;
        else:
            isPacman0 = enemy0_pos[0] >= gameState.data.layout.width / 2;
            isPacman1 = enemy1_pos[0] >= gameState.data.layout.width / 2;
        e0_state = AgentState(e0_config, isPacman0);
        e1_state = AgentState(e1_config, isPacman1);
        e0_state.numCarrying = enemy0_carrying;
        e1_state.numCarrying = enemy1_carrying;
        e0_state.scaredTimer = shared.pill_time;
        e1_state.scaredTimer = shared.pill_time;

        gameState.data.agentStates[self.enemy_idxs[0]] = e0_state;
        gameState.data.agentStates[self.enemy_idxs[1]] = e1_state;
        gameState.data._agentMoved = (my_index - 1) % gameState.getNumAgents();

        # print(gameState)
        actions_states = [(a, gameState.generateSuccessor(my_index, a)) for a in actions];

        # my_action = oracle.next_move(gameState);
        # my_action, _ = less_shit_heuristic(actions_states);
        bestscore = -10000000
        print (actions_states)
        for action, state in actions_states:
            alpha = -100000
            beta = 100000
            depth = 8

            score = self.alphaBeta(action, state, depth, alpha, beta)
            if score > bestscore:
                bestscore = score
                my_action = action
            print ("action:{} score:{},bestscore:{}".format(action, score, bestscore))
        # pause()
        if (Actions.getSuccessor(my_pos, my_action) == enemy0_pos):
            tracker.update_eaten_agent(gameState, self.enemy_idxs[0]);
        if (Actions.getSuccessor(my_pos, my_action) == enemy1_pos):
            tracker.update_eaten_agent(gameState, self.enemy_idxs[1]);
        if (Actions.getSuccessor(my_pos, my_action) in self.getCapsules(gameState)):
            shared.pill_time = 40;

        return my_action;

    def alphaBeta(self, actionNow, gameState, depth, alpha, beta):
        # alpha is the best choice for max, beta is the best choice for min
        agentIndex = (gameState.data._agentMoved + 1) % 4
        actionStates = [(a, gameState.generateSuccessor(agentIndex, a)) for a in gameState.getLegalActions(agentIndex)]
        if depth <= 0:
            score = minimax_heuristic_retard(gameState, actionNow)
            # print("Depth 0 , score: {}".format(score))
            return score

        if self.isGameOver(gameState):  # game over,terminal state
            return self.util(gameState)
        else:  # game is not over!
            # print("Our indices: {}, gamestateMoved: {}".format(self.our_idxs,agentIndex))
            if agentIndex in self.our_idxs:  # max layer
                v = -np.inf
                for action, nextState in actionStates:
                    evaluation = self.alphaBeta(action, nextState, depth-1, alpha, beta)
                    # print ("Own evaluation:{},depth:{} alpha:{} beta:{}".format(evaluation,depth,alpha,beta))
                    v = max(evaluation,v)
                    if v>=beta:
                        return v
                    alpha = max(alpha,v)
            else:
                v = np.inf
                for action, nextState in actionStates:
                    evaluation = self.alphaBeta(action, nextState, depth-1, alpha, beta)
                    # print("Ene evaluation: {},depth:{} alpha:{} beta:{}".format(evaluation,depth,alpha,beta))
                    v = min(v,evaluation)
                    if v<=alpha:
                        return v
                    beta = min(beta,v)
        return v

    def max_abpruning(self,state,alpha,beta):#return a utility value
        my_index = (state.data._agentMoved + 1) % 4
        if self.isGameOver(state):
            return self.util(state)
        v = -np.inf
        actions = state.getLegalActions(my_index)
        for a in actions:
            nextState = state.generateSuccessor(my_index,a)
            v = max(v,self.min_abpruning(nextState,alpha,beta))
            if v>= beta:
                return v
            alpha = max(alpha,v)
        return v

    def min_abpruning(self,state,alpha,beta):
        my_index = (state.data._agentMoved + 1) % 4
        if self.isGameOver(state):
            return self.util(state)
        v = np.inf
        actions = state.getLegalActions(my_index)
        for a in actions:
            nextState = state.generateSuccessor(my_index, a)
            v = min(v, self.min_abpruning(nextState, alpha, beta))
            if v <= alpha:
                return v
            alpha = min(beta, v)
        return v

    def abPruning(self,state):#return an action
        alpha = -np.inf
        beta = np.inf
        v = self.max_abpruning(state,alpha,beta)

    def isGameOver(self, state):
        return state.isOver() or state.data.timeleft == 0

    def util(self, state):
        finalScore = state.getScore()
        if self.red:
            return 10000 * finalScore
        else:
            return -10000 * finalScore







