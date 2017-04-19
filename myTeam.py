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
import numpy as np
from copy import deepcopy


def extract_features(gameState):
    features = Counter()
    my_idx = gameState.data._agentMoved
    # if my_idx is None:
    #     my_idx = 0
    #
    # print (my_idx)
    isRed = gameState.isOnRedTeam(my_idx)
    my_pos = gameState.getAgentState(my_idx).getPosition()

    if (isRed):
        home_x = gameState.data.layout.width / 2 - 1
        enemy_idxs = gameState.getBlueTeamIndices()
        foodList = gameState.getBlueFood().asList()
    else:
        home_x = gameState.data.layout.width / 2
        enemy_idxs = gameState.getRedTeamIndices()
        foodList = gameState.getRedFood().asList()

    # carrying food
    features['carryingFood'] = gameState.getAgentState(my_idx).numCarrying

    # dist to nearest enemy food
    nearestFoodDist = min([global_distancer.getDistance(my_pos, food) for food in foodList])
    features['nearestFoodDist'] = nearestFoodDist

    # dist to nearest enemy pacman, and number of enemy pacmen
    nearestEnemyPac = 9999
    nEnemyPacs = 0;
    for idx in enemy_idxs:
        if (gameState.data.agentStates[idx].isPacman):
            nEnemyPacs += 1
            nearestEnemyPac = min(nearestEnemyPac,
                                  global_distancer.getDistance(my_pos, gameState.getAgentState(idx).getPosition()))
    if (nearestEnemyPac != 9999):
        features['nearestEnemyPac'] = nearestEnemyPac
    features['nEnemyPacs'] = nEnemyPacs

    # dist to nearest enemy ghost, and #
    nearestEnemyGhost = 9999
    nEnemyGhosts = 0
    for idx in enemy_idxs:
        if (not gameState.data.agentStates[idx].isPacman):
            nEnemyGhosts += 1
            nearestEnemyGhost = min(nearestEnemyGhost,
                                    global_distancer.getDistance(my_pos, gameState.getAgentState(idx).getPosition()))
    if (nearestEnemyGhost != 9999):
        features['nearestEnemyGhost'] = nearestEnemyGhost
    features['nEnemyGhosts'] = nEnemyGhosts

    # score
    score = gameState.getScore()
    if (not isRed):
        score = score * -1
    features['score'] = score

    # dist to home
    if (gameState.data.agentStates[my_idx].isPacman):
        distHome = 9999;
        map_height = gameState.data.layout.height
        for y in range(map_height):
            pos = (home_x, y)
            if (gameState.hasWall(pos[0], pos[1])):
                continue
            distHome = min(distHome, global_distancer.getDistance(my_pos, pos))
        features['distHome'] = distHome

    # Enemy scared
    enemyScared = 0
    if (gameState.getAgentState(enemy_idxs[0]).scaredTimer > 0):
        enemyScared = 1
    features['enemyScared'] = enemyScared

    # team scared
    teamScared = 0
    if (gameState.getAgentState(my_idx).scaredTimer > 0):
        teamScared = 1
    features['teamScared'] = teamScared

    return features

# class EnemyTracker:  ########################## TODO: Update positions of eaten agents to starting position
#   # counter object for keeping track of enemy positions
#   # self.tracker : dict of counters
#   # self.enemy_idxs = [];
#
#   def init(self, gameState, isRed):
#     self.red = isRed;
#     self.first_update = True;
#     self.enemy_idxs = [];
#     if (isRed):  # enemy blue
#       self.enemy_idxs = gameState.getBlueTeamIndices();
#       self.old_food_state = gameState.getRedFood();
#     else:
#       self.enemy_idxs = gameState.getRedTeamIndices();
#       self.old_food_state = gameState.getBlueFood();
#
#     all_idxs = gameState.getRedTeamIndices();
#     all_idxs.extend(gameState.getBlueTeamIndices());
#     all_idxs.sort();
#     self.tracker = [None] * len(all_idxs);
#
#     for i in all_idxs:
#       if (i in self.enemy_idxs):
#         self.tracker[i] = Counter();
#         self.tracker[i][gameState.getInitialAgentPosition(i)] = 1.0;
#
#   def update(self, gameState, my_index):  # {
#     # check if food got eaten
#     if (self.red):  # enemy blue
#       new_food_state = gameState.getRedFood();
#     else:
#       new_food_state = gameState.getBlueFood();
#
#     eaten_food = [];
#
#     for i in range(self.old_food_state.width):
#       for j in range(self.old_food_state.height):
#         if self.old_food_state[i][j] and not new_food_state[i][j]:
#           eaten_food.append((i, j));
#
#     self.old_food_state = new_food_state;
#
#     measured_dists = gameState.getAgentDistances();
#     for i in self.enemy_idxs:  # {
#       exact_pos = gameState.getAgentPosition(i);
#       if (exact_pos != None):  # {
#         self.tracker[i] = Counter();
#         self.tracker[i][exact_pos] = 1.0;
#       else:
#         temp_tracker = Counter();
#         for key, value in self.tracker[i].iteritems():  # {
#           if (value == 0.0):
#             continue;
#
#           if (my_index == 0 and self.first_update):
#             self.first_update = False;
#             temp_tracker[key] = value;
#             continue;
#
#           if (my_index - 1) % gameState.getNumAgents() == i:  # if this agent moved last turn, update his pos
#             p_move = 0;
#             for direction, _ in Actions._directions.iteritems():
#               pos = Actions.getSuccessor(key, direction);
#               if (not gameState.hasWall(int(pos[0]), int(pos[1]))):
#                 p_move += 1;
#
#             p_move = 1.0 / p_move;
#
#             for direction, _ in Actions._directions.iteritems():  # {
#               pos = Actions.getSuccessor(key, direction);
#               if (not gameState.hasWall(int(pos[0]), int(pos[1]))):
#                 temp_tracker[pos] += p_move * value;
#                 # }
#           else:  # if this agent did not move last turn, pretend he moved using action STOP
#             temp_tracker[key] = value;
#
#         for key, value in temp_tracker.iteritems():  # {
#           true_dist = manhattanDistance(key, gameState.getAgentPosition(my_index));
#           temp_tracker[key] = value * gameState.getDistanceProb(true_dist, measured_dists[i]);
#           if (key in eaten_food):
#             temp_tracker[key] = 1;
#         self.tracker[i] = deepcopy(temp_tracker);
#         self.tracker[i].normalize();
#
#
#   def update_eaten_agent(self, gameState, index_eaten):
#     self.tracker[index_eaten] = Counter();
#     self.tracker[index_eaten][gameState.getInitialAgentPosition(index_eaten)] = 1.0;

class EnemyTracker:  ######### TODO: Keep track of eaten food, and assign it to most likely agent
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
        self.carrying_food = [0] * len(self.enemy_idxs);

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
                            if (not gameState.hasWall(int(pos[0]), int(pos[1]))):
                                p_move[k] += 1;
                            if (pos not in self.tracker[i]):
                                p_move[k] *= 2;
                            if (direction == Directions.STOP):
                                p_move[k] /= 2;

                            k += 1;
                        p_move = p_move / np.sum(p_move);

                        k = 0;
                        for direction, _ in Actions._directions.iteritems():  # {
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

        print(self.carrying_food);
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

    def update_eaten_agent(self, gameState, index_eaten):
        self.tracker[index_eaten] = Counter();
        self.tracker[index_eaten][gameState.getInitialAgentPosition(index_eaten)] = 1.0;
        self.carrying_food[index_eaten] = 0;

tracker = EnemyTracker()
#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first='BehaviourTree', second='BehaviourTree'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class BehaviourTree(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        global global_distancer;
        global_distancer = deepcopy(self.distancer);
        tracker.init(gameState,self.red)
        #grid size
        # self.x,self.y = gameState.getWalls().asList()[-1]
        self.w = gameState.data.layout.width#w = x+1
        self.h = gameState.data.layout.height#h = y+1
        self.nextPos = None
    def chooseAction(self, gameState):
        # mapType = self.evalMap(gameState)
        actions = gameState.getLegalActions(self.index)
        print ("gameState:{}".format(gameState.data._win))
        # features = extract_features(gameState)
        #
        # my_pos = gameState.getAgentPosition(self.index)
        # tracker.update(gameState, self.index)
        # self.displayDistributionsOverPositions(tracker.tracker)

        nonStopActions = list(set(actions)-set(["Stop"]))
        # print("Nonstop: {}".format(nonStopActions))
        nonDeadActions = []
        for action in nonStopActions:
            if not self.isGoingDeadCorner(action,gameState):
                nonDeadActions.append(action)
        if nonDeadActions:
            return random.choice(nonDeadActions)
        print ("Random choice: {}".format(nonDeadActions))
        return random.choice(actions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseStrategy(self):
        pass


    def evalMap(self,gameState):
        """
        evaluate the map, for now they will fall into four catagories: 1) high eaten-cost 2) hard to defense 3) hard to offense 4) easy to get pills
        :param gameState:
        :return:
        """
        layout = gameState.data.layout
        boundary_x = (layout.width+1)//2
        print (layout,type(layout))

        #check if it is easy-pill
        isEasyPills = True
        targetPills = self.getCapsules(gameState)
        for pill in targetPills:
            if abs(pill[0]-boundary_x)*1./layout.width >0.15:
                isEasyPills = False

        #check if it is hard to offense
        isHardToOffense = None
        walls = gameState.getWalls().asList()
        numOfWalls = len(walls)
        area = layout.width*layout.height
        percentageOfWalls = numOfWalls*1./area
        # print percentageOfWalls
        if percentageOfWalls>0.5:
            isHardToOffense = True
        else:
            isHardToOffense = False

        #check if is hard to offense
        isHardToDefense = None
        if percentageOfWalls<0.4:
            isHardToDefense = True
        else:
            return False

        #check if it is high eaten cost
        isHighEatenCost = True

        return isEasyPills,isHardToDefense,isHardToOffense,isHighEatenCost

    def getBack(self,gameState):
        pass

    def getPills(self,gameState):
        pass

    def isGoingDeadCorner(self,action,gameState):
        oppositeActionFinder = {"East":"West","West":"East","South":"North","North":"South"}
        # oppositeAction = oppositeActionFinder[action]

        successor = self.getSuccessor(gameState,action)
        legalActions = successor.getLegalActions(self.index)
        currentAction = action
        numOfActions = len(legalActions)
        if numOfActions<3:
            return True
        if numOfActions>3:
            return False
        while numOfActions ==3:
            if currentAction =="Stop":
                print("Afsafks, action: {}".format(action))
            oppositeAction = oppositeActionFinder[currentAction]
            invalidActions = set(["Stop",oppositeAction])
            validAction = set(legalActions)-invalidActions
            if not validAction:
                return True
            validAction = validAction.pop()
            # print ("validAction: {} currentAction: {}".format(validAction, currentAction))
            currentAction = validAction
            # successor = self.getSuccessor(gameState, validAction)
            successor = successor.generateSuccessor(self.index,validAction)
            legalActions = successor.getLegalActions(self.index)
            numOfActions = len(legalActions)
            # print ("validAction: {} currentAction: {} legalActions: {},".format(validAction, currentAction,legalActions))
            if numOfActions==2:
                return True
            if numOfActions>3:
                return False

        print "Should not show this.-From isGoingDeadCorner"

    def distToNearestFood(self):
        pass

    def distToNearestEnemy(self):
        pass

    def defense(self):
        pass

    def take_food(self):
        pass





