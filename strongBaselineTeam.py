# This file is originally uploaded at: https://github.com/cddoria/AI-PACKMAN-team-project-/blob/master/contest/team5.py
# Modified it a bit to make it self-contained for new skeleton
# Play against old baseline by: python capture.py -r strongBaselineTeam.py  -b baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
from game import Directions
import game
import random, time, util, sys
from util import nearestPoint
from util import pause
import numpy as np
from capture import SONAR_NOISE_RANGE, SONAR_NOISE_VALUES, SIGHT_RANGE



SONAR_MAX = (SONAR_NOISE_RANGE - 1) / 2
#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # How big the playing grid is
        self.x, self.y = gameState.getWalls().asList()[-1]
        # List of walls on grid
        self.walls = list(gameState.getWalls())
        # Positions that aren't walls
        self.valid_positions = [position for position in gameState.getWalls().asList(False) if position[1] > 1]
        # Pathways on agent's side of grid
        self.valid_paths = []

        # Set the offset for each agent from the middle of the grid
        if self.red:
            self.our_idxs = gameState.getRedTeamIndices()
            self.enemy_idxs = gameState.getBlueTeamIndices()
            offset = -3
        else:
            self.our_idxs = gameState.getBlueTeamIndices()
            self.enemy_idxs = gameState.getRedTeamIndices()
            offset = 4

        # Check vertical paths...
        for i in range(self.y):
            # If there is no wall at 'i' on the current side...
            if not self.walls[self.x / 2 + offset][i]:
                # If not a wall on current side...
                self.valid_paths.append(((self.x / 2 + offset), i))

        # Set different starting positions for different agents; self.index = index for this agent
        if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
            x, y = self.valid_paths[3 * len(self.valid_paths) / 4]
        else:
            x, y = self.valid_paths[len(self.valid_paths) / 4]

        # Point the agent needs to go to
        self.goto = (x, y)

        self.o_weights = {'successorScore': 100, 'distanceToFood': -1, 'numDefenders': -1000, 'defenderDistance': -10,
                          'distanceToGoal': -1, 'stop': -100, 'reverse': -2}
        self.d_weights = {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'distanceToGoal': -1,
                          'stop': -100, 'reverse': -2}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        nonStopActions = list(set(actions) - set(["Stop"]))
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        maxValue = -np.inf
        bestActions = []
        for a in nonStopActions:
            # print("is safe? {}".format(self.isSafe(a,gameState)))
            if self.isSafe(a,gameState):#never goes into dead corner
                value = self.evaluate(gameState,a)
                if value>maxValue:
                    maxValue =value
                    bestAction = a
                    # bestActions.append(a)
        return bestAction


        # maxValue = max(values)
        # bestActions = [a for a, v in zip(actions, values) if v == maxValue and not self.isGoingDeadCorner(a,gameState)]
        if not bestActions:
            pause()

        # return random.choice(bestActions)

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

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        # print(features,weights,type(features))
        # print(features*weights)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """

        return {'successorScore': 1.0}

    def alphaBeta(self, gameState, depth, alpha, beta):
        # alpha is the best choice for max, beta is the best choice for min
        agentIndex = (gameState.data._agentMoved + 1) % 4
        print("agentIndex: {} self index:{}".format(agentIndex,self.our_idxs))
        actionStates = [(a, gameState.generateSuccessor(agentIndex, a)) for a in gameState.getLegalActions(agentIndex)]
        # actions = gameState.getLegalActions(agentIndex)
        if depth <= 0:
            legalActions = gameState.getLegalActions(agentIndex)
            score = -100000
            for legalAction in legalActions:
                score = max(score,self.evaluate(gameState, legalAction))
            print("Depth 0 , score: {}".format(score))
            return score

        if gameState.isOver() or gameState.data.timeleft == 0:  # game over,terminal state
            finalScore = gameState.getScore()
            if self.red:
                return 10000 * finalScore
            else:
                return -10000 * finalScore
        else:  # game is not over!
            # print("Hi there")
            # actionStates = [(a,gameState.generateSuccessor(agentIndex, a)) for a in gameState.getLegalActions(agentIndex)]
            # score = less_shit_heuristic(actionStates)
            # print("Our indices: {}, gamestateMoved: {}".format(self.our_idxs,agentIndex))
            if agentIndex in self.our_idxs:  # if it is in our team, need to change the condition
                # print ("Here!")
                best = -1000000
                for action, nextState in actionStates:
                    evaluation = self.alphaBeta(action, nextState, depth - 1, alpha, beta)
                    # print ("Own evaluation:{},depth:{} alpha:{} beta:{}".format(evaluation,depth,alpha,beta))
                    # print("MyTeam Best: {} Eval: {}".format(best,evaluation))
                    if evaluation > best:
                        best = evaluation
                    if best > alpha:
                        alpha = best
                    if beta <= alpha:
                        break  # beta prune
            else:
                best = 1000000
                for action, nextState in actionStates:
                    evaluation = self.alphaBeta(action, nextState, depth - 1, alpha, beta)
                    # print("Ene evaluation: {},depth:{} alpha:{} beta:{}".format(evaluation,depth,alpha,beta))
                    # print("Enemy Best: {} Eval: {}".format(best, evaluation))
                    if evaluation < best:
                        best = evaluation
                    if best < beta:
                        beta = best
                    if beta <= alpha:
                        break  # alpha prune
        return best

    # def alphaBeta(self, actionNow, gameState, depth, alpha, beta):
    #     # alpha is the best choice for max, beta is the best choice for min
    #     agentIndex = (gameState.data._agentMoved + 1) % 4
    #     # actionStates = [(a, gameState.generateSuccessor(agentIndex, a)) for a in gameState.getLegalActions(agentIndex)]
    #     actions = gameState.getLegalActions(agentIndex)
    #     if depth <= 0:
    #         score = self.evaluate(gameState, actionNow)
    #         # print("Depth 0 , score: {}".format(score))
    #         return score
    #
    #     if gameState.isOver() or gameState.data.timeleft == 0:  # game over,terminal state
    #         finalScore = gameState.getScore()
    #         if self.red:
    #             return 10000 * finalScore
    #         else:
    #             return -10000 * finalScore
    #     else:  # game is not over!
    #         # print("Hi there")
    #         # actionStates = [(a,gameState.generateSuccessor(agentIndex, a)) for a in gameState.getLegalActions(agentIndex)]
    #         # score = less_shit_heuristic(actionStates)
    #         # print("Our indices: {}, gamestateMoved: {}".format(self.our_idxs,agentIndex))
    #         if agentIndex in self.our_idxs:  # if it is in our team, need to change the condition
    #             # print ("Here!")
    #             best = -1000000
    #             for action in actions:
    #                 evaluation = self.alphaBeta(action, gameState, depth - 1, alpha, beta)
    #                 # print ("Own evaluation:{},depth:{} alpha:{} beta:{}".format(evaluation,depth,alpha,beta))
    #                 # print("MyTeam Best: {} Eval: {}".format(best,evaluation))
    #                 if evaluation > best:
    #                     best = evaluation
    #                 if best > alpha:
    #                     alpha = best
    #                 if beta <= alpha:
    #                     break  # beta prune
    #         else:
    #             best = 1000000
    #             for action, nextState in actionStates:
    #                 evaluation = self.alphaBeta(action, nextState, depth - 1, alpha, beta)
    #                 # print("Ene evaluation: {},depth:{} alpha:{} beta:{}".format(evaluation,depth,alpha,beta))
    #                 # print("Enemy Best: {} Eval: {}".format(best, evaluation))
    #                 if evaluation < best:
    #                     best = evaluation
    #                 if best < beta:
    #                     beta = best
    #                 if beta <= alpha:
    #                     break  # alpha prune
    #     return best
    def isSafe(self,action,gameState):
        return True

    def isGoingDeadCorner(self,action,gameState):
        return False

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
        If an enemy is a ghost, the offensive agent labels it as a defender and determines its position
        relative to its own. It also considers the position of the closest food relative to its own. It
        checks the food list and if the distance from offensive agent to the closest food is less than
        the distance to half the distance of the closest defender, then it goes for the food.
    """
    def isSafe(self,action,gameState):
        isDeadCorner,depth = self.isGoingDeadCorner(action,gameState)
        if not isDeadCorner:#it's not dead corner
            return True
        else:
            # my_pos = gameState.getAgentPosition(self.index)
            dists = gameState.getAgentDistances()
            nearestDistToEnemy =100
            for enemy_ind in self.enemy_idxs:
                enemy_pos = gameState.getAgentPosition(enemy_ind)
                if not enemy_pos:#not in sight range
                    dist = max(SIGHT_RANGE+1,dists[enemy_ind]-3)
                else:
                    dist = dists[enemy_ind]
                nearestDistToEnemy = min(dist,nearestDistToEnemy)
            if depth*2-2<nearestDistToEnemy:
                return True
            else:
                return False


    def isGoingDeadCorner(self,action,gameState):
        oppositeActionFinder = {"East":"West","West":"East","South":"North","North":"South"}
        # oppositeAction = oppositeActionFinder[action]
        successor = self.getSuccessor(gameState,action)
        legalActions = successor.getLegalActions(self.index)
        currentAction = action
        numOfActions = len(legalActions)
        if numOfActions<3:
            return True,1
        if numOfActions>3:
            return False,0
        depth = 1
        while numOfActions ==3:
            depth +=1
            if currentAction =="Stop":
                print("Afsafks, action: {}".format(action))
            oppositeAction = oppositeActionFinder[currentAction]
            invalidActions = set(["Stop",oppositeAction])
            validAction = set(legalActions)-invalidActions
            if not validAction:
                return True,depth
            validAction = validAction.pop()
            # print ("validAction: {} currentAction: {}".format(validAction, currentAction))
            currentAction = validAction
            # successor = self.getSuccessor(gameState, validAction)
            successor = successor.generateSuccessor(self.index,validAction)
            legalActions = successor.getLegalActions(self.index)
            numOfActions = len(legalActions)
            # print ("validAction: {} currentAction: {} legalActions: {},".format(validAction, currentAction,legalActions))
            if numOfActions==2:
                return True,depth+1
            if numOfActions>3:
                return False,0

        print "Should not show this.-From isGoingDeadCorner"

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        current_state = successor.getAgentState(self.index)
        current_position = current_state.getPosition()
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        better = 999
        enemies = []
        defenders = []
        distances_to_defenders = []
        distances_to_food = []

        """
            Play Offensively
        """
        # Check the indices of the opponents...
        for agent in self.getOpponents(successor):
            # Add opponents to list of enemies
            enemies.append(successor.getAgentState(agent))

        # Check enemies...
        for enemy in enemies:
            # If there is an enemy position that we can see...
            if not enemy.isPacman and enemy.getPosition() != None:
                # Add that enemy to the list of defenders
                defenders.append(enemy)
                features['numDefenders'] = len(defenders)

        # If there is a defender...
        if len(defenders) > 0:
            # Check the indices of defenders...
            for d in defenders:
                # Find the shortest distance to the defender from current position and add to list of defender distances
                distances_to_defenders.append(self.getMazeDistance(current_position, d.getPosition()))
                features['defenderDistance'] = min(distances_to_defenders)

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            for food in foodList:
                distances_to_food.append(self.getMazeDistance(current_position, food))
                features['distanceToFood'] = min(distances_to_food)

        # Check food and determine the location to intercept
        for food in foodList:
            if distances_to_food < distances_to_defenders:
                # Set the distance equal to the distance from the current position to the food
                distances_to_food = self.getMazeDistance(current_position, food)

                # If a distance is less than the value of the more important area...
                if distances_to_food < better:
                    # Set the value of the more important area equal to that distance
                    better = distances_to_food

                # Set the point of interception to that food
                intercept = food

            if distances_to_food < 9 and distances_to_food <= better and intercept != 0:
                # Go to that point to intercept
                self.goto = intercept

        features['distanceToGoal'] = self.getMazeDistance(current_position, self.goto)

        # If the agent is at the goto point...
        if self.getMazeDistance(current_position, self.goto) == 0:
            self.food_count = len(self.getFood(gameState).asList())+1
            # self.food_count = self.food_count + 1

            if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
                self.goto = self.valid_paths[5 * len(self.valid_paths) / 6]
            else:
                self.goto = self.valid_paths[1 * len(self.valid_paths) / 6]

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return self.o_weights


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
        Our defensive agent patrols the middle border for any PacMan agents who dare to cross.
        If a PacMan agent is daring enough, the defensive agent labels it as an invader and
        determines its position relative to its own. It also considers any available paths and
        compares the distance between those paths and the enemy's position with the distance between
        its own position and the enemy's. Once a path has been chosen, our defensive agent goes in
        for the kill!
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        current_state = successor.getAgentState(self.index)
        current_position = current_state.getPosition()
        better = 999

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if current_state.isPacman: features['onDefense'] = 0

        enemies = []

        # Check the indices of the opponents...
        for agent in self.getOpponents(successor):
            # Add opponents to list of enemies
            enemies.append(successor.getAgentState(agent))

        invaders = []

        # Check enemies...
        for enemy in enemies:
            # If there is an enemy position that we can see...
            if enemy.isPacman and enemy.getPosition() != None:
                # Add that enemy to the list of invaders
                invaders.append(enemy)
                features['numInvaders'] = len(invaders)

        distances = []

        # If there is an invader...
        if len(invaders) > 0:
            # Check the indices of invaders...
            for e in invaders:
                # Find the shortest distance to the invader from current position and add to list of distances
                distances.append(self.getMazeDistance(current_position, e.getPosition()))
                features['invaderDistance'] = min(distances)
                # Intercept tracker
                intercept = 0;

        for e in invaders:
            # Check paths and determine the location to intercept
            for path in self.valid_paths:
                """
                    If the distance between the path and enemy location is less than the distance from the defensive
                    agent's current position to the invader's position...
                """
                if self.getMazeDistance(path, e.getPosition()) < distances:
                    # Set the distance equal to the distance from the path to the enemy
                    distances = self.getMazeDistance(path, e.getPosition())

                    # If a distance is less than the value of the more important area...
                    if distances < better:
                        # Set the value of the more important area equal to that distance
                        better = distances

                    # Set the point of interception to that path
                    intercept = path
            """
                If distance is less than 9 and greater than or equal to the value of the more important area and
                the agent has a path to intercept...
            """
            if distances < 9 and distances <= better and intercept != 0:
                # Go to that point to intercept
                self.goto = intercept

        # Check the indices of the opponents...
        for e in invaders:
            # Coordinates of invader
            x, y = e.getPosition()

            # If on the red team and the enemy is on the agent's left...
            if self.red and x < self.x / 2:
                # Get him/her
                self.goto = e.getPosition()
            # Else if on the blue team and the enemy is on the agent's right...
            elif not self.red and x > self.x / 2:
                # Get him/her
                self.goto = e.getPosition()

        features['distanceToGoal'] = self.getMazeDistance(current_position, self.goto)

        # If the agent is at the goto point...
        if self.getMazeDistance(current_position, self.goto) == 0:
            # The defensive agent (on either team) will continue patrolling that area
            if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
                self.goto = self.valid_paths[5 * len(self.valid_paths) / 6]
            else:
                self.goto = self.valid_paths[1 * len(self.valid_paths) / 6]


        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return self.d_weights
