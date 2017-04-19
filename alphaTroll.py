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
# from game import Directions
# import game
# import numpy as np
# from particleFilter import ParticleFilter

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


global bridgePos
bridgePos = None
##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def getEnemyPos(self, gameState):
    return [gameState.getAgentPosition(idx)
            for idx in self.getOpponents(gameState)
            if gameState.getAgentPosition(idx) is not None]

  # def connectedCount(self, gameState, obstaclePos, pos):
  #   def toIntPos(pos):
  #     return (int(pos[0]), int(pos[1]))
  #   pos = toIntPos(pos)
  #   obstaclePos = [toIntPos(p) for p in obstaclePos]
  #   def dfs(grid, pos):
  #     def inRange(x, y):
  #       return 0 <= x and x <= grid.width and 0 <= y and y <= grid.height
  #
  #     x, y = pos
  #     vis[x][y] = True
  #     conn = 1
  #     for dir in dirs:
  #       dx, dy = dir
  #       nx, ny = x + dx, y + dy
  #       nPos = (nx, ny)
  #       if inRange(nx, ny) and not vis[nx][ny] and not grid.isWall(nPos) and not nPos in obstaclePos:
  #         conn += dfs(grid, (nx, ny))
  #     return conn
  #
  #   dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
  #   vis = []
  #   for i in range(0, gameState.data.layout.width):
  #     vis.append([False] * gameState.data.layout.height)
  #   print len(vis)
  #   return dfs(gameState.data.layout, pos)

  def precomputeDeadEnd(self, layout):
    dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    legalMove = [[0 for x in range(layout.height)] for y in range(layout.width)]
    sinks = []

    def inRange(x, y):
      return 0 <= x and x < layout.width and 0 <= y and y < layout.height

    for i in range(layout.width):
      for j in range(layout.height):
        if layout.isWall((i, j)):
          continue
        for dir in dirs:
          dx, dy = dir
          nx, ny = i + dx, j + dy
          nPos = (nx, ny)
          if inRange(nx, ny) and not layout.isWall(nPos):
            legalMove[i][j] += 1

        if legalMove[i][j] == 1:
          sinks.append((i, j))

    deadEndArr = [[False for x in range(layout.height)] for y in range(layout.width)]

    def dfs(pos, prev = None):
      x, y = pos
      if legalMove[x][y] == 3:
        return

      deadEndArr[x][y] = True

      for dir in dirs:
        dx, dy = dir
        nx, ny = x + dx, y + dy
        nPos = (nx, ny)
        if inRange(nx, ny) and nPos != prev and not layout.isWall(nPos):
          dfs(nPos, pos)

    for t in sinks:
      dfs(t)

    return deadEndArr

  def debugPrint2dArr(self, arr):
    # arr is inversed
    for i in range(len(arr)):
      for j in range(len(arr[0])):
        print '*' if arr[i][j] else '.',
      print ''

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
    #
    # print self.connectedCount(gameState, self.getEnemyPos(gameState), gameState.data.layout.getRandomLegalPosition())
    CaptureAgent.registerInitialState(self, gameState)
    # self.deadEndArr = self.precomputeDeadEnd(gameState.data.layout)
    # self.debugPrint2dArr(self.deadEndArr)
    # layout = gameState.data.layout
    # self.cells = []
    # self.cnt = 0
    # self.pf = []
    # self.enemyDistObs = []

    self.initTargetPos = None
    self.distanceToHome = 0

    # self.belief = [util.Counter() for i in range(4)]

    # for i in range(0, layout.width):
    #   for j in range(0, layout.height):
    #     if (not layout.isWall((i, j))):
    #       self.cells += [(i, j)]
    # print self.cells

    # for i in range(len(gameState.blueTeam)):
    #   self.pf.append(ParticleFilter(self.cells, gameState.redTeam[i], None))

  # def getMyPos(self, gameState):
  #   return gameState.getAgentState(self.index).getPosition()

  # def resetPrior(self, opponent):
  #   for cell in self.cells:
  #     self.belief[opponent][cell] = 1.0 / len(self.cells)


  # def updateNB(self, gameState, agentPos, reading, opponent):
  #   belief = self.belief[opponent]
  #   smoothing = 0.000001
  #   sum = smoothing * len(self.cells)
  #   for cell in self.cells:
  #     prior = belief[cell]
  #     belief[cell] = (gameState.getDistanceProb(util.manhattanDistance(cell, agentPos), reading) + smoothing) * prior
  #     sum += belief[cell]
  #   print sum
  #   print sum
  #   for cell in self.cells:
  #     belief[cell] /= sum

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    print "chooseAction for %s" % self.index
    self.computeDanger(gameState)
    observableEnemy = [idx for idx in self.getOpponents(gameState) if
                       gameState.getAgentPosition(idx) is not None and util.manhattanDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(idx)) <= 5]
    players = sorted([self.index] + observableEnemy)
    playerIdx = players.index(self.index)
    bestValue, bestAction = self.alphaBeta(gameState, -float('inf'), float('inf'), playerIdx, players, 6 if len(observableEnemy) > 0 else 4)
    print bestValue, bestAction
    # self.cnt = self.cnt + 1

    # for filter in self.pf:
    #   i = filter.enemy_index
    #   if i in observableEnemy:
    #     enemy_position = gameState.getAgentPosition(i)
    #     filter.S = np.matrix([[enemy_position[0] for j in range(filter.M)],
    #                         [enemy_position[1] for j in range(filter.M)],
    #                         [1.0/filter.M for j in range(filter.M)]
    #                        ])
    #   else:
    #     agent_distances = self.getCurrentObservation().agentDistances[i]
    #     filter.iterate(agent_distances, self.getMyPos(gameState))
    #
    # for filter in self.pf:
    #   belief = self.belief[filter.enemy_index]
    #   sum = 0
    #   cell = [(filter.S[0, m], filter.S[1, m]) for m in range(filter.M)]
    #   for c in cell:
    #     belief[c] += 1.0
    #     sum += 1
    #   for b in belief:
    #     belief[b] /= sum

    # for opponent in self.getOpponents(gameState)[0:2]:
    #   self.resetPrior(opponent)
    #
    #   for observation in self.observationHistory[-4:]:
    #     self.updateNB(gameState, self.getMyPos(gameState), observation.agentDistances[opponent], opponent)
    # self.getPreviousObservation() and self.updateNB(gameState, self.getMyPos(gameState), self.getPreviousObservation().agentDistances[self.getOpponents(gameState)[0]])
    # self.displayDistributionsOverPositions(self.belief)
    return bestAction

  def getMyState(self, gameState):
    return gameState.getAgentState(self.index)

  def isEnemyDangerous(self, gameState, enemyIdx):
    enemyState = gameState.getAgentState(enemyIdx)

    return (not enemyState.isPacman and enemyState.scaredTimer == 0) or (enemyState.isPacman and self.getMyState(gameState).scaredTimer != 0)

  def getMyFood(self, gameState):
    return self.getFood(gameState).asList()

  def computeDanger(self, gameState):
    minAvgDist = float('inf')
    for opponentIdx in self.getOpponents(gameState):
      if (not self.isEnemyDangerous(gameState, opponentIdx)):
        continue
      seq = []
      weights = []
      k = 10
      avg = 0
      ws = 0
      for idx, observation in enumerate(self.observationHistory[-k:]):
        seq.append(observation.agentDistances[opponentIdx])
        weights.append(1.0 / (k - idx + 1))
        avg += seq[-1] * weights[-1]
        ws += weights[-1]
      # print weights
      avg /= ws
      minAvgDist = max(0, min(avg, minAvgDist))
    self.danger = 1.0 / (minAvgDist + 1)
    # print 'dangerous = '
    # print self.danger


  def evaluate(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    return features * weights

  def closeEnough(self, gameState, opponentIdx):
    return util.manhattanDistance(gameState.getAgentState(self.index).getPosition(), gameState.getAgentPosition(opponentIdx)) <= 5

  def timeToGo(self, gameState):
    return gameState.data.timeleft / 8 < self.distanceToHome

  def getBorderPos(self, gameState):
    borderX = gameState.data.layout.width / 2 + (-1 if gameState.isOnRedTeam(self.index) else 0)
    return [(borderX, y) for y in range(0, gameState.data.layout.height)
                                      if not gameState.data.layout.isWall((borderX, y))]

  def drawBridgePos(self, gameState):
    global bridgePos
    if bridgePos == None:
      bridgePos = random.choice(self.getBorderPos(gameState))
      return bridgePos
    else:
      newBridgePos = bridgePos
      for i in range(10):
        newBridgePos = random.choice(self.getBorderPos(gameState))
        if (util.manhattanDistance(newBridgePos, bridgePos) > 3):
          return newBridgePos
      return newBridgePos


  def getFeatures(self, gameState):
    """
    Returns a counter of features for the state
    """
    foodList = self.getMyFood(gameState)
    enemyFoodList = self.getFoodYouAreDefending(gameState).asList()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    myPos = gameState.getAgentState(self.index).getPosition()

    features = util.Counter()
    features['eatenFood'] = -len(foodList)

    features['enemyEatenFood'] = -len(enemyFoodList)

    # get nearest food
    if len(foodList) > 0:
      if gameState.data.timeleft > 1000 and self.initTargetPos != False:
        # breaking deadlock since our robbery logic is good
        if self.initTargetPos is None:
          self.initTargetPos = self.drawBridgePos(gameState)
        minDistance = self.getMazeDistance(myPos, self.initTargetPos)
        if minDistance == 0:
          self.initTargetPos = False
      else:
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # get nearest observable & dangerous enemy
    oppDist = [self.getMazeDistance(gameState.getAgentPosition(idx), myPos)
               for idx in self.getOpponents(gameState)
               if gameState.getAgentPosition(idx) is not None and self.isEnemyDangerous(gameState, idx) and self.closeEnough(gameState, idx)]
    if len(oppDist) > 0:
      features['inverseDistanceToThreat'] = 1.0 / min(oppDist)

    # check if respawned or not
    # todo: make the checking more robust
    isKilled = self.getMazeDistance(myPos, gameState.getInitialAgentPosition(self.index)) < 4
    features['killed'] = 1 if isKilled else 0

    # get nearest friend
    friendDist = [self.getMazeDistance(gameState.getAgentPosition(idx), myPos)
                  for idx in self.getTeam(gameState) if gameState.getAgentPosition(idx)
                  is not None and idx != self.index]
    if len(friendDist) > 0:
      features['inverseDistanceToFriend'] = 1.0 / (min(friendDist) + 0.01)

    # get closest distance to border


    self.distanceToHome = min([self.getMazeDistance(myPos, borderPos) for borderPos in self.getBorderPos(gameState)])

    features['distanceToHome'] = self.distanceToHome

    features['inverseDistanceToHome'] = 1.0 / (self.distanceToHome + 1)
    # get num of invaders
    invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
    features['numInvaders'] = len(invaders)
    # print len(invaders)

    # get is trapped or not
    # isTrapped = self.connectedCount(gameState, self.getEnemyPos(gameState), myPos)
    # print isTrapped

    # get enmey prob
    # enemyProb = 0.5 * sum([self.belief[enemy][myPos] for enemy in self.getOpponents(gameState)])
    # features['enemyProb'] = enemyProb
    # print enemyProb

    features['allEaten'] = 1 if len(foodList) <= 2 else 0

    # features['deadEnd'] = 1 if self.deadEndArr[int(myPos[0])][int(myPos[1])] else 0
    # print features
    return features

  def getWeights(self, gameState):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    def canEatOpp(agentState):
      return not(agentState.isPacman) and agentState.scaredTimer == 0
    myAgentState = self.getMyState(gameState)

    def deadEndWeight():
      return -100 * self.danger

    def distanceToHomeWeight():
      carrying = myAgentState.numCarrying
      if myAgentState.isPacman:
        # wanna return home if carry a lot
        # can delay a little bit if not dangerous
        # if carrying == 0 and myAgentState.isPacman:
        #   return 5
        if len(self.getMyFood(gameState)) <= 2 or self.timeToGo(gameState):
          # must return home
          return -2.1 * carrying
        else:
          return -2.1 * carrying * (self.danger + 0.01)
      else:
        return 0

    def inverseDistanceToHomeWeight():
      carrying = myAgentState.numCarrying
      if myAgentState.isPacman and carrying == 0:
        return -5
      else:
        return 0


    def nearOpp():
        return [idx
                for idx in self.getOpponents(gameState)
                if gameState.getAgentPosition(idx) is not None and self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(idx)) <= 3]
    # print myAgentState.numCarrying
    return {
      'eatenFood': 50.0,# if len(nearOpp()) == 0 or canEatOpp(myAgentState) else 5,
      'enemyEatenFood': -50.0,
      'distanceToFood': -1,
      'inverseDistanceToThreat': -50,
      'killed': -50000,
      'inverseDistanceToFriend': -5,
      'distanceToHome': distanceToHomeWeight(),
      'inverseDistanceToHome': inverseDistanceToHomeWeight(),
      'numInvaders': -100,
      # 'enemyProb': -10,
      'allEaten': 10000,
      # 'deadEnd': 0#deadEndWeight()
    }

  def alphaBeta(self, gameState, alpha, beta, playerIdx, players, depth):
    # playerIdx is internal idx in players list, not global index
    def nextPlayerIdx():
      return (playerIdx + 1) % len(players)

    if depth == 0:
      return self.evaluate(gameState), None

    player = players[playerIdx]
    actions = gameState.getLegalActions(player)
    actions = [] if actions is None else actions
    random.shuffle(actions)

    if gameState.isOnRedTeam(player) == gameState.isOnRedTeam(self.index):  # is our agent
      # is max player
      v = -float('inf')
      bestAction = None
      for action in actions:
        actionValue = self.alphaBeta(gameState.generateSuccessor(player, action), alpha, beta, nextPlayerIdx(), players, depth - 1)[0]
        if actionValue > v:
          v = actionValue
          bestAction = action
        alpha = max(alpha, v)
        if alpha >= beta:
          break
      return (v, bestAction)
    else:
      # is min player
      v = float('inf')
      bestAction = None
      for action in actions:
        actionValue = self.alphaBeta(gameState.generateSuccessor(player, action), alpha, beta, nextPlayerIdx(), players, depth - 1)[0]
        if actionValue < v:
          v = actionValue
          bestAction = action
        beta = min(beta, v)
        if beta <= alpha:
          break
      return (v, bestAction)