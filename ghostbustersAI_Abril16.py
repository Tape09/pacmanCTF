# baselineTeam.py
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
import random, time, util, sys
from game import Directions
import game
from game import Actions
from util import nearestPoint

import math
import numpy as np
from numpy import unravel_index



COLLISION_TOLERANCE = 0.7

# A shared memory class, containing a counter and a increment function.
# This might get weird if you play the same team vs itself. If you want to do that just copy this file and play myteam vs myteamcopy.
class SharedMemory:
    # this is the constructor for the class. It gets called wehn you create an instance of the class. Inits counter to 0.
    def __init__(self):
        self.counter = 0
        self.opponentsFilters = {}
        self.teammatesBehavior = {}
        self.teammatesMessage = {}
        self.allBehaviors = ['eatPacman', 'eatPills', 'runFromPacman', 'guardUnsafeArea', 'enterSafeArea', 'returnHome']
        self.defensiveBehaviors = ['eatPacman', 'guardUnsafeArea']
        self.offensiveBehaviors = ['eatPills', 'runFromPacman', 'enterSafeArea', 'returnHome']
        self.messages = ["help", "noMessage"]
        
        self.opponentsVisible = {}
        
        self.safePillsList = []
        self.safePillsMap = []
        self.numSafePills = 0

    # increment class function. it increments its own counter by one.
    def increment(self):
        self.counter += 1

# create instance of the class. The "whatever" variable is in the global scope, so it can be accessed from your agents chooseAction function.
sharedMemory = SharedMemory()




class HistogramFilter:
    def __init__(self, gridWidth, gridHeight, motionModel, x0=None):
        self.motionModel = motionModel
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

        if(x0 is None):
            self.belief = np.random.rand(gridHeight, gridWidth)
            self.belief /= np.sum(np.sum(self.belief, axis=0))
        else:
            self.belief = np.zeros( (gridHeight, gridWidth) )
            self.belief[(x0[1], x0[0])] = 1
        self.belief_hat = self.belief.copy()


    def getEstimatedPosition(self):
        position = unravel_index(self.belief.argmax(), self.belief.shape)
        if(len(list(position)) > 2):
            for p in position:
                if abs(p[0]) >= self.gridWidth or abs(p[1]) >= self.gridHeight:
                    position.remove(p)
            position = random.choice(position)
        return (position[1], position[0])


    def predict(self, gameState):
        prev_belief = self.belief.copy()
        for i in range(1, self.gridHeight-1):
            for j in range(1, self.gridWidth-1):
                location = tuple( (i, j) )
                if(prev_belief[location] != 0):
                    self.belief_hat[i-1,j] = prev_belief[location] * 0.25 * (not gameState.hasWall(j, i-1))
                    self.belief_hat[i+1,j] = prev_belief[location] * 0.25 * (not gameState.hasWall(j, i+1))
                    self.belief_hat[i,j-1] = prev_belief[location] * 0.25 * (not gameState.hasWall(j-1, i))
                    self.belief_hat[i,j+1] = prev_belief[location] * 0.25 * (not gameState.hasWall(j+1, i))
                    self.belief_hat[i,j] = prev_belief[location] * 0.5 * (not gameState.hasWall(j, i))
        nu = np.sum(np.sum(self.belief_hat, axis=0))
        if(nu != 0):
            self.belief_hat /= nu



    def update(self, gameState, z, observerPos, x=None):
        if(x is None):
            for i in range(self.gridHeight):
                for j in range(self.gridWidth):
                    location = (i, j)
                    trueDistance = util.manhattanDistance((j,i), observerPos)
                    probzGivenx = gameState.getDistanceProb(trueDistance, z)
                    self.belief[location] = probzGivenx * self.belief_hat[location]
        else:
            self.belief *= 0
            self.belief[(int(x[1]), int(x[0]))] = 1

        nu = np.sum(np.sum(self.belief, axis=0))
        if(nu != 0):
            self.belief /= nu



    def debugDistribution(self, color, agentObject):
        for i in range(self.gridHeight):
            for j in range(self.gridWidth):
                location = tuple( (i, j) )
                prob = self.belief[location]
                if(prob!=0):
                    color = tuple(np.array(color) * (1-prob))
                    CaptureAgent.debugDraw(agentObject, (j,i), color, False)



    def debugPosition(self, color, agentObject):
        position = self.getEstimatedPosition()
#        CaptureAgent.debugDraw(agentObject, position, (1.0, 1.0, 1.0), False)
        CaptureAgent.debugDraw(agentObject, position, color, False)






def reverseAction(action):
    vertical = ['South', 'North']
    horizontal = ['West', 'East']
    reverse = None
    if action in vertical:
        vertical.remove(action)
        reverse = vertical[0]
    elif action in horizontal:
        horizontal.remove(action)
        reverse = horizontal[0]
    return reverse







def UCB(self, node, child):
    C = 1.4
    return child.wins + C * math.sqrt( math.log(node.plays) / child.plays )





def UCB_sample(self, node):
    #print("-------- UCB sampling --------")
    weights = np.zeros(len(node.children))
    i = 0
    for child in node.children:
        w = UCB(self, node, child)
        weights[i] = w
        i += 1
        '''print("Child", child)
        print("Child parent", child.parent)
        print("Child plays", child.plays)
        print("Child wins", child.wins)
        print("Child action", child.action)
        print("Child children", len(child.children))'''

    sum_weights = np.sum(weights)
    if(sum_weights != 0):
        weights /= sum_weights
        i = 0
        for child in node.children:
            child.UCB = weights[i]
            i += 1
    idx_max = np.argmax(weights)
    '''
    print("weights", weights)
    print("idx_max", idx_max)
    '''

    return node.children[idx_max]






def expansion(self, node):
#    print("-------- Expansion --------")
    actionsTaken = [child.action for child in node.children]
    legalActions = self.getLegalActions(node.state)
    actionsLeft = [action for action in legalActions if action not in actionsTaken]

    if(len(actionsLeft)==0):
        return node, random.choice(node.children)
    randAction = random.choice(actionsLeft)
    '''
    print("Actions taken until now", actionsTaken)
    print("Legal actions", legalActions)
    print("Actions left to take", actionsLeft)
    print("Random action", randAction)
    '''

    nextState = node.state.generateSuccessor(self.index, randAction)
    child = Node(nextState)
    child.parent = node
    child.action = randAction
    child.children = []
    node.children.append(child)

    return node, child



def selection(self, root, maxLength):
#    print("-------- Selection --------")
    node = root
    legalActions = self.getLegalActions(node.state)
    expanded = False

    treeDepth = 0

    if(len(node.children) < len(legalActions)):
        _, node = expansion(self, node)
    else:
        while(len(node.children) > 0):
            legalActions = self.getLegalActions(node.state)
            if(len(node.children) == len(legalActions)):
                node = UCB_sample(self, node)
            else:
                _, node = expansion(self, node)
                expanded = True

            treeDepth+=1
#            print("Tree depth", treeDepth)
        if (expanded == False):
            _, node = expansion(self, node)

    node.plays += 1

    return node



def simulation(self, child, maxLength):
#    print("-------- Simulation --------")
    state = child.state
    points = 0
    itr = 1

    while (itr <= maxLength):
        legalActions = self.getLegalActions(state)
        action = random.choice(legalActions)
        points = points + self.getPoints(child.state, state, self.root.state)
        state = state.generateSuccessor(self.index, action)
        itr += 1
    if(points > 0):
        child.wins += 1

    return child





def backpropagation(self, child):
#    print("-------- Backpropagation --------")
    node = child
    while(node.parent is not None):
        parent = node.parent
        parent.plays += 1
        parent.wins += node.wins
        node = parent

    return node



def MCTS_sample(self, root, maxLength):
    node = selection(self, root, maxLength)
    node = simulation(self, node, maxLength)
    root = backpropagation(self, node)
    return root



class Node():
    def __init__(self, state=None, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.wins = 0
        self.plays = 0
        self.UCB = 0




    def __str__(self):
        if(self.parent is None):
            return ("ROOT" + "\nState: " + str(self.state) + "Plays: " + str(self.plays) + "\nWins: " + str(self.wins) + "\n")
        else:
            return ("State:" + str(self.state) + "\nAction: " + str(self.action) + "\nPlays: " + str(self.plays) + "\nWins: " + str(self.wins))





#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
#                first = 'HungryAgent', second = 'DoNothingAgent'):
                first = 'HungryAgent', second = 'SuperDefensiveAgent'):
#                first = 'SuperDefensiveAgent', second = 'SuperDefensiveAgent'):
#                first = 'DoNothingAgent', second = 'SuperDefensiveAgent'):
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

    self.opponentsIndices = self.getOpponents(gameState)
    self.teammatesIndices = self.getTeam(gameState)

    # get distance in maze distance
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    self.distancer.getMazeDistances()

    # MCTS parameters
    self.maxLength = 4 #10  # max search length
    self.maxTime = 20 #40     # max time (1s) TODO: implement timer and return best option when time is up
    self.maxActions = 1200
    self.root = Node(gameState)
    self.root.parent = None
    self.root.children = []
    self.root.plays = 1

    # Debug colors
    self.HelpRed = (1, 0, 0)
    self.Red = (1, 0.5, 0.5)
    self.Green = (0.5, 1, 0.5)
    self.Blue = (0.5, 0.5, 1)
    self.Purple = (0.6, 0.196078, 0.8)
    self.Turquoise = (0, 0.74902, 1)
    self.White = (1.0, 1.0, 1.0)
    self.opponentsColor = {self.opponentsIndices[0]: self.Purple, self.opponentsIndices[1]: self.Turquoise}

    # Tactics parameters
    self.returnFood = 5         # how much food the agent has before it returns back
    self.allowedStop = False
    self.allowedBack = False    # if the agent is allowed to turn backwards or not
    self.enemiesTotal = float(len([gameState.getAgentState(i) for i in self.getOpponents(gameState)]))
    self.foodTotal = float(len(self.getFood(gameState).asList()))
#    self.chasedLength = 4       # if the agent is this close to a ghost, it will flee from it
    self.opponentsDistThresh = 3

    self.behavior = 'Defend'

    self.scoreSign = 1
    if gameState.isOnRedTeam(self.index):
        self.scoreSign = -1

    walls = gameState.getWalls()
    self.gridWidth = walls.width
    self.gridHeight = walls.height


    if(len(sharedMemory.opponentsFilters) == 0):
        hfMotionModel = np.array([[0,0.25,0], [0.25,0.5,0.25], [0,0.25,0]])
        for opponent in self.opponentsIndices:
            hf = HistogramFilter(self.gridWidth, self.gridHeight, hfMotionModel, gameState.getInitialAgentPosition(opponent))
            hf.getEstimatedPosition()
            sharedMemory.opponentsFilters[opponent] = hf


    if(len(sharedMemory.opponentsVisible) == 0):
        for teammate in self.teammatesIndices:
            sharedMemory.opponentsVisible[teammate] = []
    self.otherTeammateIndex = [i for i in self.teammatesIndices if i != self.index][0]
    self.opponentAssigned = None

    posx = self.gridWidth/2
    self.midPositions = []
    for posy in range(0,self.gridHeight):
        self.midPositions.append((posx,posy))

    self.safeEnterPos = None
    self.safeness = []
    self.unsafeGuardPos = None
    self.unsafeness = []

    '''
    Agents Behaviors:
        notSet
        eatPacman
        eatPills
        runFromPacman
        guardUnsafeArea
        enterSafeArea

    '''

    if(len(sharedMemory.teammatesBehavior) == 0):
        sharedMemory.teammatesBehavior = {self.teammatesIndices[0]: "notSet", self.teammatesIndices[1]: "notSet"}
        sharedMemory.teammatesMessage = {self.teammatesIndices[0]: "noMessage", self.teammatesIndices[1]: "noMessage"}

    sharedMemory.safePillsList, sharedMemory.safePillsMap, sharedMemory.numSafePills = self.getSafePills(gameState)





  def runHistogramFilters(self, gameState):
    for opponent in self.opponentsIndices:
        hf = sharedMemory.opponentsFilters[opponent]
        hf.predict(gameState)

        observerPos = gameState.getAgentState(self.index).getPosition()
        opponentState = gameState.getAgentState(opponent)
        opponentPos = opponentState.getPosition()
        z = gameState.getAgentDistances()[opponent]

        hf.update(gameState, z, observerPos, opponentPos)



  def debugHistogramFilters(self):
    for opponent in self.opponentsIndices:
        hf = sharedMemory.opponentsFilters[opponent]
#        hf.debugDistribution(self.opponentsColor[opponent], self)
        hf.debugPosition(self.opponentsColor[opponent], self)
#        time.sleep(0.1)



  def agentType(self):
      return self.__class__.__name__



  def getPoints(self, state, prevState, currentState):
      return 1



  def frange(self, x, y, jump):
    while x <= y:
        yield x
        x += jump


  def getSafePills(self, state):
      pillsMap = self.getFood(state)
      safePillsMap = np.zeros( (self.gridHeight, self.gridWidth) )
      safePillsList = []
      opponent1Pos = sharedMemory.opponentsFilters[self.opponentsIndices[0]].getEstimatedPosition()
      opponent2Pos = sharedMemory.opponentsFilters[self.opponentsIndices[1]].getEstimatedPosition()
      numSafePills = 0
      for x in range(self.gridWidth):
          for y in range(self.gridHeight):
              p = (x, y)
              if pillsMap[x][y] == True:
                  if (util.manhattanDistance(p, opponent1Pos) <= self.opponentsDistThresh) or (util.manhattanDistance(p, opponent2Pos) <= self.opponentsDistThresh):
                      safePillsMap[y][x] = 0
                  else:
                      safePillsMap[y][x] = 1
                      numSafePills += 1
                      safePillsList.append(p)
                  
      return safePillsList, safePillsMap, numSafePills
      


  def getOpponentsVisible(self, state):
      opponentsVisible = []
      currentPos = state.getAgentState(self.index).getPosition()

      for i in self.opponentsIndices:
          visible = True
          opponentPos = sharedMemory.opponentsFilters[i].getEstimatedPosition()
          dy = float(currentPos[1]-opponentPos[1])
          dx = float(currentPos[0]-opponentPos[0])
          if(dx != 0):
              m = dy/dx
              if(dx > 0):
                  startX = opponentPos[0]
                  startY = opponentPos[1]
                  endX = currentPos[0]
                  endY = currentPos[1]                  
              else:
                  startX = currentPos[0]
                  startY = currentPos[1]
                  endX = opponentPos[0]
                  endY = opponentPos[1]

              b = startY - m * startX
              for x in self.frange(startX, endX, 0.2):
                  newPos = (int(x), int(m*x+b))
                  if(state.hasWall(newPos[0], newPos[1])):
                      visible = False
#                      CaptureAgent.debugDraw(self, newPos, self.Red, False)
#                  else:
#                      CaptureAgent.debugDraw(self, newPos, self.Blue, False)   
          else:
              x = currentPos[0]
              if(dy > 0):
                  startY = opponentPos[1]
                  endY = currentPos[1]
              else:
                  startY = currentPos[1]
                  endY = opponentPos[1]
              for y in self.frange(startY, endY, 0.2):
                  newPos = (int(x), int(y))
                  if(state.hasWall(newPos[0], newPos[1])):
                      visible = False
          if visible:
              opponentsVisible.append(i)
#              CaptureAgent.debugDraw(self, opponentPos, self.Green, False)
          
      print(opponentsVisible)
      return opponentsVisible



  def estimateUnsafeArea(self, state):
    prevUnsafeness = self.unsafeness
    self.unsafeness = []

    enemiesIndices = [i for i in self.opponentsIndices]
    enemiesPos = [sharedMemory.opponentsFilters[g].getEstimatedPosition() for g in enemiesIndices]
    i = 0
    for p in self.midPositions:
        unsafeVal = 0
        if state.hasWall(p[0],p[1]):
            unsafeVal = 99999
        else:
            dist =  [abs(self.getMazeDistance(p, e)) for e in enemiesPos]
            unsafeVal = min(dist)

        if len(prevUnsafeness) > 0:
            unsafeVal = (unsafeVal + prevUnsafeness[i])/2
        self.unsafeness.append(unsafeVal)
        i+=1

    i = np.argmin(self.unsafeness)
    minVal = self.unsafeness[i]
    allUnsafePos = [self.midPositions[p] for p in range(len(self.midPositions)) if self.unsafeness[p] == minVal]
    return self.midPositions[i], allUnsafePos



  def estimateSafeArea(self, state):
    prevSafeness = self.safeness
    self.safeness = []
    enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
    enemiesPos =  [e.getPosition() for e in enemies if e.getPosition() is not None]
    i = 0
    for p in self.midPositions:
        safeVal = 0
        if state.hasWall(p[0],p[1]):
            safeVal = 0
        else:
            if len(enemiesPos)>0:
                dist =  [abs(self.getMazeDistance(p, e)) for e in enemiesPos]
                safeVal = min(dist)
            else:
                safeVal = 10 #??

        if len(prevSafeness) > 0:
            safeVal = (safeVal + prevSafeness[i])/2
        self.safeness.append(safeVal)
        i+=1

    allSafePos = [self.midPositions[p] for p in range(len(self.midPositions)) if self.safeness[p] > 5]

    if len(allSafePos)==0:
        i = np.argmax(self.safeness)
        return self.midPositions[i], [self.midPositions[i]]

#    foodList = self.getFood(state).asList()
    foodList, foodMap, numSafePills = self.getSafePills(state)
    if(numSafePills > 0):
        bestVal = min( min([abs(self.getMazeDistance(p, f)) for f in foodList]) for p in allSafePos)
        bestPos = random.choice([p for p in allSafePos if min([abs(self.getMazeDistance(p, f)) for f in foodList]) == bestVal])
        return bestPos, allSafePos
    else:
        i = np.argmax(self.safeness)
        return self.midPositions[i], [self.midPositions[i]]






  def scaredGhost(self, state):
    agentState = state.getAgentState(self.index)
    if agentState.scaredTimer > 0 and not agentState.isPacman:
        return True
    return False


  def scaredAgent(self, state):
    agentState = state.getAgentState(self.index)
    if agentState.scaredTimer > 0:
        return True
    return False


  def runFromPacman(self, prevState, futureState):
      sharedMemory.teammatesBehavior[self.index] = 'runFromPacman'
      enemies = [prevState.getAgentState(i) for i in self.getOpponents(prevState)]
      futurePos = futureState.getAgentState(self.index).getPosition()
      prevPos = prevState.getAgentState(self.index).getPosition()
      futurePacmanDistances = [abs(self.getMazeDistance(futurePos, a.getPosition())) for a in enemies if a.isPacman and a.getPosition() != None]
      if len(futurePacmanDistances) == 0:
          return 1
      futurePacmanDistance = min(futurePacmanDistances)
      prevPacmanDistance = min([abs(self.getMazeDistance(prevPos, a.getPosition())) for a in enemies if a.isPacman and a.getPosition() != None])
      if(futurePacmanDistance > prevPacmanDistance):
          return 1
      return -1




  def agentDied(self, state):
      return (state.getAgentState(self.index).getPosition() == self.start) and (self.getPreviousObservation() is not None)




  def eatPacman(self, futureState, currentState, rootState):
    points = 0
    sharedMemory.teammatesBehavior[self.index] = 'eatPacman'
    rootPos = rootState.getAgentState(self.index).getPosition()
    futurePos = futureState.getAgentState(self.index).getPosition()

    if rootState.getAgentState(self.index).isPacman:
        points -= 5
    else:
        invadersIndices = [i for i in self.opponentsIndices if rootState.getAgentState(i).isPacman]
        if(len(invadersIndices) > 0):
            rootDist = min([abs(self.getMazeDistance(rootPos, sharedMemory.opponentsFilters[i].getEstimatedPosition())) for i in invadersIndices])
            futureDistances = [abs(self.getMazeDistance(futurePos, sharedMemory.opponentsFilters[i].getEstimatedPosition())) for i in invadersIndices]
            if(len(futureDistances) > 0):
                futureDist = min(futureDistances)
                if(futureDist < rootDist):
                    points += 1
                    if self.agentDied(futureState):
                        points += 5
                else:
                    points -= 5

    return points



  def isChased(self, state):
        ''' check if being chased by a ghost '''
        pos = state.getAgentState(self.index).getPosition()
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 1]
        if len(ghosts)>0:
            dist = min([abs(self.getMazeDistance(pos, g.getPosition())) for g in ghosts])
            if dist < self.opponentsDistThresh:
                return True, dist
        else:
            enemiesIndices = [i for i in self.opponentsIndices]
            dist = min([abs(self.getMazeDistance(pos, sharedMemory.opponentsFilters[e].getEstimatedPosition())) for e in enemiesIndices])

        return False, dist



  def eatPills(self, futureState, rootState, currentState, futureFoodList, currentFoodList):
    ''' Called when the agent is pacman or is a ghost aiming to be a pacman '''
    sharedMemory.teammatesBehavior[self.index] = 'eatPills'
    futurePos = futureState.getAgentState(self.index).getPosition()
    rootPos = rootState.getAgentState(self.index).getPosition()
    currentPos = currentState.getAgentState(self.index).getPosition()
    
    if len(futureFoodList) <= 2:
        return 10
    points = 0
    currentDist = min([abs(self.getMazeDistance(currentPos, f)) for f in futureFoodList])
    rootDist = min([abs(self.getMazeDistance(rootPos, f)) for f in futureFoodList])
    futureDist = min([abs(self.getMazeDistance(futurePos, f)) for f in futureFoodList])

    if futureState.getAgentState(self.index).numCarrying > currentState.getAgentState(self.index).numCarrying:
        points += 3
    elif futureDist < currentDist:
        points += 2
    elif futureDist < rootDist:
        points += 1
    else:
        points -= 20
    return points






  def returnHome(self, futureState, currentState, rootState):
    points = 0
    sharedMemory.teammatesBehavior[self.index] = 'returnHome'
    futurePos = futureState.getAgentState(self.index).getPosition()
    futureDist = abs(self.getMazeDistance(futurePos, self.start))
    rootPos = rootState.getAgentState(self.index).getPosition()
    rootDist = abs(self.getMazeDistance(rootPos, self.start))
    currentPos = currentState.getAgentState(self.index).getPosition()
    currentDist = abs(self.getMazeDistance(currentPos, self.start))


    agentWillDie = self.agentDied(futureState)

    if futureDist < currentDist and agentWillDie == False:
#        CaptureAgent.debugDraw(self, currentPos, self.Blue, False)
        points += 2
    elif futureDist < rootDist and agentWillDie == False:
        points += 1
    else:
#        CaptureAgent.debugDraw(self, futurePos, self.Red, False)
        points -= 5


    return points




  def enterSafeArea(self, futureState, currentState, rootState):
    points = 0
    rootPos = rootState.getAgentState(self.index).getPosition()
    currentPos = currentState.getAgentState(self.index).getPosition()

    sharedMemory.teammatesBehavior[self.index] = 'enterSafeArea'

    rootDist = abs(util.manhattanDistance(rootPos, self.safeEnterPos))
    futureDist = abs(util.manhattanDistance(futureState.getAgentState(self.index).getPosition(), self.safeEnterPos))
    currentDist = abs(util.manhattanDistance(currentState.getAgentState(self.index).getPosition(), self.safeEnterPos))
    
    agentWillDie = self.agentDied(futureState)
    
    if futureDist < COLLISION_TOLERANCE and agentWillDie == False:
        CaptureAgent.debugDraw(self, currentPos, self.White, False)
        points += 3
    elif futureDist < currentDist and agentWillDie == False:
        points += 2
    elif futureDist < rootDist and agentWillDie == False:
        points += 1
    else:
        points -= 20

    return points




  def guardUnsafeArea(self, futureState, currentState, rootState):
    points = 0
    rootPos = rootState.getAgentState(self.index).getPosition()

#    CaptureAgent.debugDraw(self, currentPos, self.White, False)
    sharedMemory.teammatesBehavior[self.index] = 'guardUnsafeArea'

    rootDist = abs(self.getMazeDistance(rootPos, self.unsafeGuardPos))
    futureDist = abs(self.getMazeDistance(futureState.getAgentState(self.index).getPosition(), self.unsafeGuardPos))
    currentDist = abs(self.getMazeDistance(currentState.getAgentState(self.index).getPosition(), self.unsafeGuardPos))

    agentWillDie = self.agentDied(futureState)

    points = 0
    if rootState.getAgentState(self.index).isPacman:
        points -= 5
    else:
        if futureDist < COLLISION_TOLERANCE and agentWillDie == False:
            points += 3
        elif futureDist < currentDist and agentWillDie == False:
            points += 2
        elif futureDist < rootDist and agentWillDie == False:
            points += 1
        else:
            points -= 20

    return points




  def generalGhostBehavior(self, points, futureState, currentState, rootState):
    rootPos = rootState.getAgentState(self.index).getPosition()
    invadersIndices = [i for i in self.opponentsIndices if rootState.getAgentState(i).isPacman]

    if(rootState.getAgentState(self.index).isPacman):
        if(len(invadersIndices)> 0) or sharedMemory.teammatesMessage[self.otherTeammateIndex] == "help":
            ''' Go back home behavior '''
            points +=  self.returnHome(futureState, currentState, rootState)
            return points
    else:
        points +=  self.eatPacman(futureState, currentState, rootState)
        if(len(invadersIndices) > 0):
            ''' Defending behavior when there is one or more invaders in our field '''
            points +=  self.eatPacman(futureState, currentState, rootState)
        else:
            ''' Defending behavior when there are no invaders around '''
            rootPos = rootState.getAgentState(self.index).getPosition()
            #if (self.unsafeGuardPos is not None) and (not rootState.getAgentState(self.index).isPacman) and (rootPos is not self.unsafeGuardPos):
            if (self.unsafeGuardPos is not None) and (rootPos is not self.unsafeGuardPos):
                points += self.guardUnsafeArea(futureState, currentState, rootState)
                CaptureAgent.debugDraw(self, self.unsafeGuardPos, self.Red, False)

    return points




  def generalPacmanBehavior(self, points, chased, distToGhost, futureState, currentState, rootState):
    rootPos = rootState.getAgentState(self.index).getPosition()
    rootDistStart = abs(self.getMazeDistance(rootPos, self.start))
#    rootFoodList = self.getFood(rootState).asList()
    rootFoodList = sharedMemory.safePillsList
    
    futureChased, futureDistToGhost = self.isChased(futureState)
    
    if chased:
        ''' Chased pacman behavior '''
        if not futureChased or futureDistToGhost > distToGhost:
            points += 1
        else:
            points -= 5

        points += self.returnHome(futureState, rootState, currentState)


    elif ( (rootState.getAgentState(self.index).scaredTimer == 0) and ( len(rootFoodList) <= 2) or (rootState.getAgentState(self.index).numCarrying > 5 and distToGhost <= self.opponentsDistThresh) ) or rootState.data.timeleft*2 + 10<= rootDistStart:
        ''' Food carrying pacman behavior '''
        points += self.returnHome(futureState, rootState, currentState)

    else:
        ''' General attacking behavior '''
        futureFoodList, futureFoodMap, futureNumSafePills = self.getSafePills(futureState)
        currentFoodList, currentFoodMap, currentNumSafePills = self.getSafePills(currentState)
        if sharedMemory.numSafePills > 0:
            if len(sharedMemory.opponentsVisible[self.index]) > 0:         
#                ''' Attacking behavior when Pacman is in ghost state '''
#                if (self.safeEnterPos is not None) and (not rootState.getAgentState(self.index).isPacman) and abs(util.manhattanDistance(rootPos, self.safeEnterPos)) > 1.5:
                points += self.enterSafeArea(futureState, currentState, rootState)
            else:
                points += self.eatPills(futureState, rootState, currentState, futureFoodList, currentFoodList)
        else:
            points += self.returnHome(futureState, rootState, currentState)


    return points




  def getLegalActions(self, state):
    self.allowedBack = False
    self.allowedStop = False

    chased, _ = self.isChased(self.root.state)
    if chased or self.scaredGhost(self.root.state):
        self.allowedBack = True
        self.allowedStop = True
    else:
        if sharedMemory.teammatesBehavior[self.index] in sharedMemory.defensiveBehaviors:
            enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
            if(len(invaders) > 0):
                self.allowedBack = True
            elif self.root.state.getAgentState(self.index).isPacman:
                self.allowedBack = True
            elif sharedMemory.teammatesBehavior[self.index] == "guardUnsafeArea":
                if(self.getMazeDistance(state.getAgentState(self.index).getPosition(), self.unsafeGuardPos) <= 1.5):
                    self.allowedStop = True
                    return ["Stop"]
                else:
                    self.allowedBack = True
                    self.allowedStop = True

    legalActions = state.getLegalActions(self.index)

    if not self.allowedStop:
        if("Stop" in legalActions):
            legalActions.remove("Stop")


    ''' check if any action leads to death '''
    if(self.agentDied(self.root.state) == False):
        for action in legalActions:
          futureState = state.generateSuccessor(self.index, action)
          if self.agentDied(futureState):
              legalActions.remove(action)

    if (len(legalActions) == 0):
        return ["Stop"] # all actions lead to death anyway

    if self.allowedBack:
        if state == self.root.state:
            return legalActions

    ''' Stop agent from moving backwards '''
    prevAction = state.getAgentState(self.index).getDirection()

    reverse = reverseAction(prevAction)
    if prevAction != 'Stop' and len(legalActions) > 1:
        if reverse in legalActions:
            legalActions.remove(reverse)

    return legalActions






class SuperDefensiveAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """


    """
    Picks among the actions with the highest Q(s,a).
    """
    def chooseAction(self, gameState):
        """
        Pick up a random (feasible) action, just in case MCTS needs to be stopped before giving a good result
        """
        time_start = time.time()
        elapsed = 0
        CaptureAgent.debugClear(self)
        actions = gameState.getLegalActions(self.index)
        bestAction = random.choice(actions)

        self.runHistogramFilters(gameState)
        sharedMemory.safePillsList, sharedMemory.safePillsMap, sharedMemory.numSafePills = self.getSafePills(gameState)
        sharedMemory.opponentsVisible[self.index] = self.getOpponentsVisible(gameState)

#        self.debugHistogramFilters()

        """Monte Carlo Tree Search"""
        print("\n*********  MCTS *********")
        print("Agent", self.index, ", Type:", self.agentType())
        if(gameState.getAgentState(self.index).getPosition() == self.start) and (self.getPreviousObservation() is not None):
            sharedMemory.teammatesMessage[self.index] = "help"
            print("AGENT HAS BEEN EATEN!!!")

        if(sharedMemory.teammatesBehavior[self.index] in sharedMemory.defensiveBehaviors):
            if not gameState.getAgentState(self.index).isPacman:
                unsafePos, allSafe = self.estimateUnsafeArea(gameState)
                if self.unsafeGuardPos not in allSafe or self.unsafeGuardPos is None:
                    self.unsafeGuardPos = unsafePos
        else:
            if not gameState.getAgentState(self.index).isPacman:
                safePos, allSafe = self.estimateSafeArea(gameState)
                if self.safeEnterPos not in allSafe or self.safeEnterPos is None:
                    self.safeEnterPos = safePos
                CaptureAgent.debugDraw(self, self.safeEnterPos, self.Green, False)


        sharedMemory.teammatesMessage[self.index] = "noMessage"

        if (self.root.state != gameState):
            self.root = Node(gameState)
            self.root.parent = None
            self.root.children = []
            self.root.plays = 1

        t = 0
        while(t < self.maxTime and elapsed < 0.95):
            self.root = MCTS_sample(self, self.root, self.maxLength)
            t+=1
            elapsed = time.time() - time_start
        next_node = UCB_sample(self, self.root)
        bestAction = next_node.action

        print("Current agent behavior:", sharedMemory.teammatesBehavior[self.index])
        print("Current agent message:", sharedMemory.teammatesMessage[self.index])
        print("Results of MCTS")
        for child in self.root.children:
            print("Action = " + child.action + " ;  UCB = " + str(child.UCB * 100) + "%")
        print("Best action extracted")
        print(bestAction)
        print("Time taken")
        print(time.time()-time_start)
        
        
        
    

        return bestAction




    def getPoints(self, futureState, currentState, rootState):
        points = 0
        chased, distToGhost = self.isChased(rootState)
        sharedMemory.teammatesMessage[self.index] = "noMessage"
        

        ''' Scared ghost behavior '''
        if self.scaredAgent(rootState):
            if chased:
                return self.returnHome(futureState, rootState, currentState)
            else:
                return self.generalPacmanBehavior(points, chased, distToGhost, futureState, currentState, rootState)


        invadersIndices = [i for i in self.opponentsIndices if rootState.getAgentState(i).isPacman]
        distToTeammate = [self.getMazeDistance(rootState.getAgentState(self.index).getPosition(), rootState.getAgentState(i).getPosition()) for i in self.teammatesIndices if i != self.index][0]
        diffOpponentsVisible = [opponentIndex for opponentIndex in sharedMemory.opponentsVisible[self.index] if opponentIndex not in sharedMemory.opponentsVisible[self.otherTeammateIndex]]
        

        if(len(invadersIndices) >= 0) and (len(invadersIndices) < 2) and distToTeammate >= self.gridWidth/3 and len(diffOpponentsVisible) > 0:
            points += self.generalGhostBehavior(points, futureState, currentState, rootState)
        elif (len(invadersIndices) >= 0) and (len(invadersIndices) < 2) and distToTeammate < self.gridWidth/3:
            points += self.generalPacmanBehavior(points, chased, distToGhost, futureState, currentState, rootState)
        else:
            points += self.generalPacmanBehavior(points, chased, distToGhost, futureState, currentState, rootState)



        if futureState.getScore() < rootState.getScore():
            points += self.scoreSign
        elif futureState.getScore() > rootState.getScore():
            points -= self.scoreSign

        return points





class HungryAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """


    """
    Picks among the actions with the highest Q(s,a).
    """
    def chooseAction(self, gameState):
        """
        Pick up a random (feasible) action, just in case MCTS needs to be stopped before giving a good result
        """
        time_start = time.time()
        elapsed = 0
#        CaptureAgent.debugClear(self)
        actions = gameState.getLegalActions(self.index)
        bestAction = random.choice(actions)

        self.runHistogramFilters(gameState)
        sharedMemory.safePillsList, sharedMemory.safePillsMap, sharedMemory.numSafePills = self.getSafePills(gameState)
        sharedMemory.opponentsVisible[self.index] = self.getOpponentsVisible(gameState)

        """Monte Carlo Tree Search"""
        print("\n*********  MCTS *********")
        print("Agent", self.index, ", Type:", self.agentType())
        if(gameState.getAgentState(self.index).getPosition() == self.start) and (self.getPreviousObservation() is not None):
            sharedMemory.teammatesMessage[self.index] = "help"
            print("*********************************************************")
            print("EATEN!!!")
            print("*********************************************************")

        # change safe enter position if needed
        if(sharedMemory.teammatesBehavior[self.index] in sharedMemory.defensiveBehaviors):
            if not gameState.getAgentState(self.index).isPacman:
                unsafePos, allSafe = self.estimateUnsafeArea(gameState)
                if self.unsafeGuardPos not in allSafe or self.unsafeGuardPos is None:
                    self.unsafeGuardPos = unsafePos
        else:
            if not gameState.getAgentState(self.index).isPacman:
                safePos, allSafe = self.estimateSafeArea(gameState)
                if self.safeEnterPos not in allSafe or self.safeEnterPos is None:
                    self.safeEnterPos = safePos
                    CaptureAgent.debugDraw(self, self.safeEnterPos, self.Blue, False)


        sharedMemory.teammatesMessage[self.index] = "noMessage"

        if (self.root.state != gameState):
            self.root = Node(gameState)
            self.root.parent = None
            self.root.children = []
            self.root.plays = 1

        t = 0
        while(t < self.maxTime and elapsed < 0.95):
            self.root = MCTS_sample(self, self.root, self.maxLength)
            t+=1
            elapsed = time.time() - time_start
        next_node = UCB_sample(self, self.root)
        bestAction = next_node.action

        print("Current agent behavior:", sharedMemory.teammatesBehavior[self.index])
        print("Current agent message:", sharedMemory.teammatesMessage[self.index])
        print("Results of MCTS")
        for child in self.root.children:
            print("Action = " + child.action + " ;  UCB = " + str(child.UCB * 100) + "%")
        print("Best action extracted")
        print(bestAction)
        print("time taken")
        print(time.time()-time_start)

        return bestAction





    def getPoints(self, futureState, currentState, rootState):
        points = 0
        chased, distToGhost = self.isChased(rootState)
        sharedMemory.teammatesMessage[self.index] = "noMessage"

        ''' Scared ghost behavior '''
        if self.scaredAgent(rootState):
            if chased:
#                return self.runFromPacman(rootState, futureState)
                return self.returnHome(futureState, rootState, currentState)
            elif rootState.getAgentState(self.index).numCarrying > 10:
                return self.returnHome(futureState, rootState, currentState)
            else:
                return self.generalPacmanBehavior(points, chased, distToGhost, futureState, currentState, rootState)


        if rootState.getAgentState(self.index).isPacman==False and len(sharedMemory.opponentsVisible[self.index]) > 0:
            points += self.generalGhostBehavior(points, futureState, currentState, rootState)
        else:
            points += self.generalPacmanBehavior(points, chased, distToGhost, futureState, currentState, rootState)


        if futureState.getScore() < rootState.getScore():
            points += self.scoreSign
        elif futureState.getScore() > rootState.getScore():
            points -= self.scoreSign

        return points




        #                points += self.generalGhostBehavior(points, futureState, currentState, rootState)
class DoNothingAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    return "Stop"
