#
# States
#
StateSuccess = 1
StateRunning = 0
StateFailure = -1

#
# Composite
#
class Sequence():

    def __init__(self, subnodes, param=None):
        self.subnodes = subnodes
        self.param = param

    def process(self, context, param=None):
        if self.param is not None:
            param = self.param

        state = StateSuccess
        for i in xrange(len(self.subnodes)-1):
            subnode = self.subnodes[i].process(context, param)

            state = StateRunning
            while state == StateRunning:
                state = subnode.next()
                if state == StateRunning:
                    yield state

            if state == StateSuccess:
                yield StateRunning
            else:
                break

        if state == StateSuccess:
            state = StateRunning
            proc = self.subnodes[-1].process(context, param)
            while state == StateRunning:
                state = proc.next()
                if state == StateRunning:
                    yield state

            yield state
        else:
            yield StateFailure

class Selector():
    def __init__(self, subnodes, param=None):
        self.subnodes = subnodes
        self.param = param

    def process(self, context, param=None):
        if self.param is not None:
            param = self.param

        state = StateFailure

        for i in xrange(len(self.subnodes)-1):
            subnode = self.subnodes[i].process(context, param)

            state = StateRunning
            while state == StateRunning:
                state = subnode.next()
                if state == StateRunning:
                    yield state

            if state == StateFailure:
                yield StateRunning
            else:
                break
        
        if state == StateSuccess:
            yield StateSuccess
        else:
            state = StateRunning
            proc = self.subnodes[-1].process(context, param)
            while state == StateRunning:
                state = proc.next()
                if state == StateRunning:
                    yield state

            yield state

#
# Decorator
#
class Inverter():
    def __init__(self, subnode, param=None):
        self.subnode = subnode
        self.param = param

    def process(self, context, param=None):
        if self.param is not None:
            param = self.param

        state = StateRunning
        proc = self.subnode.process(context, param)

        while state == StateRunning:
            state = proc.next()

            if state == StateRunning:
                yield StateRunning

        if state == StateSuccess:
            yield StateFailure
        elif state == StateFailure:
            yield StateSuccess

class Succeeder():
    def __init__(self, subnode, param=None):
        self.subnode = subnode
        self.param = param

    def process(self, context, param=None):
        if self.param is not None:
            param = self.param

        state = StateRunning
        proc = self.subnode.process(context, param)

        while state == StateRunning:
            state = proc.next()

            if state == StateRunning:
                yield state

        yield StateSuccess

class ParamNode():
    def __init__(self, subnode, param=None):
        self.subnode = subnode
        self.param = param

    def process(self, context, param=None):
        if self.param is not None:
            param = self.param

        state = StateRunning
        proc = self.subnode.process(context, param)

        while state == StateRunning:
            state = proc.next()

            if state == StateRunning:
                yield StateRunning

        yield state

class Leaf():
    def __init__(self, func, param=None):
        self.func = func
        self.param = param

    def process(self, context, param=None):
        if self.param is not None:
            param = self.param

        if param is None:
            yield self.func(context)
        else:
            yield self.func(context, param)

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first = 'TreeAgent', second = 'TreeAgent'):
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

class TreeAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.tree = None
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def getContext(self, gameState):
        self.gameState = gameState
        context = dict()

        context['gamestate'] = gameState

        context['possible_actions'] = gameState.getLegalActions(self.index)
        context['current_position'] = gameState.getAgentPosition(self.index)
        context['is_pacman'] = gameState.getAgentState(self.index).isPacman

        indexes = self.getTeam(gameState)

        if indexes[0] == self.index:
            context['agent2_position'] = gameState.getAgentPosition(indexes[1])
        else:
            context['agent2_position'] = gameState.getAgentPosition(indexes[0])

        opponents = self.getOpponents(gameState)
        context['opponent_positions'] = map(lambda i: gameState.getAgentState(i).getPosition(), opponents)
        context['opponent_distance'] = map(lambda i: gameState.getAgentDistances()[i], opponents)
        context['opponent_pacman'] = map(lambda i: gameState.getAgentState(i).isPacman, opponents)

        context['food'] = self.getFood(gameState).asList()
        context['own_food'] = self.getFoodYouAreDefending(gameState).asList()

        context['carried_food'] = gameState.getAgentState(self.index).numCarrying

        context['carried_food_opponents'] = map(lambda i: gameState.getAgentState(i).numCarrying, opponents)

        # print "================Opponent carrying food ==================", context['carried_food_opponents']

        context['scared_timer'] = map(lambda i: gameState.getAgentState(i).scaredTimer, opponents)

        # print "########### Scared timer ############", context['scared_timer']

        return context

    def createTree(self):
        def areWePacman(context):
            if context['is_pacman']:
                return StateSuccess
            else:
                return StateFailure

        def isOpponentOnOurSide(context, index):
            """
            Checks if an opponent is in our area
            """
            if context['opponent_pacman'][index] and context['opponent_positions'][index] is not None:
                # print "Opponent on our side, pos = ", context['opponent_positions'][index]
                return StateSuccess
            else:
                return StateFailure

        def getOpponentPosition(context, index):
            previous_state = self.getPreviousObservation()

            if context['opponent_positions'][index] is None and previous_state is not None:
                pos = set(self.getFoodYouAreDefending(previous_state).asList()) - set(context['own_food'])
                pos = list(pos)

                if len(pos) == 0:
                    return StateFailure

                context['opponent_positions'][index] = pos[0]

                # print "Opponent on our side, estimated pos = ", context['opponent_positions'][index], context['own_food'], self.getFoodYouAreDefending(previous_state).asList() 
                return StateSuccess
            else:
                return StateFailure


        def areWeNearToOpponent(context, index):
            """
            Checks if the current agent is the closes agent
            """
            if context['opponent_positions'][index] is None:
                return StateFailure

            own_dist = self.getMazeDistance(context['current_position'], context['opponent_positions'][index])
            agent2_dist = self.getMazeDistance(context['agent2_position'], context['opponent_positions'][index])
            if own_dist < agent2_dist:
                context['opponent_to_catch'] = index
                return StateSuccess
            else:
                return StateFailure


        def inOwnArea(context, param=None):
            """
            Is our agent in our own area
            """
            if not context['is_pacman']:
                return StateSuccess
            else:
                return StateFailure


        def isOpponentScared(context, index):
            # print "isOpponentScared"
            gameState = context['gamestate']
            index = self.getOpponents(gameState)[index]
            opponent_pos = gameState.getAgentState(index).getPosition()
            scared = gameState.getAgentState(index).scaredTimer
            # print "scared: ", scared
            # print "index: ", index

            if opponent_pos is not None and scared > 5:
                return StateSuccess
            else:
                return StateFailure

        def removeDangerousActionsAgent(context, index):
            # print "########################"
            # print "removeDangerousActionsAgent"
            gameState = context['gamestate']
            index = self.getOpponents(gameState)[index]
            opponent = gameState.getAgentState(index).getPosition()
            isOpponentPacman = gameState.getAgentState(index).isPacman
            # print "index: ", index

            our_pos = context['current_position']

            good_actions = []
            second_best_action = []

            for act in context['possible_actions']:
                if act == 'Stop':
                    new_pos = our_pos
                elif act == 'North':
                    new_pos = (our_pos[0], our_pos[1]+1)
                elif act == 'East':
                    new_pos = (our_pos[0]+1, our_pos[1])
                elif act == 'South':
                    new_pos = (our_pos[0], our_pos[1]-1)
                elif act == 'West':
                    new_pos = (our_pos[0]-1, our_pos[1])


                if opponent is not None:
                    dist = self.getMazeDistance(new_pos, opponent)
                    if not isOpponentPacman and dist < 2:
                        continue
                    elif not isOpponentPacman and dist < 3:
                        second_best_action.append(act)
                        continue


                good_actions.append(act)

            if len(good_actions) > 0:
                context['possible_actions'] = good_actions
            elif len(second_best_action) > 0:
                context['possible_actions'] = second_best_action
            else:
                context['possible_actions'] = ['Stop']
            
            return StateSuccess

        
        def removeDangerousActions(context):
            opponent0 = context['opponent_positions'][0]
            opponent1 = context['opponent_positions'][1]

            our_pos = context['current_position']

            good_actions = []
            second_best_action = []

            for act in context['possible_actions']:
                if act == 'Stop':
                    new_pos = our_pos
                elif act == 'North':
                    new_pos = (our_pos[0], our_pos[1]+1)
                elif act == 'East':
                    new_pos = (our_pos[0]+1, our_pos[1])
                elif act == 'South':
                    new_pos = (our_pos[0], our_pos[1]-1)
                elif act == 'West':
                    new_pos = (our_pos[0]-1, our_pos[1])

                if opponent0 is not None and context['scared_timer'][0] == 0 and not context['opponent_pacman'][0]:
                    dist = self.getMazeDistance(new_pos, opponent0)
                    if not context['opponent_pacman'][0] and dist < 2:
                        continue
                    elif not context['opponent_pacman'][0] and dist < 3:
                        second_best_action.append(act)
                        continue
                
                if opponent1 is not None and context['scared_timer'][1] == 0 and not context['opponent_pacman'][1]:
                    dist = self.getMazeDistance(new_pos, opponent1)
                    if not context['opponent_pacman'][1] and dist < 2:
                        continue
                    elif not context['opponent_pacman'][1] and dist < 3:
                        second_best_action.append(act)
                        continue
                    
                good_actions.append(act)

            if len(good_actions) > 0:
                context['possible_actions'] = good_actions
            elif len(second_best_action) > 0:
                context['possible_actions'] = second_best_action
            else:
                context['possible_actions'] = ['Stop']

            # print "===========", good_actions
            
            return StateSuccess


        def removeDangerousActionsOld(context):
            opponent_to_catch = context['opponent_positions'][context['opponent_to_catch']]

            if context['opponent_to_catch'] == 0:
                other_oppenent = context['opponent_positions'][1]
            else:
                other_oppenent = context['opponent_positions'][0]

            our_pos = context['current_position']

            good_actions = []

            for act in context['possible_actions']:
                if act == 'Stop':
                    new_pos = our_pos
                elif act == 'North':
                    new_pos = (our_pos[0], our_pos[1]+1)
                elif act == 'East':
                    new_pos = (our_pos[0]+1, our_pos[1])
                elif act == 'South':
                    new_pos = (our_pos[0], our_pos[1]-1)
                elif act == 'West':
                    new_pos = (our_pos[0]-1, our_pos[1])

                if new_pos != other_oppenent and np.sum(np.abs(np.array(new_pos)-np.array(other_oppenent))) > 1:
                    good_actions.append(act)

            if len(good_actions) > 0:
                context['possible_actions'] = good_actions
            else:
                context['possible_actions'] = ['Stop']
            
            return StateSuccess


        def moveToOpponent(context):
            """
            Move to opponent directly
            """
            pos = context['opponent_positions'][context['opponent_to_catch']]
            our_pos = context['current_position']

            if pos is not None:
                values = []
                for act in context['possible_actions']:
                    if act == 'Stop':
                        values.append(self.getMazeDistance(our_pos, pos))
                    elif act == 'North':
                        values.append(self.getMazeDistance((our_pos[0], our_pos[1]+1), pos))
                    elif act == 'East':
                        values.append(self.getMazeDistance((our_pos[0]+1, our_pos[1]), pos))
                    elif act == 'South':
                        values.append(self.getMazeDistance((our_pos[0], our_pos[1]-1), pos))
                    elif act == 'West':
                        values.append(self.getMazeDistance((our_pos[0]-1, our_pos[1]), pos))

                context['best_action'] = context['possible_actions'][np.argmin(values)]

                return StateSuccess
            else:
                return StateFailure

        def findNearestFood(context):
            our_pos = context['current_position']

            min_dist = 10e6

            for i in xrange(len(context['food'])):
                pos = context['food'][i]
                dist = self.getMazeDistance(our_pos, pos)

                if dist < min_dist:
                    context['nearest_food'] = pos
                    min_dist = dist

            return StateSuccess
            

        def moveToFood(context):
            our_pos = context['current_position']

            min_dist = 10e6

            try:
                pos = context['nearest_food']
            except KeyError:
                # print "No nearest food"
                return StateFailure

            values = []
            for act in context['possible_actions']:
                if act == 'Stop':
                    values.append(self.getMazeDistance(our_pos, pos))
                elif act == 'North':
                    values.append(self.getMazeDistance((our_pos[0], our_pos[1]+1), pos))
                elif act == 'East':
                    values.append(self.getMazeDistance((our_pos[0]+1, our_pos[1]), pos))
                elif act == 'South':
                    values.append(self.getMazeDistance((our_pos[0], our_pos[1]-1), pos))
                elif act == 'West':
                    values.append(self.getMazeDistance((our_pos[0]-1, our_pos[1]), pos))

            context['best_action'] = context['possible_actions'][np.argmin(values)]

            return StateSuccess


        def gotEnoughFood(context):
            if context['carried_food'] >= 5:
                return StateSuccess
            else:
                return StateFailure

        def moveToStart(context):
            our_pos = context['current_position']

            min_dist = 10e6

            pos = self.start

            values = []
            for act in context['possible_actions']:
                if act == 'Stop':
                    values.append(self.getMazeDistance(our_pos, pos))
                elif act == 'North':
                    values.append(self.getMazeDistance((our_pos[0], our_pos[1]+1), pos))
                elif act == 'East':
                    values.append(self.getMazeDistance((our_pos[0]+1, our_pos[1]), pos))
                elif act == 'South':
                    values.append(self.getMazeDistance((our_pos[0], our_pos[1]-1), pos))
                elif act == 'West':
                    values.append(self.getMazeDistance((our_pos[0]-1, our_pos[1]), pos))

            context['best_action'] = context['possible_actions'][np.argmin(values)]

            return StateSuccess

        
        def checkOpponentPrecedence(context):
            carried_food_opponents = context['carried_food_opponents']
            if carried_food_opponents[0] >= carried_food_opponents[1]:
                return StateSuccess
            else:
                return StateFailure
                

        def saveDefenseState(context, index):
            self.defendCounter = 15
            self.defendIndex = index
            self.opponentPosition = context['opponent_positions'][index]

            return StateSuccess


        def recoverDefenceState(context):
            if self.defendCounter > 0:
                context['opponent_to_catch'] = self.defendIndex
                
                if context['opponent_positions'][self.defendIndex] is None:
                    previous_state = self.getPreviousObservation()
                    pos = set(self.getFoodYouAreDefending(previous_state).asList()) - set(context['own_food'])
                    pos = list(pos)
                    if len(pos) > 0:
                        context['opponent_positions'][self.defendIndex] = pos[0]
                    else:
                        context['opponent_positions'][self.defendIndex] = self.opponentPosition
                
                self.defendCounter -= 1
                self.opponentPosition = context['opponent_positions'][self.defendIndex]

                if self.opponentPosition == context['current_position'] or not context['opponent_pacman'][self.defendIndex]:
                    self.defendCounter = 0
                    return StateFailure
               
                return StateSuccess

            else:
                return StateFailure



        shouldDefend = Sequence([
                Selector([
                    Leaf(isOpponentOnOurSide),
                    Leaf(getOpponentPosition)
                ]),
                Leaf(areWeNearToOpponent),
                Leaf(saveDefenseState)
            ])

        catchOpponent = Selector([
            Sequence([
                Leaf(inOwnArea),
                Leaf(moveToOpponent)
            ]),
            Sequence([
                Leaf(removeDangerousActions),
                Leaf(moveToOpponent)
            ])
        ])

        defend = Sequence([
            Selector([
                Leaf(recoverDefenceState),
                Selector([
                    Sequence([
                        Leaf(checkOpponentPrecedence),
                        ParamNode(shouldDefend, 0)
                    ]),
                    ParamNode(shouldDefend, 1)
                ])
            ]),
            catchOpponent
        ])


        # scared_tactics = Sequence([
        #     Selector([
        #         Leaf(isOpponentScared),
        #         Leaf(removeDangerousActionsAgent)
        #         ], 0), 
        #         Selector([
        #         Leaf(isOpponentScared),
        #         Leaf(removeDangerousActionsAgent)
        #         ], 1)
        #     ])

        def doRandomMove(context):
            return StateFailure

        def opponentIsClose(context, param=None):
            if context['opponent_positions'][0] is not None and context['scared_timer'][0] < 3 and not context['opponent_pacman'][0]:
                if self.getMazeDistance(context['current_position'], context['opponent_positions'][0]) <= 3: #and context['carried_food'] > 0:
                    return StateSuccess

            if context['opponent_positions'][1] is not None and context['scared_timer'][1] < 3 and not context['opponent_pacman'][1]:
                if self.getMazeDistance(context['current_position'], context['opponent_positions'][1]) <= 3: #and context['carried_food'] > 0:
                    return StateSuccess

            return StateFailure

        collect = Selector([
            Sequence([
                Leaf(gotEnoughFood),
                Leaf(removeDangerousActions),
                Leaf(moveToStart)
            ]),
            Sequence([
                Leaf(findNearestFood),
                Leaf(removeDangerousActions),
                Selector([
                    Sequence([
                        Inverter(Leaf(opponentIsClose)),
                        Leaf(moveToFood)
                    ]),
                    Leaf(moveToStart)
                ])
            ])
        ])

        self.tree = Selector([
            defend,
            collect
        ])

        self.move_seq = []
        self.stuck_cnt = 0
        self.defendCounter = 0
        self.defendIndex = -1
        self.opponentPosition = None

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        if self.tree is None:
            self.createTree()

        context = self.getContext(gameState)

        state = StateRunning

        proc = self.tree.process(context)
        while state == StateRunning:
            state = proc.next()

        # print "state=", state, "context =", context

        if state == StateFailure:
            act = random.choice(context['possible_actions'])
            self.move_seq.append(act)
            return act
        else:
            # print "Best action:", context['best_action'], self.index
            self.move_seq.append(context['best_action'])
            return context['best_action']
