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
from game import Directions
import game
from util import nearestPoint
from util import normalize
import numpy as np
from numpy import *
from util import pause
from util import Counter
from game import Actions as AT
import distanceCalculator

#################
# Team creation #
#################


def nextag(i):
    if i==3:
        return 
    else:
        return i+1

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




class brain():
    def __init__(self,isRed):
        self.red=isRed
        self.distancer = None
        self.isinit=0
        self.size=None
        self.boundR=[]
        self.boundB=[]

        self.key_boundR=[]
        self.key_boundB=[]

        self.agent1 = 0
        self.agent2 = 0
        self.patrol_aim1 = 0
        self.patroling_agent1 = 0
        self.patrol_aim2 = 0
        self.patroling_agent2 = 0

        self.agentstate=[int(1),int(1)]





    def init(self,gameState):
        self.totalfood=len(gameState.getRedFood().asList())



        self.distancer = distanceCalculator.Distancer(gameState.data.layout)
        self.size=(gameState.data.layout.width,gameState.data.layout.height)
        i=int(self.size[0]/2)



        for j in range(0,self.size[1]):
            loc=(i-1,j)
            locb=(i,j)
            if(gameState.hasWall(loc[0],loc[1])):
                continue
            if(gameState.hasWall(locb[0],locb[1])):
                continue
            self.boundR.append(loc)
            self.boundB.append(locb)

        foods=gameState.getBlueFood()
        kdR=Counter()
        for food in foods:
            for bd in self.boundB:
                d=int(self.getMazeDistance(food,bd))
                id=int(self.boundB.index(bd))
                kdR[id] +=d
        bdlist=kdR.sortedKeys()
        self.key_boundR=[self.boundB[ i] for i in bdlist[0:int(len(bdlist)/1) ]]
        

        foods=gameState.getRedFood()
        kdB=Counter()
        for food in foods:
            for bd in self.boundR:
                d=int(self.getMazeDistance(food,bd))
                id=int(self.boundR.index(bd))
                kdB[id] +=d
        bdlist=kdB.sortedKeys()
        self.key_boundB=[self.boundR[ i] for i in bdlist[0:int(len(bdlist)/1) ]]



        self.rndgoal=[]


        self.numag=None

        if(self.red):

            self.numag=gameState.getRedTeamIndices()
        else:
            self.numag=gameState.getBlueTeamIndices()

        self.isinit=1


    def getgoal(self,ag):
        if len(self.rndgoal)==0:

            # print self.key_boundR
            # random.choice(self.key_boundR)
            if(self.red):
                self.rndgoal.append(self.key_boundR[random.randint(0,len(self.key_boundR)-1)])
                self.rndgoal.append(self.key_boundR[random.randint(0,len(self.key_boundR)-1)])
            else:
                self.rndgoal.append(self.key_boundB[random.randint(0,len(self.key_boundB)-1)])
                self.rndgoal.append(self.key_boundB[random.randint(0,len(self.key_boundB)-1)])
        return self.rndgoal[ag]

    def getnum(self,index):
        return self.numag.index(index)


    def getMazeDistance(self, pos1, pos2):
        """
        Returns the distance between two points; These are calculated using the provided
        distancer object.

        If distancer.getMazeDistances() has been called, then maze distances are available.
        Otherwise, this just returns Manhattan distance.
        """
        d = self.distancer.getDistance(pos1, pos2)
        return d









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
    ## Things I've added
    self.start = gameState.getAgentPosition(self.index)
    #self.check_for_enemies = check_for_enemies() 
    #self.BT = Behavior_Tree()

    self.lastaction=None


    global mainBrain

    mainBrain=brain(self.red)

    if(mainBrain.isinit==0 ):
        mainBrain.init(gameState)
        kl = self.getTeam(gameState)
        mainBrain.patroling_agent1 = kl[0]
        mainBrain.patroling_agent2 = kl[1]
        mainBrain.patrol_aim1 = 0
        mainBrain.patrol_aim2 = 0
        mainBrain.agent1=kl[0]
        mainBrain.agent2=kl[1]



    global enemyBrain

    enemyBrain=brain(not self.red)

    if(enemyBrain.isinit==0 ):
        enemyBrain.init(gameState)





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

#########minime maxima



  def abvalue(self,gameState, ghost,pacman,isGhostmove):

    

    gpos=gameState.getAgentPosition(ghost)
    ppos=gameState.getAgentPosition(pacman)
    D=self.getMazeDistance(gpos,ppos)

    d=np.Inf


    #print(gameState.isOnRedTeam(pacman))

    if (gameState.isOnRedTeam(pacman)==True):

        bound=mainBrain.boundR
    else:
        bound=mainBrain.boundB
    cell=None

    for bd in bound:
        dist=self.getMazeDistance(ppos,bd)
        if dist<d:
           d=dist
           cell=bd
    #self.debugDraw(bd,(100,100,100),False)

    #print(D,d,1.5*D-d)

    return 1.5*D-0.5*d






  def absearch(self,gameState, ghost,pacman,isGhost):
    self.debugClear()
    abit=int(7)



    a=-np.Inf
    b=np.Inf


    if isGhost==False:
        value,action=self.maxvalue(gameState,ghost,pacman,a,b,abit)
    else:
        value,action=self.minvalue(gameState,ghost,pacman,a,b,abit)

    #print(value)
    #print(action)

    # pause()

    # if(isGhost):
    #     print(value)
    #     print(action)
    #     pause()

    return action





  def maxvalue(self,gameState, ghost,pacman,a,b,it):###pac want it to max
    if(it!=7):

        if(gameState.getAgentState(pacman).start==gameState.getAgentState(pacman).configuration):
            #print("DDDDDDDDDDDDDDDDDDDDDDDead")
            # ppp=gameState.getAgentPosition(ghost)
            # if(ppp!=None):
            #     self.debugDraw(ppp,(100,100,100),False)
            return -1000*it,'Stop'
        if(gameState.getAgentPosition(pacman)==None):
            return -500,'Stop'

        if(gameState.getAgentPosition(ghost)==None):
            return 500,'Stop'

        if(gameState.getAgentState(ghost).scaredTimer>0):
            return 500,'Stop'


        if (gameState.getAgentState(ghost).isPacman==True):
            #print("Noob!")
            return 500,'Stop'




    if it==0:
        return self.abvalue(gameState,ghost,pacman,True),0#value after a ghost move
    v=-np.Inf
    ract=None
    actions = gameState.getLegalActions(pacman)
    for act in actions:
        succ=gameState.generateSuccessor(pacman, act)
        getm,token=self.minvalue(succ, ghost,pacman,a,b,it-1)
        if getm>v:
            v=getm
            ract=act
        if(v>=b):
            # print("it=",it)
            # print("value",v,act)
            return v,ract
        a=max(a,v)
        #print("a===============",a)
    # print("it=",it)
    # print("value",v,act)
    return v,ract







  def minvalue(self,gameState, ghost,pacman,a,b,it):### ghost want it to min

    if(it!=7):
        if(gameState.getAgentState(pacman).start==gameState.getAgentState(pacman).configuration):
            #print("DDDDDDDDDDDDDDDDDDDDDDDead")
            # ppp=gameState.getAgentPosition(ghost)
            # if(ppp!=None):
            #     self.debugDraw(ppp,(100,100,100),False)
            return -1000*it,0
        if(gameState.getAgentPosition(pacman)==None):
           
            return 500,1

        if(gameState.getAgentPosition(ghost)==None):
            
            return -500,2

        if(gameState.getAgentState(ghost).scaredTimer>0):
            return 500,3


        if (gameState.getAgentState(ghost).isPacman==True):
            #print("Noob!")
            return 500,4


    if it==0:
        return self.abvalue(gameState,ghost,pacman,False),0#value after a pacman move
    v=np.Inf
    ract='Stop'
    actions = gameState.getLegalActions(ghost)
    for act in actions:
        succ=gameState.generateSuccessor(ghost, act)
        getm,token=self.maxvalue(succ, ghost,pacman,a,b,it-1)
        if getm<v:
            v=getm
            ract=act
        #print("!!!!!!!!!!!!!!!!!!!!!",v,a)
        if(v<=a):
            # print("it=",it)
            # print("value",v,act)
            return v,ract
        b=min(b,v)
        #print("b====================",b)
    # print("it=",it)
    # print("value",v,act)
    return v,ract


  def capcommand(self,gameState):
    st=0
    for enemy in self.getOpponents(gameState):
        if gameState.getAgentState(enemy).scaredTimer>st:
            st=gameState.getAgentState(enemy).scaredTimer
    caps=self.getCapsules(gameState)
    team=self.getTeam(gameState)

    response='Stop'
    mindist=np.Inf

    for ag in team:
        if(mainBrain.agentstate[ mainBrain.getnum(ag)]==0):
            continue
        for cap in caps:
            dist=self.getMazeDistance(cap,gameState.getAgentPosition(ag))
            if dist<mindist:
                mindist=dist
                response=ag

    if(mindist<st):
        return None
    else:
        #print ag
        return ag




  def statemachine(self,gameState):
    state=mainBrain.agentstate[ mainBrain.getnum(self.index)]
    next=state
    Teamstate=sum(mainBrain.agentstate)
    me=gameState.getAgentState(self.index)
    pos=gameState.getAgentPosition(self.index)
    if(state==1):
        bigboss=False
        nearby=False
        scared=False
        score=self.getScore(gameState)
        for enemy in self.getOpponents(gameState):
            st=gameState.getAgentState(enemy)
            loc=gameState.getAgentPosition(enemy)

            if st.numCarrying>=mainBrain.totalfood/2:
                bigboss=True
            if st.scaredTimer>=5:
                scared=True

            if loc==None:
                continue
            if st.isPacman==False:
                continue
        
            if self.getMazeDistance(loc,pos)<=4:
                nearby=True
        # if Teamstate-state==1:
        #     next=0
        if nearby:
            next=0

        if score<-mainBrain.totalfood/2:
            next=1
        if bigboss:
            next=0
        if score>mainBrain.totalfood/2:
            next=0
    else:
        bigboss=False
        score=self.getScore(gameState)
        for enemy in self.getOpponents(gameState):
            st=gameState.getAgentState(enemy)
            loc=gameState.getAgentPosition(enemy)

            if st.numCarrying>=mainBrain.totalfood/2:
                bigboss=True

        if Teamstate-state==0:
            next=1

        if score<-mainBrain.totalfood/2:
            next=1
        if bigboss:
            next=0
        if score>mainBrain.totalfood/2:
            next=0


    mainBrain.agentstate[ mainBrain.getnum(self.index)]=next














###########choose action

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    self.statemachine(gameState)

    if(mainBrain.agentstate[ mainBrain.getnum(self.index)]==1):
        action=self.offPlan(gameState)
    else:
        action=self.DeffPlan(gameState)


    self.lastaction=action

    #pause()

    return action




  def DeffPlan(self, gameState):
    actions = gameState.getLegalActions(self.index)
    self.patrol(gameState)
    # print("aim!!!!!!!!!!!!!!!!!!1",mainBrain.patrol_aim)
    # self.debugClear()
    # self.debugDraw(mainBrain.patrol_aim,(100,100,50),False)
    A=zeros((len(actions),9))
    i=0
    for action in actions:
        successor = self.getSuccessor(gameState, action)
        w=self.getDefensivefeature(successor)
        #print(w)
        A[i]=w
        i=i+1
    #print(A, "Where is this")
    
    # 0. Distance to own food
    # 1. Food sum
    # 2. Distance to enemy Pacman
    # 3. Distance to our capsule if it's still present
    # 4. dist to bound
    # 5. score
    # 6. Nr of enemy ghosts
    # 7 Get to patrol point!
    # 8. Penalize Pacmanization

    Wlist=[]
    w0=np.array([[0, 0, -20, -5, -3, 100, 500, -10, -20]])#go to enemy
    w1=np.array([[0, 0, 0, 0, -5, 0, 500, -10, -20]])#Patrol, if higher score then patrolling might be good
    w2= np.array([[10, 0, 0, 1, 0, 0, 1, 0, -20]])  # Un-Pacmanize
    #w2=np.array([[0,0,100,   0,   -10,0,500,  0,100]])#cap
    #w3=np.array([[0,-0.1,100,   101,   0,0,0,  -1,100]])#hide
    #w4=np.array([[-1,-0.01,100,   1,   0,0,0,  0,100]])#next


    Wlist.append(w0)
    Wlist.append(w1)
    Wlist.append(w2)
    #Wlist.append(w2)
    #Wlist.append(w3)
    #Wlist.append(w4)

    me=gameState.getAgentState(self.index)
    selfpos=me.getPosition()

    ghost=-1
    if me.isPacman:
        bh = 2
        #print "whwhwhwwhwhww"
        #print self.index, "turk"
    else:
        enemys=self.getOpponents(gameState)
        for enemy in enemys:
            ag=gameState.getAgentState(enemy)
            if ag.numCarrying>1:#3:
                bh = 0
                break
            else:
                
                bh = 1

    pacman=-1
    enemydist=np.Inf


    if(bh==0):
        for enemy in enemys:
            if(gameState.getAgentState(enemy).isPacman==False):
                continue
            agpos=gameState.getAgentPosition(enemy)
            if(agpos!=None):
                dist=self.getMazeDistance(agpos, selfpos)
                if dist<enemydist:
                    enemydist=dist
                    pacman=enemy

        #print("LLLLLLLLLLLong",enemydist)

        if (pacman!=-1):
            #print("to MMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
            return self.absearch(gameState,self.index,pacman,True)







    #print("agent",self.index,"bh=: ",bh)



    #if(bh==3): # The minmax
    #    return self.absearch(gameState, ghost,self.index,False)




    W=Wlist[bh]

    S=np.dot(A,W.T).T[0]

    stopid=actions.index('Stop')
    S[stopid]=S[stopid]-0.1*abs(S[stopid])


    #print(actions)

    #print(S)

    maxValue=max(S)


    bestActions = [a for a, v in zip(actions, S) if v == maxValue]
    #print bestActions, "wat"
    #util.pause()
    return random.choice(bestActions)

    
  def patrol(self, gameState):
    #print "Its patrol time MOFO"
    current_agent = self.index
    if current_agent == mainBrain.agent1:
        if mainBrain.patrol_aim1 == 0:
            foods = self.getFoodYouAreDefending(gameState).asList()
            mainBrain.patrol_aim1 = foods[random.randint(0,len(foods)-1)]
            mainBrain.patroling_agent1 = current_agent    

        pos2 = gameState.getAgentPosition(self.index) # Supposed to be current position
        if self.getMazeDistance(pos2,mainBrain.patrol_aim1) == 0:
            foods = self.getFoodYouAreDefending(gameState).asList()
            mainBrain.patrol_aim1 = foods[random.randint(0,len(foods)-1)]
    else:
        if mainBrain.patrol_aim2 == 0:
            foods = self.getFoodYouAreDefending(gameState).asList()
            mainBrain.patrol_aim2 = foods[random.randint(0,len(foods)-1)]
            mainBrain.patroling_agent2 = self.index    

        pos2 = gameState.getAgentPosition(self.index) # Supposed to be current position
        if self.getMazeDistance(pos2,mainBrain.patrol_aim2) == 0:
            foods = self.getFoodYouAreDefending(gameState).asList()
            mainBrain.patrol_aim2 = foods[random.randint(0,len(foods)-1)]



    


  def offPlan(self, gameState):
    #print(self.index)
    actions = gameState.getLegalActions(self.index)
 
    A=zeros((len(actions),10))
    i=0
    for action in actions:
        successor = self.getSuccessor(gameState, action)
        w=self.getfeature(successor)
        #print(w)
        A[i]=w
        i=i+1
    #print(A)


    # 0. distance to nearest food
    # 1. sum of distance to all foods
    # 2. food sum
    # 3. distance from ghostenemy
    # 4. distance Enemy cap()
    # 5. distance to friend
    # 6. sum cap
    # 7. dist to keybound
    # 8.dist to near bound
    # 9. score
    
    Wlist=[]
    w0=np.array([[-2,-0.2,100,   2.5,   -5,3,0,  -5,0,0]])#go to enemy
    w1=np.array([[0,0,0,   6,   0,0,0,  0,-5,100]])#escape
    w2=np.array([[0,0,100,   0,   -10,0,500,  0,0,100]])#cap
    w3=np.array([[0,-0.1,100,   101,   0,0,0,  0,-1,100]])#hide
    w4=np.array([[-1,0,100,   0,   0,0.5,0,  0,-0.5,100]])#next
    w5=np.array([[-1,0,100,   -5,   0,0.5,0,  0,0,-0.5,00]])#next, kill mode


    Wlist.append(w0)
    Wlist.append(w1)
    Wlist.append(w2)
    Wlist.append(w3)
    Wlist.append(w4)
    Wlist.append(w5)

    me=gameState.getAgentState(self.index)



    presentState=self.getfeature(gameState)

    ghost=-1
    isScared=False

    if(me.isPacman==False):
        bh=0
    else:

        if(me.numCarrying>=10):
            bh=1
        else:
            enemys=self.getOpponents(gameState)
            dist=np.Inf
            pos=gameState.getAgentPosition(self.index)
            for enemy in enemys:
                loc=gameState.getAgentPosition(enemy)
                if loc==None:
                    continue
                ag=gameState.getAgentState(enemy)
                if ag.scaredTimer>=2:
                    isScared=True
                    continue

                d=self.getMazeDistance(loc,pos)
                if d<dist:
                    dist=d
                    ghost=enemy
            if (dist<=4):
                bh=3
            else:
                caps=len(self.getCapsules(gameState))
                if(caps!=0):
                    if(self.index==self.capcommand(gameState)):
                        bh=2
                    else:
                        bh=4
                else:
                    bh=4
                # if (caps!=0 and (mainBrain.gocap==None or mainBrain.gocap==self.index)):
                #     bh=2
                #     mainBrain.gocap=self.index
                # else:
                #     bh=4
                # if (caps==0):
                #     mainBrain.gocap=None
                # if(isScared):
                #     bh=2
    if(bh==4):
        if(presentState[0]>presentState[8] and me.numCarrying>=1):
            bh=1
        if(ghost!=-1):
            if(gameState.getAgentState(ghost).scaredTimer>=5):
                bh=5
    if(bh==4):

        pos=gameState.getAgentPosition(self.index)

        foods = self.getFood(gameState).asList()
        target=None
        minfood=np.Inf

        for food in foods:
            dist = self.getMazeDistance(pos,food)
            #print(dist)
            if dist<minfood:
                minfood=dist
                target=food

        tgbd=None
        minbd=np.Inf

        if self.red==True:

            bound=mainBrain.boundR
        else:
            bound=mainBrain.boundB

        for bd in bound:
            dist=self.getMazeDistance(target,bd)
            if dist<minbd:

                minbd=dist
                tgbd=bd

        # if(tdbd==None):
        #     self.debugDraw(target,(100,100,150),False)
        #     pause()

        # if(target==None):
        #     self.debugDraw(tdbd,(100,150,100),False)
        #     pause()

        totldist=self.getMazeDistance(target,pos)+self.getMazeDistance(target,tgbd)

        if(totldist>gameState.data.timeleft-1):
            bh=1

            







    



    if(bh==3):
        return self.absearch(gameState, ghost,self.index,False)



    if( len(self.getFood(gameState).asList())<=2 ):
        bh=1    
    


    #print("agent",self.index,"bh=: ",bh)







    W=Wlist[bh]

    S=np.dot(A,W.T).T[0]

    stopid=actions.index('Stop')
    S[stopid]=S[stopid]-0.2*abs(S[stopid])

    if(self.lastaction!=None):
        reverse=AT.reverseDirection(self.lastaction)
        if(reverse in actions):
            rid=actions.index(reverse)
            S[rid]=S[rid]-0.1*abs(S[rid])


    #print(actions)

    #print(S)

    maxValue=max(S)


    bestActions = [a for a, v in zip(actions, S) if v == maxValue]


    #pause()
    #print("HERE THEY ARE", actions)
    ## IDea for super basic tree: get legal actions, check which is closest to food, choose this.

    '''
    You should change this in your own agent.
    '''

    #util.pause()
    return random.choice(bestActions)

  # def getfeatureD(self,succState,point):
  #   score=np.zeros(6)
    #0. distance to pac1
    #1.distance to pac 2
    #2.cap dist
    #3.sum of dist to food
    #4.dist to bounds
    #5. dist to point




  def getDefensivefeature(self, succState):
    # 0. Distance to own food
    # 1. Food sum
    # 2. Distance to enemy Pacman
    # 3. Distance to our capsule if it's still present
    # 4. dist to bound
    # 5. score
    # 6. Number of enemy Ghosts
    # 7. Get to patrol point
    # 8. Avoid Pacmanization
    score = np.zeros(9)
    idx=self.index
    pos=succState.getAgentPosition(idx)


    foods = self.getFoodYouAreDefending(succState).asList()

    ##############Distance to own food
    score[0]=np.Inf
    if(len(foods)==0): score[0]=0

    for food in foods:
        dist = self.getMazeDistance(pos,food)
        #print(dist)
        if dist<score[0]:
            score[0]=dist
        score[1]=dist+ score[1]

    ######### Food sum, sum of the food left on our game field
    score[1]=len(foods)

    ######### distance to enemy Pacman
    son=succState.getAgentDistances()
    enemys=self.getOpponents(succState)
    score[2]=100
    for enemy in enemys:
        E=succState.getAgentState(enemy)
        if not E.isPacman:
            #print "this guy is a GHOST"
            # dist=son[enemy]
            # if(score[2]>dist**2): # We're defending so we want to get closer!
            #     score[2]=dist**2

            continue
        loc=succState.getAgentPosition(enemy)

        st= succState.getAgentState(enemy)

        if (loc!=None):
            dist=self.getMazeDistance(pos,loc)

            #print dist, "###########################################################je topp"
        else:
            dist=son[enemy]

        if(score[2]>dist**2): # We're defending so we want to get closer!
            score[2]=dist**2
    ########## Distance to our capsule

    caps_pos = self.getCapsulesYouAreDefending(succState)
    if len(caps_pos)!=0:
        dist = self.getMazeDistance(pos,caps_pos[0])
        score[3] = dist
    else: 
        score[3] = 0

    ########## Dist to bound
    score[4]=np.Inf
    if self.red==True:
        bound=mainBrain.boundR
    else:
        bound=mainBrain.boundB
    for bd in bound:
        dist=self.getMazeDistance(pos,bd)
        if dist<score[4]:
            score[4]=dist

    ########### Score
    score[5]=self.getScore(succState)

    ########## Get # of enemy ghosts, if we eat pacman he becomes a ghost        
    enemys=self.getOpponents(succState)
    #print enemys, "forsvinner en har"
    for enemy in enemys:
        E=succState.getAgentState(enemy)
        if not E.isPacman:
            #print "sduahsduahsdua"
            score[6] += 1

    ########### Get to patrol point
    if self.index == mainBrain.agent1:
        dist = self.getMazeDistance(pos,mainBrain.patrol_aim1)
        score[7] = dist
    else:
        dist = self.getMazeDistance(pos,mainBrain.patrol_aim2)
        score[7] = dist
    #else:
    #    score[7] = 0

    ##### AVoid becoming Pacman
    st= succState.getAgentState(enemy)
    if st.isPacman:
        score[8] = 1
    else:
        score[8] = 0


    return score



  def getfeature(self, succState):
    score=np.zeros(10)
    # 0. distance to nearest food
    # 1. sum of distance to all foods
    # 2. food sum
    # 3. distance from ghostenemy
    # 4. distance Enemy cap()
    # 5. distance to friend
    # 6. sum cap
    # 7. dist to keybound
    # 8.dist to near bound
    # 9. score
    idx=self.index
    pos=succState.getAgentPosition(idx)

    score[9]=self.getScore(succState)
    #if(self.red!=False): score[8]=-score[8]


    
    foods = self.getFood(succState).asList()

    ##############foods
    score[0]=np.Inf
    if(len(foods)==0): score[0]=0

    for food in foods:
        dist = self.getMazeDistance(pos,food)
        #print(dist)
        if dist<score[0]:
            score[0]=dist
        score[1]=dist+ score[1]

    ############caps

    caps=self.getCapsules(succState)
    if(len(caps)>0):
        score[4]=np.Inf
        for cap in caps:
            dist = self.getMazeDistance(pos,cap)
        if dist<score[4]:
            score[4]=dist

    score[6]=-len(caps)    

    Team=self.getTeam(succState)
    friend=-1
    for ag in Team:
        if ag!=self.index:
            friend=ag
    score[5]=self.getMazeDistance(pos,succState.getAgentPosition(friend))

    ##################enemy

    enemyloc=[]

    son=succState.getAgentDistances()
    enemys=self.getOpponents(succState)
    score[3]=25
    for enemy in enemys:
        E=succState.getAgentState(enemy)
        if E.isPacman==True:
            continue
        # if E.scaredTimer>=3:
        #     continue

        loc=succState.getAgentPosition(enemy)


        st= succState.getAgentState( enemy)

        if (loc!=None):
            dist=self.getMazeDistance(pos,loc)
            enemyloc.append(loc)
        else:
            dist=son[enemy]

        if(score[3]>dist**2):
            score[3]=dist**2



    score[2]=-len(foods)

    score[7]=np.Inf


    num=mainBrain.getnum(idx)
    bd=mainBrain.getgoal(num)


    if(len(enemyloc)!=0):
        for eloc in enemyloc:
            enemydist=self.getMazeDistance(eloc,bd)
            if(enemydist>5):
                score[7]=self.getMazeDistance(pos,bd)
                self.debugDraw(bd,(100,100,100),False)
            else:
                    if self.red==True:

                        bound=mainBrain.key_boundR
                    else:
                        bound=mainBrain.key_boundB

                    for bd in bound:
                        dist=self.getMazeDistance(pos,bd)
                        if dist<score[7]:
                            if(len(enemyloc)!=0):
                                for eloc in enemyloc:
                                    enemydist=self.getMazeDistance(eloc,bd)
                                    if(enemydist<=3):
                                        continue

                            score[7]=dist
    else:
        score[7]=self.getMazeDistance(pos,bd)
        self.debugDraw(bd,(100,100,100),False)

        

    score[8]=np.Inf

    if self.red==True:

        bound=mainBrain.boundR
    else:
        bound=mainBrain.boundB

    for bd in bound:
        dist=self.getMazeDistance(pos,bd)
        if dist<score[8]:
            # if(len(enemyloc)!=0):
            #     for eloc in enemyloc:
            #         enemydist=self.getMazeDistance(eloc,bd)
            #         if(enemydist<=3):
            #             continue

            score[8]=dist


    return score














