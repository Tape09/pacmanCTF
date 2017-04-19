import math
import itertools
import copy
import time
import sys

class MinimaxAgent:

	def __init__(self,color,values):
		self.color = color
		self.values = values
		self.nodes = 0
		self.totalTime = 0

	def move(self,state):
		#returns optimal move for this state
		return self.Minimax(state)

	def result(self,move,state):
		self.nodes += 1
		#state representation (array,depth)
		#[0] prefix mean we're reffering to the board and not the depth (same amount of code)
		newState = [copy.deepcopy(state[0]),state[1]+1] #initialize new state to whatever the old state was before adding new move
		color = self.color #get player from class
		newState[0][move[0]][move[1]] = color
		if color in (newState[0][move[0]+1][move[1]],newState[0][move[0]-1][move[1]],newState[0][move[0]][move[1]+1],newState[0][move[0]][move[1]-1]):
				#if adjacent to any of the same color
				for (x,y) in ((move[0]+1,move[1]),(move[0]-1,move[1]),(move[0],move[1]+1),(move[0],move[1] -1)):
					if newState[0][x][y] == self.next(color): #oponents piece, replace it
						newState[0][x][y] = color #flip the color
		return newState


	def Min(self,state):
		#returns the utility value of a move by searching through tree
		if self.terminalTest(state): return self.util(state)
		if self.depthTest(state): return self.evaluation(state) #once we hit the bottom
		v = float('inf')
		for i in range(1,7):
			for j in range(1,7):
				if (state[0][i][j] != 'blue') and (state[0][i][j] != 'green'):
					v = min(v,self.Max(self.result((i,j),state))) #min of max
		return v

	def Max(self,state):
		if self.terminalTest(state): return self.util(state)
		if self.depthTest(state): return self.evaluation(state)
		v = -float('inf')
		for i in range(1,7):
			for j in range(1,7):
				if (state[0][i][j] != 'green') and (state[0][i][j] != 'blue'):
					v = max(v,self.Min(self.result((i,j),state))) #max of min
		return v


	def Minimax(self,state):
		#returns optimal move accprding to minimax given the current state of the game 
		util = -float('inf') #initial utility of 0, we will loop through and update this
		newState = [state,0]
		for i in range(1,7):
			for j in range(1,7):
				if (newState[0][i][j] != 'blue') and (newState[0][i][j] != 'green'):
					#loop through all possible legal moves, checking their utilites
					curr = self.Min(self.result((i,j),newState))
					if curr > util: 
						util = curr
						move = (i,j) #update best move
		return move


	def depthTest(self,state):
		#depth check of some sort (how should I do this? a depth limited search?)
		if state[1] >= 3:
			return True
		return False

	def terminalTest(self,state):
		#check to see if the game is over (all pieces have been placed
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] != 'green' and state[0][x][y] != 'blue':
					return False #still an empty spot left, game is not over
		return True


	def evaluation(self,state):
		#returns value of current player's position so far (our heuristic)
		blueVal = 0
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] == 'blue':
					blueVal += self.values[x][y]

		greenVal = 0
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] == 'green':
					greenVal += self.values[x][y]
		######this might be it
		if self.color == 'blue': return blueVal - greenVal
		if self.color == 'green': return greenVal - blueVal
		#return blueVal - greenVal
		#if blueVal > greenVal: return 50 
		#if blueVal < greenVal: return -50

	def util(self,state):
		#returns value of current player's position so far (our heuristic)
		blueVal = 0
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] == 'blue':
					blueVal += self.values[x][y]

		greenVal = 0
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] == 'green':
					greenVal += self.values[x][y]
		######this might be it
		if self.color == 'blue': return blueVal - greenVal
		if self.color == 'green': return greenVal - blueVal
		#return blueVal-greenVal
		#if blueVal > greenVal: return 50 
		#if blueVal < greenVal: return -50
		

	def next(self,color):
		if color == 'blue':
			return 'green'
		if color == 'green':
			return 'blue'

	def addTime(self,time):
		self.totalTime += time
		return 0

class AlphaBetaAgent:

	#new state: [pieces,depth,alpha,beta] (chnage as little as possible)
	def __init__(self,color,values):
		self.color = color
		self.values = values
		self.nodes = 0
		self.totalTime = 0

	def move(self,state):
		#returns optimal move for this state
		return self.AlphaBeta(state)

	def result(self,move,state):
		self.nodes += 1
		#state representation (array,depth)
		#[0] prefix mean we're reffering to the board and not the depth (same amount of code)
		newState = [copy.deepcopy(state[0]),state[1]+1] #initialize new state to whatever the old state was before adding new move
		color = self.color #get player from class
		newState[0][move[0]][move[1]] = color
		if color in (newState[0][move[0]+1][move[1]],newState[0][move[0]-1][move[1]],newState[0][move[0]][move[1]+1],newState[0][move[0]][move[1]-1]):
				#if adjacent to any of the same color
				for (x,y) in ((move[0]+1,move[1]),(move[0]-1,move[1]),(move[0],move[1]+1),(move[0],move[1] -1)):
					if newState[0][x][y] == self.next(color): #oponents piece, replace it
						newState[0][x][y] = color #flip the color
		return newState


	def Max(self,state,alpha,beta):
		if self.terminalTest(state): return self.util(state)
		if self.depthTest(state): return self.evaluation(state)
		#v = -float('inf')
		#newAlpha = copy.deepcopy(alpha)
		#newBeta = copy.deepcopy(beta)
		v = -float('inf')
		for i in range(1,7):
			for j in range(1,7):
				if (state[0][i][j] != 'green') and (state[0][i][j] != 'blue'):
					v = max(v,self.Min(self.result((i,j),state),alpha,beta)) #max of min
					if v >= beta: return v #utility is now at least larger than the beta value, so no other nodes need be checked
					alpha = max(alpha,v) #prune the tree
		return v

	def Min(self,state,alpha,beta):
		#returns the utility value of a move by searching through tree
		if self.terminalTest(state): return self.util(state)
		if self.depthTest(state): return self.evaluation(state) #once we hit the bottom
		#v = float('inf')
		#newAlpha = copy.deepcopy(alpha)
		#newBeta = copy.deepcopy(beta)
		v = float('inf')
		for i in range(1,7):
			for j in range(1,7):
				if (state[0][i][j] != 'blue') and (state[0][i][j] != 'green'):
					v = min(v,self.Max(self.result((i,j),state),alpha,beta)) #min of max
					if v <= alpha: return v
					beta = min(beta,v) 
		return v


	def AlphaBeta(self,state):
		#returns optimal move accprding to minimax given the current state of the game 
		util = -float('inf') #initial utility of 0, we will loop through and update this
		#newState = [state,0,-float('inf'),float('inf')]
		newState = [state,0]
		for i in range(1,7):
			for j in range(1,7):
				if (newState[0][i][j] != 'blue') and (newState[0][i][j] != 'green'):
					#loop through all possible legal moves, checking their utilites
					curr =  self.Min(self.result((i,j),newState),-float('inf'),float('inf')) #utility of given node
					if curr > util: #update if utility is larger
						util = curr
						move = (i,j) #update best move
		return move

	'''def AlphaBeta(self,state):
		util = float('inf') #initial utility of 0, we will loop through and update this
		#returns optimal move accprding to minimax given the current state of the game 
		#returns optimal move accprding to minimax given the current state of the game 
		#newState = [state,0,-float('inf'),float('inf')]
		#returns optimal move accprding to minimax given the current state of the game 
		#returns optimal move accprding to minimax given the current state of the game 
		newState = [state,0]
		v = self.Max(newState,-float('inf'),float('inf'))
		#returns optimal move accprding to minimax given the current state of the game 
		newState = [state,0]
		v = self.Max(newState,-float('inf'),float('inf'))
		newState = [state,0]
		v = self.Max(newState,-float('inf'),float('inf'))
		for i in range(1,7):
			for j in range(1,7):
				if (newState[0][i][j] != 'blue') and (newState[0][i][j] != 'green'):
					#loop through all possible legal moves, checking their utilites
					curr = self.evaluation(self.result((i,j),newState))
					if curr == v:
						return (i,j)'''
					

	def maximum(self,key1,key2):
		if key2 < key1:
			return key1
		else: return key2


	def depthTest(self,state):
		#depth check of some sort (how should I do this? a depth limited search?)
		if state[1] >= 3:
			return True
		return False

	def terminalTest(self,state):
		#check to see if the game is over (all pieces have been placed
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] != 'green' and state[0][x][y] != 'blue':
					return False #still an empty spot left, game is not over
		return True


	def evaluation(self,state):
		#returns value of current player's position so far (our heuristic)
		blueVal = 0
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] == 'blue':
					blueVal += self.values[x][y]

		greenVal = 0
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] == 'green':
					greenVal += self.values[x][y]
		######this might be it
		if self.color == 'blue': return blueVal - greenVal
		if self.color == 'green': return greenVal - blueVal
		#if blueVal > greenVal: return 50 
		#if blueVal < greenVal: return -50

	def util(self,state):
		#returns the utility of a board position for first (max) player: -1,0 or 1
		blueVal = 0
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] == 'blue':
					blueVal += self.values[x][y]

		greenVal = 0
		for x in range(1,7):
			for y in range(1,7):
				if state[0][x][y] == 'green':
					greenVal += self.values[x][y]
		######this might be it
		if self.color == 'blue': return blueVal - greenVal
		if self.color == 'green': return greenVal - blueVal
		#if blueVal > greenVal: return 50 
		#if blueVal < greenVal: return -50

	def next(self,color):
		if color == 'blue':
			return 'green'
		if color == 'green':
			return 'blue'

	def addTime(self,time):
		self.totalTime += time
		return 0


class Game:

	def __init__(self,values):
		self.values = [[0 for x in xrange(8)] for x in xrange(8)] 
		self.pieces = [[0 for x in xrange(8)] for x in xrange(8)] 
		self.startingPlayer = 'blue'
		self.currentPlayer = self.startingPlayer
		#initial values, we will create a game state object for the tree itself

		for i in range(1,7):
			for j in range(1,7):
				self.pieces[i][j] = 'n' #initialize board to be empty

		self.values = values #initialize values to board values


	def result(self,move):
		#result of moving on the entire board (not for search purposes)
		#initialize new state to whatever the old state was before adding new move
		color = self.currentPlayer #get player from class
		self.pieces[move[0]][move[1]] = color
		
		if color in (self.pieces[move[0]+1][move[1]],self.pieces[move[0]-1][move[1]],self.pieces[move[0]][move[1]+1],self.pieces[move[0]][move[1]-1]):
			for (x,y) in ((move[0]+1,move[1]),(move[0]-1,move[1]),(move[0],move[1]+1),(move[0],move[1] -1)):
				if self.pieces[x][y] == self.next(color): #oponents piece, replace it
					self.pieces[x][y] = color #flip the color
		return 0

	def score(self,color):
		#returns the score of a give player
		score = 0
		for x in range(1,7):
			for y in range(1,7):
				if self.pieces[x][y] == color:
					score += self.values[x][y]
		return score

	def next(self,color):
		if color == 'blue':
			return 'green'
		else: return 'blue'

	def isOver(self):
		#a terminal test for the entire game, not just minimax
		for x in range(1,7):
			for y in range(1,7):
				if self.pieces[x][y] != 'green' and self.pieces[x][y] != 'blue':
					return False #still an empty spot left, game is not over
		return True

	def getPieces(self):
		return self.pieces

	def switch(self):
		if self.currentPlayer == 'blue':
			self.currentPlayer = 'green'
			return
		else: self.currentPlayer = 'blue'
		return

	def printBoard(self):
		print self.pieces[1][1:7],'\n',self.pieces[2][1:7],'\n',self.pieces[3][1:7],'\n',self.pieces[4][1:7],'\n',self.pieces[5][1:7],'\n',self.pieces[6][1:7],'\n'


def main():

	almond = [[0 for x in xrange(8)] for x in xrange(8)] 
	
	#Almond Joy
	for i in range(1,7):
		for j in range(1,7):
			almond[i][j] = 1

	#Ayds
	ayds =[[0,0,0,0,0,0,0,0],
	[0,99, 1, 99, 1, 99, 1,0],
	[0, 1,	99,	1,	99,	1,	99,0],
	[0,99, 1,	99,	1,	99,	1,0],
	[0,1, 99, 1, 99, 1, 99,0],
	[0,99, 1, 99, 1, 99, 1,0],
	[0,1, 99, 1, 99, 1, 99,0],
	[0,0,0,0,0,0,0,0]]

	#Bit-O-Honey

	honey = [[0 for x in xrange(8)] for x in xrange(8)]
	for i in range(1,7):
		for j in range(1,7):
			honey[i][j] = math.pow(2,i-1)

	#Mounds

	mounds =[[0,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,1,0],
	[0,1,3,4,4,3,1,0],
	[0,1,4,2,2,4,1,0],
	[0,1,4,2,2,4,1,0],
	[0,1,3,4,4,3,1,0],
	[0,1,1,1,1,1,1,0]]

	#reeces
	reeces =[[0,0,0,0,0,0,0,0],
	[0,66,76,28,66,11,9,0],
	[0,31,39,50,8,33,14,0],
	[0,80,76,39,59,2,48,0],
	[0,50,73,43,3,13,3,0],
	[0,99,45,72,87,49,4,0],
	[0,80,63,92,28,61,53,0],
	[0,0,0,0,0,0,0,0]]

	game = Game(reeces)

	blue = MinimaxAgent('blue',reeces)
	green = AlphaBetaAgent('green',reeces)
	#Blue ALWAYS expands more than green
	#moves are identical regardless of board
	########################
	'''while not game.isOver():
		if game.currentPlayer == 'blue':
			board = copy.deepcopy(game.getPieces())
			start = time.clock()
			move = blue.move(board)
			stop = time.clock()
			elapsed = stop - start
			blue.addTime(elapsed)
			#print game.printBoard()
			print 'Blue: ',move
			game.result(move)
		if game.currentPlayer == 'green':
			board = copy.deepcopy(game.getPieces())
			start = time.clock()
			move = green.move(board)
			stop = time.clock()
			elapsed = stop - start
			green.addTime(elapsed)
			#print game.printBoard()
			print 'Green: ',move
			game.result(move)
		game.switch()'''

	##########################
	while not game.isOver():
		game.printBoard()
		inputvar = input("User move x: ")
		i = inputvar
		inputvar = input("User move y: ")
		j = inputvar
		game.result((i,j)) #create new game, start taking input from terminal
		board = copy.deepcopy(game.getPieces)
		move = blue.move(board)
		game.result(move)
	

	'''print 'Blue score: ',game.score('blue')
	print 'Green score: ',game.score('green')
	game.printBoard()
	print 'Nodes expanded by Blue: ',blue.nodes
	print 'Average per move: ',blue.nodes/18
	print 'Average time: ',blue.totalTime/18
	print 'Nodes expanded by Green: ',green.nodes
	print 'Average per move: ',green.nodes/18
	print 'Average time: ',green.totalTime/18'''

	

if  __name__ =='__main__':main()