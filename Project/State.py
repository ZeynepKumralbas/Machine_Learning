import random
import operator


class State:

    def __init__(self, stateNo, size):

        # state parameters
        self.__validActions = ["L", "R", "U", "D"]
        self.__stateNo = stateNo
        self.__environmentSize = size
        self.__reward = 0
        self.__policies = {"mu": {}}
        self.__isGoal = False
        self.__isPunishment = False
        self.__QValues = {"L": 0.0, "R": 0.0, "U": 0.0, "D": 0.0}

        # generate initial action set
        self.generateActions("mu")

    @property
    def validActions(self):
        return self.__validActions

    @property
    def stateNo(self):
        return self.__stateNo

    @property
    def environmentSize(self):
        return self.__environmentSize

    @property
    def reward(self):
        return self.__reward

    @reward.setter
    def reward(self, reward):
        self.__reward = reward

    @property
    def policies(self):
        return self.__policies

    @property
    def isGoal(self):
        return self.__isGoal

    @isGoal.setter
    def isGoal(self, isGoal):
        if isGoal:
            self.__reward = 500
            self.__isGoal = True
        else:
            self.__reward = 0
            self.__isGoal = False

    @property
    def isPunishment(self):
        return self.__isPunishment

    @isPunishment.setter
    def isPunishment(self, isPunishment):
        if isPunishment:
            self.__reward = -500
            self.__isPunishment = True
        else:
            self.__reward = 0
            self.__isPunishment = False

    def getQValues(self, action):
        return self.__QValues[action]

    def setQValues(self, action, QValue):
        self.__QValues[action] = QValue

    def generateActions(self, policy="mu"):
        if policy in self.__policies:
            if self.__stateNo % self.__environmentSize[1] == 0:  # column 0
                self.__validActions.remove("L")
                self.__QValues.pop("L")
            if self.__stateNo % self.__environmentSize[1] == (self.__environmentSize[1]-1):  # last column
                self.__validActions.remove("R")
                self.__QValues.pop("R")
            if self.__stateNo in range(0, self.__environmentSize[1]):  # row 0
                self.__validActions.remove("U")
                self.__QValues.pop("U")
            if self.__stateNo in range((self.__environmentSize[0] - 1) * self.__environmentSize[1],
                                     self.__environmentSize[0] * self.__environmentSize[1]):  # bottom row
                self.__validActions.remove("D")
                self.__QValues.pop("D")

    def choseAction(self, epsilon):
        rand = random.random()  # random number between [0,1)
        if rand <= epsilon:
            chosenAction = self.explorative()
        else:
            chosenAction = self.exploitative()
        return chosenAction

    def explorative(self):
        return random.choice(list(self.__QValues))

    def exploitative(self):
        return max(self.__QValues.items(), key=operator.itemgetter(1))[0]

    def getMaxQValue(self):
        return self.__QValues[max(self.__QValues.items(), key=operator.itemgetter(1))[0]]
