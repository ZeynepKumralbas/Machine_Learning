import numpy as np
from State import State


class Environment:

    def __init__(self, size):

        self.__environmentSize = size
        self.__stateList = np.empty(shape=self.__environmentSize, dtype=State)
        self.__goalReward = 500
        self.__punishment = -500
        self.__startCoordinates = (int(self.__environmentSize[0] / 2), int(self.__environmentSize[1] / 2))
        self.__goalCoordinates = (self.__environmentSize[0] - 1, self.__environmentSize[1] - 1)
        self.__punishmentCoordinates = (0, 0)

        stateNo = 0
        # generate states and put into stateList
        for row in range(0, self.__environmentSize[0]):
            for col in range(0, self.__environmentSize[1]):
                self.__stateList[row][col] = State(stateNo, self.__environmentSize)
                stateNo += 1

        # goal state
        self.__stateList[self.__environmentSize[0] - 1][self.__environmentSize[1] - 1].isGoal = True

        # punishment state
        self.__stateList[0][0].isPunishment = True

    @property
    def environmentSize(self):
        return self.__environmentSize

    @property
    def stateList(self):
        return self.__stateList

    @property
    def goalReward(self):
        return self.goalReward

    @goalReward.setter
    def goalReward(self, goalReward):
        self.__goalReward = goalReward

    @property
    def punishment(self):
        return self.__punishment

    @punishment.setter
    def punishment(self, punishment):
        self.__punishment = punishment

    @property
    def startCoordinates(self):
        return self.__startCoordinates

    @property
    def goalCooridinates(self):
        return self.__goalCoordinates

    @property
    def punishmentCoordinates(self):
        return self.__punishmentCoordinates

    def getState(self, index):
        for row in range(0, self.__environmentSize[0]):
            for col in range(0, self.__environmentSize[1]):
                if self.__stateList[row][col].stateNo == index:
                #    print("index:", index, "row:", row, "col:", col, "ee:", (self.__stateList[row][col]).stateNo)
                    return self.__stateList[row][col]

    def getDestState(self, coordinate, action):
        if action == 'L':
            destCoordinate = coordinate[0], coordinate[1] - 1
        elif action == 'R':
            destCoordinate = coordinate[0], coordinate[1] + 1
        elif action == 'U':
            destCoordinate = coordinate[0] - 1, coordinate[1]
        elif action == 'D':
            destCoordinate = coordinate[0] + 1, coordinate[1]

        return destCoordinate
