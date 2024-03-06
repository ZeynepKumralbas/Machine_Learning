from statistics import variance
import numpy as np

from Environment import Environment


class RLTask:

    def __init__(self, size):

        self.__environmentSize = size
        self.__environment = Environment(self.__environmentSize)
        # parameters
        self.__alpha = 0.1
        self.__gamma = 0.90
        self.__maxEpsilon = 0.7
        self.__minEpsilon = 0.00001
        self.__epsilonDecrease = 0.001
        self.__currentEpsilon = self.__maxEpsilon
        self.__goalReward = 50
        self.__actionDictionary = {"LEFT": "1", 'UP': "2", "RIGHT": "3", "DOWN": "4"}
        self.__meanValues = []
        self.__numOfEpisodes = 0

        # convergence parameters
        self.__convergenceInterval = 10

        self.__lastNSteps = np.zeros(self.__convergenceInterval)  # create array for converge interval

    @property
    def environmentSize(self):
        return self.__environmentSize

    @property
    def environment(self):
        return self.__environment

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        self.__alpha = alpha

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma):
        self.__gamma = gamma

    @property
    def maxEpsilon(self):
        return self.__maxEpsilon

    @maxEpsilon.setter
    def maxEpsilon(self, maxEpsilon):
        self.__maxEpsilon = maxEpsilon

    @property
    def minEpsilon(self):
        return self.__minEpsilon

    @minEpsilon.setter
    def minEpsilon(self, minEpsilon):
        self.__minEpsilon = minEpsilon

    @property
    def epsilonDecrease(self):
        return self.__epsilonDecrease

    @epsilonDecrease.setter
    def epsilonDecrease(self, epsilonDecrease):
        self.__epsilonDecrease = epsilonDecrease

    @property
    def goalReward(self):
        return self.__goalReward

    @goalReward.setter
    def goalReward(self, goalReward):
        self.__goalReward = goalReward

    @property
    def actionDictionary(self):
        return self.__actionDictionary

    @property
    def numOfEpisodes(self):
        return self.__numOfEpisodes

    def setEpsilonDecrease(self, epsilonDecrease):
        self.__epsilonDecrease = epsilonDecrease

    def setGoalReward(self, goalReward):
        self.__goalReward = goalReward

    def setConvergenceInterval(self, convergenceInterval):
        self.__convergenceInterval = int(convergenceInterval)

    @property
    def meanValues(self):
        return self.__meanValues

    def applyQLearning(self):

        stateList = self.__environment.stateList

        numOfActionsInEpisode = totalNumOfActions = 0
        self.__currentEpsilon = self.__maxEpsilon

        var = 1

        # episode loop
        while var != 0 and self.__currentEpsilon > self.__minEpsilon:

            agentCurrentCoordinates = self.__environment.startCoordinates  # initial coordinates of the agent
            self.__numOfEpisodes += 1
            numOfActionsInEpisode = 0

            # action loop
            currentState = stateList[agentCurrentCoordinates]

            # 1 episode finishes when the agent reaches the goal state or the punishment state
            while not (currentState.isGoal or currentState.isPunishment):
                currentState = stateList[agentCurrentCoordinates]

                chosenAction = currentState.choseAction(self.__currentEpsilon)  # chose an action based on epsilon

                # get the destination coordinates
                destinationCoordinates = self.__environment.getDestState(agentCurrentCoordinates, chosenAction)

                destinationState = stateList[destinationCoordinates]  # get the destination state

                # update the QValue
                currentState.setQValues(chosenAction, currentState.getQValues(chosenAction) + self.__alpha * (
                        destinationState.reward + self.__gamma * destinationState.getMaxQValue() -
                        currentState.getQValues(chosenAction)))

                agentCurrentCoordinates = destinationCoordinates
                numOfActionsInEpisode += 1

                currentState = destinationState

            # 1 episode finishes

            print("Qlearning currentEpsilon:", self.__currentEpsilon)

            # convergence calculation

            # if the lastNSteps array is full, shift the array and get the new value
            if self.__numOfEpisodes > self.__convergenceInterval:
                self.__lastNSteps[0:self.__convergenceInterval - 1] = self.__lastNSteps[1:]
                self.__lastNSteps[self.__convergenceInterval - 1] = numOfActionsInEpisode

            # if the lastNSteps array is not full, get the values
            else:
                self.__lastNSteps[self.__numOfEpisodes - 1] = numOfActionsInEpisode

            totalNumOfActions += numOfActionsInEpisode

            # mean values of the number of actions in episodes calculation

            if len(self.__meanValues) == 0:
                self.__meanValues.append(numOfActionsInEpisode / self.__numOfEpisodes)
            else:
                lastMean = self.__meanValues[-1]
                self.__meanValues.append(
                    (lastMean * (self.__numOfEpisodes - 1) + numOfActionsInEpisode) / self.__numOfEpisodes)

            self.__currentEpsilon -= self.__epsilonDecrease
            var = variance(self.__lastNSteps)
            print("Qlearning variance:", var)


        # print Qvalues
        self.printQValues(stateList, "Qlearning")

    def applySARSA(self):

        stateList = self.__environment.stateList

        numOfActionsInEpisode = totalNumOfActions = 0
        self.__currentEpsilon = self.__maxEpsilon

        var = 1

        # episode loop
        while var != 0 and self.__currentEpsilon > self.__minEpsilon:

            agentCurrentCoordinates = self.__environment.startCoordinates  # initial coordinates of the agent
            self.__numOfEpisodes += 1
            numOfActionsInEpisode = 0

            # action loop
            currentState = stateList[agentCurrentCoordinates]

            # 1 episode finishes when the agent reaches the goal state or the punishment state
            while not (currentState.isGoal or currentState.isPunishment):
                currentState = stateList[agentCurrentCoordinates]

                chosenAction = currentState.choseAction(self.__currentEpsilon)  # chose an action based on epsilon
                    
                # get the destination coordinates
                destinationCoordinates = self.__environment.getDestState(agentCurrentCoordinates, chosenAction)

                destinationState = stateList[destinationCoordinates]  # get the destination state
                
                chosenAction_t1 = destinationState.choseAction(self.__currentEpsilon)  # get at+1

                # update the QValue
                currentState.setQValues(chosenAction, currentState.getQValues(chosenAction) + self.__alpha * (
                        destinationState.reward + self.__gamma * destinationState.getQValues(chosenAction_t1) -
                        currentState.getQValues(chosenAction)))

                agentCurrentCoordinates = destinationCoordinates
                numOfActionsInEpisode += 1

                currentState = destinationState

            # 1 episode finishes

            print("SARSA currentEpsilon:", self.__currentEpsilon)

            # convergence calculation

            # if the lastNSteps array is full, shift the array and get the new value
            if self.__numOfEpisodes > self.__convergenceInterval:
                self.__lastNSteps[0:self.__convergenceInterval - 1] = self.__lastNSteps[1:]
                self.__lastNSteps[self.__convergenceInterval - 1] = numOfActionsInEpisode

            # if the lastNSteps array is not full, get the values
            else:
                self.__lastNSteps[self.__numOfEpisodes - 1] = numOfActionsInEpisode

            totalNumOfActions += numOfActionsInEpisode

            # mean values of the number of actions in episodes calculation

            if len(self.__meanValues) == 0:
                self.__meanValues.append(numOfActionsInEpisode / self.__numOfEpisodes)
            else:
                lastMean = self.__meanValues[-1]
                self.__meanValues.append(
                    (lastMean * (self.__numOfEpisodes - 1) + numOfActionsInEpisode) / self.__numOfEpisodes)

            self.__currentEpsilon -= self.__epsilonDecrease
            var = variance(self.__lastNSteps)
            print("SARSA variance:", var)

        # print Qvalues
        self.printQValues(stateList, "SARSA")

    def printQValues(self, stateList, method):
        print(method)
        for i in range(0, self.__environmentSize[0]):
            for j in range(0, self.__environmentSize[1]):
                print(i, j)
                if 'L' in stateList[i][j].validActions:
                    print("L:", stateList[i][j].getQValues("L"), " ")
                if 'R' in stateList[i][j].validActions:
                    print("R:", stateList[i][j].getQValues("R"), " ")
                if 'U' in stateList[i][j].validActions:
                    print("U:", stateList[i][j].getQValues("U"), " ")
                if 'D' in stateList[i][j].validActions:
                    print("D:", stateList[i][j].getQValues("D"), " ")
