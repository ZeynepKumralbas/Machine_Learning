from RLTask import RLTask
from PS import PS
import numpy as np
import matplotlib.pyplot as plt

# =======Q-LEARNING VS PS======================================================================

RLTask_Qlearning = RLTask((7, 7))
RLTask_Qlearning.applyQLearning()

episodesForQlearning = np.arange(1, RLTask_Qlearning.numOfEpisodes+1)  # [1, numOfEpisodes+1)
plt.title("Q-Learning vs PS mean of number of steps in episodes vs number of episodes")
plt.xlabel("number of episodes")
plt.ylabel("mean of number of steps in episodes")
plt.plot(episodesForQlearning, RLTask_Qlearning.meanValues)
print("Qlearning total # of episodes:", RLTask_Qlearning.numOfEpisodes)

PS_task = PS((7, 7))
PS_task.applyQLearningPS()
episodesForPS = np.arange(1, PS_task.numOfEpisodes+1)  # [1, numOfEpisodes+1)
plt.plot(episodesForPS, PS_task.meanValues)
plt.legend(['Q-Learning', 'PS'], loc='upper right')
print("PS total # of episodes:", PS_task.numOfEpisodes)
plt.show()

# =======Q-LEARNING VS PS======================================================================



# =======Q-LEARNING VS SARSA======================================================================
RLTask_Qlearning = RLTask((5, 5))
RLTask_Qlearning.applyQLearning()

episodesForQlearning = np.arange(1, RLTask_Qlearning.numOfEpisodes+1)  # [1, numOfEpisodes+1)
plt.title("Q-Learning vs PS mean of number of steps in episodes vs number of episodes")
plt.xlabel("number of episodes")
plt.ylabel("mean of number of steps in episodes")
plt.plot(episodesForQlearning, RLTask_Qlearning.meanValues)
plt.show()

RLTask_SARSA = RLTask((5, 5))
RLTask_SARSA.applySARSA()
episodesForSARSA = np.arange(1, RLTask_SARSA.numOfEpisodes+1)  # [1, numOfEpisodes+1)
plt.title("SARSA mean of number of steps in episodes vs number of episodes")
plt.xlabel("number of episodes")
plt.ylabel("mean of number of steps in episodes")
plt.plot(episodesForSARSA, RLTask_SARSA.meanValues)
plt.show()

print("Qlearning total # of episodes:", RLTask_Qlearning.numOfEpisodes)
print("SARSA total # of episodes:", RLTask_SARSA.numOfEpisodes)


# =======Q-LEARNING VS SARSA======================================================================
