import heapq

from queue import PriorityQueue
pq = PriorityQueue()

class PQueue(object):

    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

        # for checking if the queue is empty

    def isEmpty(self):
        return len(self.queue) == 0  # []

        # for inserting an element in the queue

    def insert(self, theta, tuple):

        self.queue.append([theta, tuple])      

        # for popping an element based on Priority
    def delete(self):
        try:
            maxx = 0
            for i in range(len(self.queue)):
                if self.queue[i][0] > self.queue[maxx][0]:
                    maxx = i
            item = self.queue[maxx][1]
            del self.queue[maxx]
            return item
        except IndexError:
            print()
            exit()
'''
if __name__ == '__main__':
    myQueue = PQueue()
    myQueue.insert(5, (2, 5))
    myQueue.insert(3, (1, 3))
    a = myQueue.delete()
    print(a)
 #   myQueue.insert(14)
  #  myQueue.insert(7)
'''