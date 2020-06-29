# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
from collections import defaultdict
import pickle

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvalues = util.Counter()
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #print(self.qvalues[(state, action)])
        return self.qvalues[(state, action)]

        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #현재 state에서의 action들 반환
        actions = self.getLegalActions(state)
        #기본값을 설정해둔다.
        value = -1000
        #각 action에 대해서 Qvalue값을 받은 후, 그것이 value값보다 크면 value값을 해당값으로 바꿔준다.
        for a in actions:
            if self.getQValue(state, a) >= value:
                value = self.getQValue(state, a)
        #최종 value가 처음 지정해둔 값과 같다면, 즉 변하지 않았다면 0.0을 return 한다.
        #print(value)
        if value == -1000:
            return 0.0
        else:
            return value
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # 현재 state에서의 action들 반환
        actions = self.getLegalActions(state)
        # max를 찾기 위한 시작값.
        max = -1000
        # 이제 action을 찾아야하는데, 이 때 같은 value값이 두개 이상 나올 경우, 랜덤으로 골라야하는 것에 유의해야한다.
        for action in actions:
            #print(self.getQValue(state,action))
            if self.getQValue(state, action) > max:
                max = self.getQValue(state, action)
                #그 때의 action을 my_action으로 둔다.
                my_action = action
        #액션을 저장할 list이다.
        action_list = []
        #max값과 같은 action을 list에 저장한다.
        for action in actions:
            if self.getQValue(state, action) == max:
                action_list.append(action)
        #list에 저장된 action중 랜덤으로 하나를 고른다.
        real_action = random.choice(action_list)

        if max == -1000:
            return None
        else:
            return real_action

        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # action을 선택한다.
        Actions = self.getLegalActions(state)
        # epsilon에 따른 flipcoin을 사용하는데, epsilon의 확률로 랜덤한 action을 고르고,
        # 1-epsilon의 확률로 computeActionFromQValues의 action을 return한다.
        if util.flipCoin(self.epsilon):
            return random.choice(Actions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qvalues[(state, action)] = (1 - self.alpha) * (self.getQValue(state, action)) + (self.alpha) * (
                    reward + self.discount * self.computeValueFromQValues(nextState))

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.cum_weights = defaultdict(list)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Q는 w와 f의 곱으로 표현된다.
        weight = self.getWeights()
        F = self.featExtractor.getFeatures(state, action)

        return weight * F

        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Q값을 반환.
        Q = self.getQValue(state, action)
        #Q값은 (f*w) 의 합으로 구성된다.
        F = self.featExtractor.getFeatures(state, action)
        #다음스텝의 max Q(s',a')의 값을 계산한다.
        Max = self.computeValueFromQValues(nextState)

        #각 feature에 대해서 weight를 합한다.
        for f in F:
            #difference는 아래의 식과 같다.
            difference = reward + (self.discount*Max) - Q
            self.weights[f] = self.weights[f] + self.alpha * difference * F[f]

        #util.raiseNotDefined()

        "***  DO NOT DELETE BELOW ***"
        self.write()

    def write(self):
        """
          DO NOT DELETE
        """
        for i in ["bias", "#-of-ghosts-1-step-away", "eats-food", "closest-food"]:
            self.cum_weights[i].append(self.weights[i])

    def save(self):
        """
          DO NOT DELETE
        """
        with open('./cmu_weights.pkl','wb') as f:
            pickle.dump(self.cum_weights,f)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


            "***  DO NOT DELETE BELOW ***"
            self.save()