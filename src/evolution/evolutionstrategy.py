import copy
import numpy as np
from abc import ABC, abstractmethod
import random
class EvolutionStrategy(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def generate(self, selection):
        """
        This function should produce a new population given a selection of the previous population.
        :param selection: the selection of the previous population
        :return new_pop: the new population
        """
        pass


class GP(EvolutionStrategy):
    def __init__(self, params):
        super().__init__(params)

    def generate(self, selection):
        new_pop = []
        r = np.random.random()
        while len(new_pop) < self.params['pop_size']:
            if r < self.params['p_m']:
                ind1 = random.sample(selection, 1)[0]
                new_pop.append(ind1.mutate())
            elif r < self.params['p_m'] + self.params['p_c']:
                ind1, ind2 =  random.sample(selection, 2)
                new_pop.extend(ind1.crossover(ind2))
            else:
                ind1 =  random.sample(selection, 1)[0]
                new_pop.append(ind1)
        return new_pop[:self.params['pop_size']]


class EA(EvolutionStrategy):
    def __init__(self, params):
        super().__init__(params)

    def generate(self, selection):
        new_pop = []

        # Crossover
        while len(new_pop) < self.params['pop_size']:
            if rand() < self.params['p_c']:
                ind1, ind2 = choice(selection, 2, replace=False)
                new_pop.extend(copy.deepcopy(ind1).crossover(copy.deepcopy(ind2)))
            else:
                new_pop.append(choice(selection, 1))

        # Mutation
        new_pop = [copy.deepcopy(c).mutate() if rand() < self.params['p_m'] else c for c in
                   new_pop[:self.params['pop_size']]]

        return new_pop
