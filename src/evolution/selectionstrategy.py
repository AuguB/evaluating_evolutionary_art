import numpy as np
from numpy.random import choice


class SelectionStrategy:
    def __init__(self, params):
        self.params = params

    def select(self, population, fitnesses):
        """
        :param population: a list of representations
        :param fitnesses: the fitnesses of each representation such that fitness(population[i]) = fitnesses[i]
        :return selection: the selection of the population from which to evolve a new population
        """
        pass


class RouletteWheelSelection(SelectionStrategy):
    def __init__(self, params):
        super().__init__(params)

    def select(self, population, fitnesses):
        fitness_np = np.array(fitnesses)
        fitness_pd = fitness_np / np.sum(fitness_np)
        selection = choice(population, size=self.params['selection_pool_size'], replace=True, p=fitness_pd)
        return list(selection)


class TournamentSelection(SelectionStrategy):
    def __init__(self, params):
        super().__init__(params)

    def get_fitnessfinder(self, fitnesses):
        def fitnessfinder(i):
            return fitnesses[i]

        return fitnessfinder

    def select(self, population, fitnesses):
        fitnessfinder = np.vectorize(self.get_fitnessfinder(fitnesses))
        indices = np.random.randint(low=0, high=len(population),
                                    size=(self.params['selection_pool_size'], self.params['K']))
        fitness_matrix = fitnessfinder(indices)
        argmax_fitness_matrix = np.argmax(fitness_matrix, axis=1)
        best_indices = indices[np.arange(len(argmax_fitness_matrix)), argmax_fitness_matrix]
        selection = [population[i] for i in best_indices]
        return selection
