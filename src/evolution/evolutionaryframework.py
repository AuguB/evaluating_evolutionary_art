import numpy as np

from utils.archiver import Archiver


class EvolutionaryFramework:
    def __init__(self, params, archiver: Archiver):
        self.params = params
        self.population = []
        self.fitnesses = []
        for i in range(self.params['population_size']):
            self.population.append(self.params['representation'](self.params['representation_params']))
        self.evaluationstrategy = self.params['evaluationstrategy'](self.params['evaluation_params'])
        self.selectionstrategy = self.params['selectionstrategy'](self.params['selection_params'])
        self.evolutionstrategy = self.params['evolutionstrategy'](self.params['evolution_params'])
        self.archiver = archiver

    def evolve(self):
        self.fitnesses = self.evaluationstrategy.evaluate(self.population)
        self.archive()
        selection = self.selectionstrategy.select(self.population, self.fitnesses)
        self.population = self.evolutionstrategy.generate(selection)

    def archive(self):
        self.archiver.archive(self.fitnesses,self.population)

    def get_output(self, dimx, dimy):
        if len(self.population[0].statistics) > 0:
            stats = np.array(list(map(lambda x: x.get_statistics(), self.population)))
            meanstats = np.mean(stats, axis=0)
            for i in range(len(meanstats)):
                print(f"mean {self.population[0].statistics[i]}: {meanstats[i]}")
        return list(map(lambda x: x.get_output(dimx, dimy), self.population))

    def set_fitness_buffer(self, fitnesses):
        self.evaluationstrategy.set_fitness_buffer(fitnesses)
