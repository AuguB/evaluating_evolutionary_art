from src.evolution.evolutionaryframework import EvolutionaryFramework


class Model:
    """
    This is the class that wraps all the evolutionary things, along with any data access
    """
    def __init__(self, control, archiver, params):
        self.control = control
        self.archiver = archiver
        self.params = params
        self.evofram = EvolutionaryFramework(self.params['evofram_params'], archiver)
        self.dimx = self.dimy = 250

    def set_scores(self, scores):
        self.evofram.set_fitness_buffer(scores)

    def get_new_output(self):
        self.evofram.evolve()
        return self.get_output()

    def archive_manually(self):
        self.evofram.archive()

    def get_output(self):
        return self.evofram.get_output(self.dimx, self.dimy)

    def refresh(self):
        self.evofram = EvolutionaryFramework(self.params['evofram_params'], self.archiver)
