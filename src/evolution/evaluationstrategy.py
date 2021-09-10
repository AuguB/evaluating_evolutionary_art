from __future__ import annotations
from typing import List
import numpy as np
from evolution.representation import Representation


class EvaluationStrategy:
    def __init__(self, params):
        self.params = params
        self.name = "EvaluationStrategy"
        self.has_name = True

    def evaluate(self, reps: List[Representation]) -> List[float]:
        pass

    def set_fitness_buffer(self, fitnesses):
        pass


class MeanDistance(EvaluationStrategy):
    def __init(self, params):
        super().__init__(params)
        self.name = "MeanDistance"
        self.has_name = True

    def evaluate(self, reps: List[Representation]) -> List[float]:
        """
        Note: distance measure must be symmetric for this to properly work
        :param reps:
        :return:
        """
        nreps = len(reps)
        mat = np.zeros(2 * [nreps])
        # David: If the distance measure is symmetric, we can do half the work here.
        for i in range(nreps):
            for j in range(i, nreps):
                distance = self.params['distance_metric'](reps[i], reps[j])
                mat[i, j] = distance
                mat[j, i] = distance
        mean_distances = np.mean(mat, axis=1)
        return list(mean_distances)


class UserInput(EvaluationStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.name = "UserInput"
        self.has_name = False
        self.fitness_buffer = []

    def evaluate(self, reps: List[Representation]) -> List[float]:
        assert len(self.fitness_buffer) > 0, "Fitness buffer is empty, did you return the user score from the view?"
        return self.fitness_buffer.copy()

    def set_fitness_buffer(self, fitnesses: List[float]):
        self.fitness_buffer = fitnesses

    def set_name(self, name):
        self.name = name
        self.has_name = True
