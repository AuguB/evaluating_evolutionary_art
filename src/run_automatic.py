import os

from MVC.control import Control
from MVC.model import Model
from evolution.evolutionaryframework import EvolutionaryFramework
from utils.archiver import Archiver
from utils.distancemetrics import *
from src.evolution.evaluationstrategy import *
from src.evolution.evolutionstrategy import *
from src.evolution.representation import *
from src.evolution.selectionstrategy import *

# pattern =
curdir = os.getcwd()
while not curdir.endswith("NatCo-Project"):
    os.chdir(os.path.dirname(curdir))
    curdir = os.getcwd()

# TODO: implement grid_height and grid_width
grid_height = 3
grid_width = 3
pop_size = grid_height * grid_width

evofram_params = {
    'population_size': pop_size,
    'representation': NChannelTreeRepresentationV2,
    'representation_params': {
        'n_channels': 3,
        'tree_params': {
            'method': 'hybrid',
            'd_max': 7
        }
    },
    'selectionstrategy': RouletteWheelSelection,
    'selection_params': {
        'selection_pool_size': 10,
        'K': 2
    },
    'evaluationstrategy': MeanDistance,
    'evaluation_params': {
        'distance_metric': variance
    },
    'evolutionstrategy': GP,
    'evolution_params': {
        'pop_size': pop_size,
        # 'p_m': 0.01,
        'p_m': 0,
        'p_c': 1
    }
}
archiver = Archiver()
archiver.set_custom_name("Autorun_10iters")
n_runs = 10
iter_per_run = 10
for run in range(n_runs):
    evofram = EvolutionaryFramework(evofram_params, archiver)
    for i in range(iter_per_run):
        evofram.evolve()
    archiver.make_run_dir()

archiver = Archiver()
archiver.set_custom_name("Autorun_1iter")
n_runs = 10
iter_per_run = 1
for run in range(n_runs):
    evofram = EvolutionaryFramework(evofram_params, archiver)
    for i in range(iter_per_run):
        evofram.evolve()
    if run < (n_runs - 1):
        archiver.make_run_dir()

