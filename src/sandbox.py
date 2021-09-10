from utils.distancemetrics import *
from utils.archiver import Archiver
from MVC.control import Control

from utils.distancemetrics import *
from evolution.evaluationstrategy import *
from evolution.evolutionstrategy import *
from evolution.selectionstrategy import *
from src.evolution.representation import *
import matplotlib.pyplot as plt
grid_height = 3
grid_width = 3
pop_size = grid_height * grid_width
params = {
    'view_params': {
        'scoring_granularity':4,
        'grid_dim': (grid_height, grid_width),
        'application_resolution': (1440, 800),
        'painting_resolution': (250, 250),
        'app_name':'MANTAAAAAAAAAAA'
    },
    'model_params': {
        'evofram_params': {
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
                'K':2
            },
            'evaluationstrategy': UserInput,
            'evaluation_params': {
                'distance_metric': euclidean
            },
            'evolutionstrategy': GP,
            'evolution_params': {
                'pop_size':pop_size,
                # 'p_m': 0.01,
                'p_m': 0,
                'p_c': 1
            }
        }
    }
}

control = Control(params)
control.get_turing_images()
control.set_turing_selection(1)