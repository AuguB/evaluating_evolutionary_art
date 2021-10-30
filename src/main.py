import os
from MVC.control import Control
from utils.distancemetrics import *
from evolution.evaluationstrategy import *
from evolution.evolutionstrategy import *
from evolution.representation import *
from evolution.selectionstrategy import *

def main():

    curdir = os.getcwd()
    while not curdir.endswith("Evaluating_Evolutionary_Art"):
        os.chdir(os.path.dirname(curdir))
        curdir = os.getcwd()

    # TODO: implement grid_height and grid_width
    grid_height = 3
    grid_width = 3
    pop_size = grid_height * grid_width
    params = {
        'view_params': {
            'scoring_granularity': 4,
            'grid_dim': (grid_height, grid_width),
            'application_resolution': (1200, 700),
            'painting_resolution': (250, 250),
            'app_name': 'Evaluating Evolutionary Art',
            'color_mode': 'RGB',
            'anim_frames': 50,
            'turing_total_iter': 20,
            'experiment_art_generations': 10,
            'experiment_art_sessions': 3
        },
        'model_params': {
            'evofram_params': {
                'population_size': pop_size,
                'representation': NChannelTreeRepresentationV2,
                'art_representation': NChannelTreeRepresentationV2,
                'automatron_representation': CellularAutomatron,
                'kernel_representation': KernelCA,
                'representation_params':
                {
                    'n_channels': 3,
                    'tree_params':
                    {
                        'method': 'hybrid',
                        'd_max': 7
                    },
                    'ca_params':
                    {
                        'size': 64,
                    },

                    'dl_params':
                    {
                        'space_res': 256,
                        'time_res': 50,
                        'state_res': 50
                    }
                },
                'selectionstrategy': TournamentSelection,
                'selection_params':
                {
                    'selection_pool_size': 5,
                    'K': 2
                },
                'evaluationstrategy': UserInput,
                'evaluation_params':
                {
                    'distance_metric': euclidean
                },
                'evolutionstrategy': GP,
                'evolution_params':
                {
                    'pop_size': pop_size,
                    'p_m': 0,
                    'p_c': 1
                }
            }
        }
    }

    control = Control(params)
    control.run()

if __name__ == "__main__":
    main()