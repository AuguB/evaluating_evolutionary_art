from evolution.evaluationstrategy import EvaluationStrategy
import os
from datetime import datetime
from typing import List
from PIL import Image, ImageTk
import numpy as np
from evolution.representation import Representation
import pandas as pd

class Archiver:
    def __init__(self):
        self.evofram = None

        self.timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        self.archivedir = os.path.join(os.getcwd(), 'data', 'archives', self.timestamp)

        if not os.path.isdir(self.archivedir):
            os.mkdir(self.archivedir)

        self.score_and_string_csv = os.path.join(self.archivedir, 'info.csv')
        if not os.path.exists(self.score_and_string_csv):
            csv = open(self.score_and_string_csv, "w")
            csv.write("Run;Iteration;Individual;Fitness;StringRepresentation;LatexRepresentation\n")
            csv.close()

        self.runcount = 0
        self.currentrundir = ""

        self.iterationcount = 0
        self.currentiterdir = ""

        self.make_run_dir()

        self.turing_results_csv = os.path.join(self.archivedir, 'turingtest.csv')
        if not os.path.exists(self.turing_results_csv):
            csv = open(self.turing_results_csv, "w")
            csv.write("g;name1;name2;correct_idx;selected_idx\n")
            csv.close()

        self.tmp = \
            {
                'g': '0',
                'names': ['0','0'],
                'correct_idx': '0',
                'selection': -1
            }

    def make_record(self):
        pass

    def set_evofram(self, evofram):
        self.evofram = evofram

    def archive(self, fitnesses: List[float], population: List[Representation]):
        self.make_iteration_dir()
        csv = open(self.score_and_string_csv, "a")
        for i, (f, ind) in enumerate(zip(fitnesses, population)):
            csv.write(f"{self.runcount-1};{self.iterationcount-1};{i};{f};{str(ind)};{ind.latex_repr()}\n")
            im = Image.fromarray(np.uint8(ind.get_output(300, 300) * 255))
            im.save(os.path.join(self.currentiterdir, f"{i}.png"))
        csv.close()

    def make_run_dir(self):
        rundir = os.path.join(self.archivedir, "run" + str(self.runcount))
        if not os.path.isdir(rundir):
            os.mkdir(rundir)
        self.currentrundir = rundir
        self.runcount += 1
        self.iterationcount = 0

    def make_iteration_dir(self):
        iterdir = os.path.join(self.currentrundir, "iter" + str(self.iterationcount))
        if not os.path.isdir(iterdir):
            os.mkdir(iterdir)
        self.currentiterdir = iterdir
        self.iterationcount += 1

    def set_custom_name(self, custom_name):
        new_archive_name = os.path.join(os.getcwd(), 'data', 'archives', custom_name + "_" + self.timestamp)
        os.rename(self.archivedir, new_archive_name)
        self.archivedir = new_archive_name
        self.runcount = 0
        self.make_run_dir()
        self.score_and_string_csv = os.path.join(self.archivedir, 'info.csv')
        self.turing_results_csv = os.path.join(self.archivedir, 'turingtest.csv')

    def get_turing_images(self):

        turing_data_folder = os.path.join(os.getcwd(), 'data', 'turing_test_data')

        # Decide whether to use g1 or g10
        g = 'g1' if np.random.rand() < 0.5 else 'g10'

        # Select Computer output
        computer_folder = os.path.join(turing_data_folder, f"computer_{g}")
        n_computer_files = len(os.listdir(computer_folder))
        comp_sel_str = os.path.join(computer_folder, f'{np.random.randint(0, n_computer_files)}.png')
        # comp_sel_str = "/home/guus/Uni/AI_Master/Years/1/sem2/NatCo/NatCo-Project/data/turing_test_data/computer_g10/0.png"
        computer_selection = np.array(Image.open(comp_sel_str))

        # Select Human output
        human_folder = os.path.join(turing_data_folder, f"human_g10")
        n_human_files = len(os.listdir(human_folder))
        hum_sel_str = os.path.join(human_folder, f'{np.random.randint(0, n_human_files)}.png')
        # hum_sel_str = "/home/guus/Uni/AI_Master/Years/1/sem2/NatCo/NatCo-Project/data/turing_test_data/human_g10/0.png"
        human_selection = np.array(Image.open(hum_sel_str))

        outputs = [computer_selection, human_selection]
        names = [comp_sel_str, hum_sel_str]
        # Determine random ordering
        reverse = np.random.rand() < 0.5
        if reverse:
            outputs.reverse()
            names.reverse()

        self.tmp = \
            {
                'g': g,
                'names': names,
                'correct_idx': 0 if reverse else 1,
                'selection': -1
            }
        return outputs

    def set_turing_selection(self, idx):
        # correct = self.tmp['correct_idx']
        # print(f"human selected {idx}, the correct selection was {correct}")
        self.tmp['selection'] = idx

    def write_turing_line(self):
        csv = open(self.turing_results_csv, "a")
        if self.tmp['selection'] == -1:
            print("no selection made")
        csv.write(';'.join([self.tmp['g'], self.tmp['names'][0], self.tmp['names'][1], str(self.tmp['correct_idx']),
                            str(self.tmp['selection'])]) + '\n')
        csv.close()

    def get_final_turing_score(self):
        """
        Returns the total number of correctly guessed turing thingies
        """
        tdf = pd.read_csv(self.turing_results_csv)
        return np.sum(tdf['correct_idx'] == tdf['selected_idx'])