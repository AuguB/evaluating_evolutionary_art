from utils.archiver import Archiver
from MVC.model import Model
from MVC.view import View
import tkinter as Tk  # python 3
from typing import List
import numpy as np

class Control:
    """
    This is the main class that runs the whole process. It is in charge of controlling the model and the view
    """
    def __init__(self, params):
        self.root = Tk.Tk()
        self.root.configure(bg='white')
        self.params = params
        self.archiver = Archiver()
        self.model = Model(self, self.archiver, self.params['model_params'])
        self.view = View(self, self.archiver, self.root, self.params['view_params'])


    def update_score_buffer(self, scores):
        self.model.set_scores(scores)

    def get_new_images(self):
        return self.model.get_new_output()

    def get_images(self):
        return self.model.get_output()

    def archive_manually(self):
        self.model.archive_manually()

    def run(self):
        self.root.mainloop()

        # self.view.run()

    def set_image_dimensions(self, dimx, dimy):
        self.model.dimx = dimx
        self.model.dimy = dimy

    def get_turing_images(self) -> List[np.array]:
        return self.archiver.get_turing_images()

    def set_turing_selection(self,idx):
        self.archiver.set_turing_selection(idx)
        self.archiver.write_turing_line()

    def start_new_tree(self):
        self.model.refresh()
        self.archiver.make_run_dir()

    def get_final_turing_score(self):
        return self.archiver.get_final_turing_score()