import tkinter as Tk  # python 3
from PIL import Image, ImageTk
import numpy as np

class View:
    """
    This class contains stuff that draws on the screen. It interfaces with the user.
    """
    def __init__(self, control, archiver, root, params):
        """
        :param control: The control object. All interaction with the model goes through the control.
        :param root: The Tkinter object. All TK objects are created here. mainloop() is called by
        the control object.
        :param params: parameters
        """
        self.control = control
        self.params = params
        # self.anim = self.control.model.params['evofram_params']['representation'].animate
        # self.go_back = self.control.model.params['evofram_params']['representation'].go_back
        self.anim = False
        self.go_back = False
        self.imgs = []
        self.root = root
        self.archiver = archiver

        # Visual configurations
        self.color_mode = self.params['color_mode']
        self.grid_height, self.grid_width = self.params['grid_dim']
        self.x, self.y = self.params['application_resolution']  # to do: make resolution part of settings later
        self.x_pixels, self.y_pixels = self.params['painting_resolution']  # true image resolution
        self.border_width = self.border_height = 2

        self.button_bar_width = self.x / 7  # 100/7 % of the screen is allocated to buttons
        self.button_bar_height = self.y

        # Find full available space for images
        self.image_frame_width_full = self.x - self.button_bar_width
        self.image_frame_height_full = self.y

        # Find proportion of the full space
        self.proportion_full_xy = self.image_frame_width_full/self.image_frame_height_full

        # Get the proportion of the effective used grid
        self.proportion_used_xy = (self.x_pixels*self.params['grid_dim'][0])/ (self.y_pixels*self.params['grid_dim'][1])

        # If the full has more x than y, adapt the effective grid to the y
        if self.proportion_full_xy > self.proportion_used_xy:
            self.image_frame_height = self.image_frame_height_full
            self.image_frame_width = self.image_frame_height_full*self.proportion_used_xy
        # Otherwise, adapt efffective grid to x
        else:
            self.image_frame_height = self.image_frame_width_full/ self.proportion_used_xy
            self.image_frame_width = self.image_frame_width_full

        self.x_pixels,self.y_pixels=self.image_frame_width/self.grid_width, self.image_frame_height/self.grid_height

        self.frames = self.params['anim_frames']

        # Root configurations
        self.root.geometry(f"{str(self.x)}x{str(self.y)}")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.title(self.params['app_name'])
        self.root.deiconify()

        # Initialize screen classes
        self.art_screen = ArtSelectionScreen(root, self, params)
        self.turing_screen = TuringScreen(root, self, params)
        self.start_screen = StartScreen(root, self, params)
        self.participant_run = ParticipantRun(self.root, self, self.params)
        self.start_screen.show()

    def show_startscreen(self):
        self.art_screen.destroy()
        self.turing_screen.destroy()
        self.participant_run.destroy()
        self.start_screen.show()

    def show_artscreen(self):
        self.art_screen.set_representation('art_representation')
        self.start_screen.destroy()
        self.turing_screen.destroy()
        self.participant_run.destroy()
        self.art_screen.show()

    def show_automatronscreen(self):
        self.art_screen.set_representation('automatron_representation')
        self.anim = True
        self.go_back = True
        self.start_screen.destroy()
        self.turing_screen.destroy()
        self.participant_run.destroy()
        self.art_screen.show()

    def show_kernelCAscreen(self):
        self.art_screen.set_representation('kernel_representation')
        self.anim = True
        self.start_screen.destroy()
        self.turing_screen.destroy()
        self.participant_run.destroy()
        self.art_screen.show()

    def show_turingscreen(self):
        self.start_screen.destroy()
        self.art_screen.destroy()
        self.participant_run.destroy()
        self.turing_screen.show()

    def show_participantscreen(self):
        self.art_screen.destroy()
        self.turing_screen.destroy()
        self.start_screen.destroy()
        self.participant_run.show_experiment_start_screen()


class AbstractScreen:
    def __init__(self, root, view, params):
        self.root = root
        self.view = view
        self.params = params

    def show(self):
        pass

    def update(self):
        pass

    def update_images(self):
        pass


class StartScreen(AbstractScreen):
    def __init__(self, root, view, params):
        super().__init__(root, view, params)

    def show(self):
        self.start_frame = Tk.Frame(self.root, width=self.view.image_frame_width, height=self.view.image_frame_height,
                                    background="white", bd=-2)
        self.start_frame.grid(row=0, column=0, padx=0, pady=0)
        self.start_frame.grid_columnconfigure(0, weight=1)
        self.start_frame.grid_rowconfigure(2, weight=1)

        # art screen button
        button_width = 20
        self.btn_art = Tk.Button(self.start_frame, text="Evolve art - Function trees", width=button_width,
                                 command=self.view.show_artscreen)
        self.btn_art.grid(row=0, column=0, padx=0, pady=5, sticky='news')

        # automatron button
        self.btn_automatron = Tk.Button(self.start_frame, text="Evolve art - Cellular Automata", width=button_width,
                                 command=self.view.show_automatronscreen)
        self.btn_automatron.grid(row=1, column=0, padx=0, pady=5, sticky='news')

        # kernel CA button
        self.btn_kernelCA = Tk.Button(self.start_frame, text="Evolve art - Kernel CA", width=button_width,
                                 command=self.view.show_kernelCAscreen)
        self.btn_kernelCA.grid(row=2, column=0, padx=0, pady=5, sticky='news')

        # Turing test button
        self.btn_turing = Tk.Button(self.start_frame, text="Turing test", width=button_width,
                                    command=self.view.show_turingscreen)
        self.btn_turing.grid(row=3, column=0, padx=0, pady=5, sticky='news')

        # Participant button
        self.btn_participant = Tk.Button(self.start_frame, text="I am a participant", width=button_width,
                                         command=self.view.show_participantscreen)
        self.btn_participant.grid(row=4, column=0, padx=0, pady=5, sticky='news')

    def destroy(self):
        if hasattr(self, 'start_frame'):
            self.start_frame.destroy()


class ArtSelectionScreen(AbstractScreen):

    def __init__(self, root, view, params, representation='art_representation'):
        ## todo: set representation in init here
        super().__init__(root, view, params)
        self.rectangle_colors = ['white', 'red', 'orange', 'green']
        self.scoring_granularity = self.params['scoring_granularity']
        assert len(self.rectangle_colors) >= self.params["scoring_granularity"], print(f"Granularity of {self.scoring_granularity} was selected, but only {len(self.rectangle_colors)} colors are available right now.")
        self.scoring_granularity = min(self.params["scoring_granularity"], len(self.rectangle_colors))

    def get_representation(self):
        return self.view.control.params['model_params']['evofram_params']['representation']

    def set_representation(self, representation):
        self.view.control.params['model_params']['evofram_params']['representation'] = self.view.control.params['model_params']['evofram_params'][representation]

    def show(self):
        """
        Initializes frames, configures columns, adds widgets for scoring
        """
        self.image_frame = Tk.Frame(self.root, width=self.view.image_frame_width, height=self.view.image_frame_height,
                                    background="white", bd=-2)
        self.btn_frame = Tk.Frame(self.root, width=self.view.button_bar_width, height=self.view.button_bar_height,
                                  background="white", bd=-2)

        self.image_frame.grid(row=0, column=0, padx=0, pady=0)
        self.btn_frame.grid(row=0, column=1, padx=0, pady=0)
        self.scoresbuffer = np.zeros((int(self.view.grid_height*self.view.grid_width)))

        # set Paintings
        self.img_padx, self.img_pady = 2,2
        self.image_width = int(
            (self.view.image_frame_width - self.view.grid_width * self.view.border_width * self.img_padx * 2) / self.view.grid_width)
        self.image_height = int(
            (self.view.image_frame_height - self.view.grid_height * self.view.border_height * self.img_pady * 2) / self.view.grid_height)
        self.view.control.set_image_dimensions(self.image_width, self.image_height)

        self.view.control.start_new_tree()
        self.scoresbuffer = np.zeros((int(self.view.grid_height * self.view.grid_width)))
        self.imgs = []
        self.get_new_images()
        self.update_images()
        self.current_gen = 0

        if self.view.anim:
            self.cellular()

        # set Buttons
        self.btn_width = 12
        self.generation_label = Tk.Label(self.btn_frame, text=f'Generation {self.current_gen+1}', background='white', font=('Arial', -13))
        self.generation_label.grid(row=0, column=0, sticky=Tk.E + Tk.W)

        self.nextgen_btn = Tk.Button(self.btn_frame, text="New generation", command=self.update,
                                     activeforeground="black", activebackground="red", pady=0, width=self.btn_width)
        self.nextgen_btn.grid(row=1, column=0, padx=10, pady=3)

        self.reset_btn = Tk.Button(self.btn_frame, text="Reset scores", command=self.reset_scores,
                                   activeforeground="black", activebackground="red", pady=0, width=self.btn_width)
        self.reset_btn.grid(row=2, column=0, padx=10, pady=3, sticky=Tk.S)

        # set legend
        font_size = 15
        col = 0
        row = 3
        for _ in range(3):
            e = Tk.Label(self.btn_frame, text=' ', background='white', font=('Arial', -5))
            e.grid(row=row, column=col, sticky=Tk.E + Tk.W)
            row += 1
        e = Tk.Label(self.btn_frame, text='Legend:', background='white', font=('Arial', -font_size, 'italic'))
        e.grid(row=row, column=col, sticky=Tk.E + Tk.W)
        row += 1
        for i in range(self.scoring_granularity):
            if i == 0:
                color = 'grey'
            else:
                color = self.rectangle_colors[i]
            e = Tk.Label(self.btn_frame, text=f'{i} points', background=color, font=('Arial', -font_size))
            e.grid(row=row, column=col, sticky=Tk.E + Tk.W)
            row += 1

        for _ in range(4):
            e = Tk.Label(self.btn_frame, text=' ', background='white', font=('Arial', -5))
            e.grid(row=row, column=col, sticky=Tk.E + Tk.W)
            row += 1

        self.newrun_btn = Tk.Button(self.btn_frame, text="New run", command=self.new_run,
                                    width=self.btn_width)
        self.newrun_btn.grid(row=row, column=0, padx=10, pady=1, sticky=Tk.S)

        row += 1
        self.menu_btn = Tk.Button(self.btn_frame, text="Main menu", command=self.quit_to_main_menu,
                                  width=self.btn_width)
        self.menu_btn.grid(row=row, column=0, padx=10, pady=1, sticky=Tk.S)

    def update_images(self):
        """ Update the pictures shown to the user """
        grid_height, grid_width = self.params['grid_dim']
        self.pil_img = []
        self.buttons = []  # paintings are drawn on top of buttons
        i = 0
        for w in range(grid_width):
            for h in range(grid_height):
                # Find more cool colormaps here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
                out_im = self.imgs[i]
                int_im = np.uint8(out_im * 255)
                img = Image.fromarray(int_im, mode=self.view.color_mode)
                img = img.resize((self.image_width, self.image_height), Image.ANTIALIAS)
                self.pil_img.append(ImageTk.PhotoImage(
                    image=img))  # storing a copy is required to prevent the iamge disappearing in the garbage collection of tkinter

                # self.buttons.append(Tk.Button(self.image_frame, image=self.pil_img[i], width=100,
                #                               height=100, bd=5,
                #                               borderwidth=self.view.border_width, activebackground='white', bg='white',
                #                               relief='solid'))

                self.buttons.append(Tk.Button(self.image_frame, image=self.pil_img[i], width=self.image_width - 5,
                                              height=self.image_height - 5, bd=5,
                                              borderwidth=self.view.border_width, activebackground='white', bg='white',
                                              relief='solid'))
                self.buttons[i]['command'] = lambda c=i: self.create_rectangle(self.buttons[c], c)
                self.buttons[i].grid(row=w, column=h, padx=self.img_padx, pady=self.img_pady)
                i += 1

        if self.view.anim:
            self.cellular()

    def update(self):
        self.send_scores()
        self.current_gen += 1
        self.generation_label = Tk.Label(self.btn_frame, text=f'Generation {self.current_gen+1}', background='white', font=('Arial', -13))
        self.generation_label.grid(row=0, column=0, sticky=Tk.E + Tk.W)
        self.view.archiver.make_record()
        self.get_new_images()
        self.update_images()

    def create_rectangle(self, button, c):
        self.scoresbuffer[c] += 1
        self.scoresbuffer %= self.params["scoring_granularity"]
        button.config(highlightbackground=self.rectangle_colors[int(self.scoresbuffer[c])])
        button.config(bg=self.rectangle_colors[int(self.scoresbuffer[c])])
        button.config(activebackground=self.rectangle_colors[int(self.scoresbuffer[c])])


    def new_run(self):
        self.reset_scores()
        self.current_gen = 0
        self.generation_label = Tk.Label(self.btn_frame, text=f'Generation {self.current_gen+1}', background='white',
                                         font=('Arial', -13))
        self.generation_label.grid(row=0, column=0, sticky=Tk.E + Tk.W)
        self.view.control.start_new_tree()
        self.imgs = []
        self.get_new_images()
        self.update_images()

    def send_scores(self):
        """ Sends the scores to the controller and resets scores"""
        self.view.control.update_score_buffer(self.scoresbuffer)  # to do: not implemented yet
        self.reset_scores()

    def get_new_images(self):
        """ Get new images from the controller """
        if len(self.imgs) == 0:  # This one ideally only holds before the first cycle.
            self.imgs = self.view.control.get_images()
        else:
            self.imgs = self.view.control.get_new_images()

    def reset_scores(self):
        for button in self.buttons:
            button.config(highlightbackground=self.rectangle_colors[0])
        self.scoresbuffer = np.zeros((int(self.view.grid_height * self.view.grid_width)))

    def animate(self, e, button, frames, i):
        if i < self.view.frames:
            img = Image.fromarray(frames[i], mode=self.view.color_mode)
            img = img.resize((self.image_width, self.image_height), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            button.config(image=img)
            button.image = img
            i += 1
            self.image_frame.after(15, lambda button=button, frames=frames: self.animate(e, button, frames, i))

    def cellular(self):
        all_frames = np.swapaxes(np.stack([self.view.control.get_images() for _ in range(self.view.frames)]), 0, 1)
        all_frames = np.uint8(all_frames * 255)
        for i, b in enumerate(self.buttons):
            b.bind('<Enter>', lambda e, button=b, frames=all_frames[i]: self.animate(e, button, frames, 0))
            if self.view.go_back:
                b.bind('<Leave>', lambda e, button=b: self.back(e, button))

    def back(self, e, button):
        idx = self.buttons.index(button)
        img = self.pil_img[idx]
        button.config(image=img)
        button.image = img

    def quit_to_main_menu(self):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        win = Tk.Toplevel()
        win.geometry(f'+{x + 400}+{y + 300}')
        win.wm_title("Quit experiment")

        l = Tk.Label(win, text="Are you sure?")
        l.grid(row=0, column=0)

        def move_on():
            win.destroy()
            self.view.show_startscreen()

        def go_back():
            win.destroy()

        def maybe():
            win.destroy()
            win2 = Tk.Toplevel()
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            win2.geometry(f'+{x + 500}+{y + 300}')
            win2.wm_title("MANTAAA")

            l = Tk.Label(win2, text="manta?")
            l.grid(row=0, column=0)
            l.grid(row=0, column=0)

            def set_imgs():
                for i in range(int(self.params['grid_dim'][0] ** 2)):
                    image = Image.open('data/mantas2.jpg')
                    if self.get_representation() == self.view.control.params['model_params']['evofram_params']['art_representation']:
                        image = np.array(image.resize((self.image_width, self.image_height),
                                         Image.ANTIALIAS)) / 255
                    elif self.get_representation() == self.view.control.params['model_params']['evofram_params']['kernel_representation']:
                        size = self.view.control.params['model_params']['evofram_params']['representation_params']['dl_params']['space_res']
                        image = np.array(image.resize((size, size),
                                         Image.ANTIALIAS)) / 255
                    else:
                        size = self.view.control.params['model_params']['evofram_params']['representation_params']['ca_params']['size']
                        image = np.array(image.resize((size, size),
                                         Image.ANTIALIAS)) / 255
                    self.imgs[i] += np.array(image)
                self.update_images()
                win2.destroy()

            continue_btn = Tk.Button(win2, text="manta!", command=set_imgs, width=self.btn_width, height=2)
            continue_btn.grid(row=1, column=0)

        yes_btn = Tk.Button(win, text="Yes", command=move_on, width=self.btn_width, height=2)
        yes_btn.grid(row=1, column=0)
        no_btn = Tk.Button(win, text="No", command=go_back, width=self.btn_width, height=2)
        no_btn.grid(row=1, column=1)
        maybe_btn = Tk.Button(win, text="Maybe", command=maybe, width=self.btn_width, height=2)
        maybe_btn.grid(row=1, column=2)

    def destroy(self):
        if hasattr(self, 'image_frame'):
            self.image_frame.destroy()
            self.btn_frame.destroy()


class AutomatronSelectionScreen(ArtSelectionScreen):
    def __init__(self, root, view, params):
        super().__init__(root, view, params)


class TuringScreen(AbstractScreen):
    def __init__(self, root, view, params):
        super().__init__(root, view, params)
        self.turing_total_iter = self.params['turing_total_iter']
        self.turing_current_iter = 0

    def show(self):
        self.turing_image_frame = Tk.Frame(self.root, width=self.view.image_frame_width, height=self.view.image_frame_height,
                                    background="white", bd=-2)
        self.turing_btn_frame = Tk.Frame(self.root, width=self.view.button_bar_width, height=self.view.button_bar_height,
                                  background="white", bd=-2)

        self.turing_image_frame.grid(row=0, column=0, padx=0, pady=0)
        self.turing_btn_frame.grid(row=0, column=1, padx=0, pady=0)


        self.img_padx, self.img_pady = 5, 5
        self.border_width = 2
        self.image_width = int(
            (self.view.image_frame_width - self.view.grid_width * self.border_width * self.img_padx * 2) / self.view.grid_width)
        self.image_height = int(
            (self.view.image_frame_height - self.view.grid_height * self.border_width * self.img_pady * 2) / self.view.grid_height)

        self.image_width = int((self.view.image_frame_width - 2 * self.border_width * self.img_padx * 2) / 2)
        self.image_height = int((self.view.image_frame_width - 2 * self.border_width * self.img_padx * 2) / 2)

        self.turing_current_iter = 0
        self.imgs = self.view.control.get_turing_images()
        self.update_images()

        ## set Buttons
        btn_width = 12
        e = Tk.Label(self.turing_btn_frame, text=f'Progress: {self.turing_current_iter+1}/{self.turing_total_iter} ', font=('Arial', -14))
        e.grid(row=0, column=0, sticky=Tk.E + Tk.W)
        row = 1
        for _ in range(3):
            e = Tk.Label(self.turing_btn_frame, text=' ', background='white', font=('Arial', -5))
            e.grid(row=row, column=0, sticky=Tk.E + Tk.W)
            row += 1

        self.menu_btn = Tk.Button(self.turing_btn_frame, text="Main menu", command=self.view.show_startscreen,
                                  width=btn_width)
        self.menu_btn.grid(row=row, column=0, padx=10, pady=1, sticky=Tk.S)

    def update_images(self):
        self.buttons = []
        self.pil_img = []
        for i in range(2):
            out_im = self.imgs[i]
            int_im = np.uint8(out_im * 255)
            img = Image.fromarray(int_im, mode="RGB")
            img = img.resize((self.image_width, self.image_height), Image.ANTIALIAS)
            self.pil_img.append(ImageTk.PhotoImage(image=img))  # storing a copy is required to prevent the image disappearing in the garbage collection of tkinter
            self.buttons.append(Tk.Button(self.turing_image_frame, image=self.pil_img[i], width=self.image_width - 5, height=self.image_height - 5,
                      bd=5,borderwidth=self.border_width, activebackground='white', bg='white', relief='solid'))
            self.buttons[i]['command'] = lambda c=i: self.update(c)
            self.buttons[i].grid(row=0, column=i, padx=self.img_padx, pady=self.img_pady)


    def update(self, i):
        self.view.control.set_turing_selection(i)
        self.turing_current_iter += 1
        self.imgs = []

        if self.turing_current_iter >= self.turing_total_iter:
            self.turing_image_frame.destroy()
            self.turing_btn_frame.destroy()
            self.score_window = Tk.Frame(self.root, width=self.view.image_frame_width, height=self.view.image_frame_height, background="white", bd=-2)
            self.score_window.grid(row=0, column=0, padx=0, pady=0)

            def end_turing():
                self.score_window.destroy()
                self.view.show_startscreen()

            score = self.view.control.get_final_turing_score()
            percentage = int(score/self.turing_total_iter * 100)
            l = Tk.Label(self.score_window, text=f"You answered {percentage}% correct", font=('Arial', -14), width=30,height=2)
            l.grid(row=0, column=0)
            b = Tk.Button(self.score_window, text=f"Back to main menu", width=30, command=end_turing)
            b.grid(row=1, column=0)

        else:
            e = Tk.Label(self.turing_btn_frame, text=f'Progress: {self.turing_current_iter + 1}/{self.turing_total_iter} ',
                         font=('Arial', -14))
            e.grid(row=0, column=0, sticky=Tk.E + Tk.W)

            self.imgs = self.view.control.get_turing_images()
            self.update_images()

    def destroy(self):
        if hasattr(self, 'turing_image_frame'):
            self.turing_image_frame.destroy()
            self.turing_btn_frame.destroy()


class ParticipantRun(AbstractScreen):
    def __init__(self, root, view, params):
        super().__init__(root, view, params)
        self.N_art_sessions = params['experiment_art_sessions']
        self.N_art_generations = params['experiment_art_generations']
        self.current_art_round = 0
        self.current_gen = 0
        self.current_turing_round = 0

    def show_experiment_start_screen(self):
        # Initialize frame and grid sizes
        self.start_text_frame = Tk.Frame(self.root, width=self.view.image_frame_width, height=self.view.image_frame_height*0.5,
                                    background="white", bd=-2)
        self.start_btn_frame = Tk.Frame(self.root, width=self.view.image_frame_width, height=self.view.image_frame_height*0.5,
                                    background="white", bd=-2)

        self.start_text_frame.grid(row=0, column=0, padx=0, pady=0)

        text_width = 80
        button_width = 30
        info_str_1 = f"During the first part, you are asked to evolve art by rating each painting between 0 and {self.params['scoring_granularity']-1}."
        info_str_2 = f"Scores can be given by clicking on a painting. The score is depicted by the color of the border. "
        info_str_3 = f"This will be repeated {self.N_art_sessions} times. Note that the number of generations is fixed to {self.N_art_generations}."
        info_str_4 = " "
        info_str_5 = "During the second part test, you are asked to choose between two images. "
        info_str_6 = "One created by a human and the other by a computer. "
        info_str_7 = "Please click on the image you belief is created by a human. "
        info_str = [info_str_1, info_str_2, info_str_3, info_str_4, info_str_5, info_str_6, info_str_7]
        self.information_labels = []

        # welcome label
        information_label = Tk.Label(self.start_text_frame, text="Thank you for participating in the experiment.",
                                     width=text_width, anchor='center', background='white', font=('Arial', -15, "bold"))
        information_label.grid(row=0, column=0, padx=0, pady=5, sticky='news')
        self.information_labels.append(self.information_labels)

        # instruction labels
        for i, str in enumerate(info_str):
            information_label = Tk.Label(self.start_text_frame, text=str, width=text_width, anchor='w',
                                               background='white', font=('Arial', -15))
            information_label.grid(row=i+1, column=0, padx=0, pady=0, sticky='news')
            self.information_labels.append(information_label)

        # Enter participant name
        # todo: why is the button and label width not changing here?
        self.empty_label = Tk.Label(self.start_text_frame, text=" ", width=button_width, background='white')
        self.empty_label.grid(row=i + 2, column=0, padx=0, pady=0, sticky='news')
        self.name_label = Tk.Label(self.start_text_frame, text="Please enter your name: ", width=button_width, anchor='center',
                                     background='white', font=('Arial', -15, 'italic'))
        self.name_label.grid(row=i+3, column=0, padx=0, pady=0, sticky='news')

        def to_experiment():
            name = self.type_box.get()
            self.view.archiver.set_custom_name(name)
            self.start_experiment()

        self.type_box = Tk.Entry(self.start_text_frame, width=button_width)
        self.type_box.grid(row=i+4, column=0, padx=0, pady=5, sticky='news')
        self.type_box.focus_set()

        # Start button
        self.start_btn = Tk.Button(self.start_text_frame, text="Get started!", width=button_width,
                                         command=to_experiment)
        self.start_btn.grid(row=i+5, column=0, padx=0, pady=5, sticky='news')

    def start_experiment(self):
        self.destroy_start_screen()
        self.experiment_art_selection()

    def experiment_art_selection(self):
        self.experiment_art_screen = ArtSelectionScreen(self.root, self.view, self.params)
        self.experiment_art_screen.show()
        self.experiment_art_screen.btn_frame.destroy()
        self.make_buttons_experiment()

    def update_buttons(self):
        self.experiment_art_screen.generation_label = Tk.Label(self.experiment_art_screen.btn_frame,
                                                               text=f'Generation {self.current_gen + 1}/{self.N_art_generations}',
                                                               background='white', font=('Arial', -13))
        self.experiment_art_screen.generation_label.grid(row=0, column=0, sticky=Tk.W+Tk.E)
        self.experiment_art_screen.round_label = Tk.Label(self.experiment_art_screen.btn_frame,
                                                          text=f'Round {self.current_art_round + 1}/{self.N_art_sessions}',
                                                          background='white', font=('Arial', -13))
        self.experiment_art_screen.round_label.grid(row=1, column=0, sticky=Tk.W+Tk.E)

        if self.current_art_round+1 >= self.N_art_sessions and self.current_gen+1 >= self.N_art_generations:
            self.experiment_art_screen.nextgen_btn = Tk.Button(self.experiment_art_screen.btn_frame, text="To Turing test",
                                        command=self.experiment_update,
                                        width=self.btn_width)
            self.experiment_art_screen.nextgen_btn.grid(row=2, column=0, padx=10, pady=3, sticky=Tk.S)
        elif self.current_gen+1 >= self.N_art_generations:
            self.experiment_art_screen.nextgen_btn = Tk.Button(self.experiment_art_screen.btn_frame, text="Next round", command=self.experiment_update,
                                    width=self.btn_width)
            self.experiment_art_screen.nextgen_btn.grid(row=2, column=0, padx=10, pady=3, sticky=Tk.S)
        else:
            self.experiment_art_screen.nextgen_btn = Tk.Button(self.experiment_art_screen.btn_frame,
                                                               text="New generation", command=self.experiment_update,
                                                               activeforeground="black", activebackground="red", pady=0,
                                                               width=self.btn_width)
            self.experiment_art_screen.nextgen_btn.grid(row=2, column=0, padx=10, pady=3)

    def make_buttons_experiment(self):
        ## make buttons for the experiment
        self.experiment_art_screen.btn_frame = Tk.Frame(self.root, width=self.view.button_bar_width, height=self.view.button_bar_height,
                                  background="white", bd=-2)
        self.experiment_art_screen.btn_frame.grid(row=0, column=1, padx=0, pady=0)
        self.btn_width = 12

        self.experiment_art_screen.generation_label = Tk.Label(self.experiment_art_screen.btn_frame, text=f'Generation {self.current_gen+1}/{self.N_art_generations}', background='white',font=('Arial', -13))
        self.experiment_art_screen.generation_label.grid(row=0, column=0, sticky=Tk.W+Tk.E)
        self.experiment_art_screen.round_label = Tk.Label(self.experiment_art_screen.btn_frame, text=f'Round {self.current_art_round + 1}/{self.N_art_sessions}', background='white', font=('Arial', -13))
        self.experiment_art_screen.round_label.grid(row=1, column=0, sticky=Tk.W+Tk.E)

        self.experiment_art_screen.nextgen_btn = Tk.Button(self.experiment_art_screen.btn_frame, text="New generation", command=self.experiment_update,
                                     activeforeground="black", activebackground="red", pady=0, width=self.btn_width)
        self.experiment_art_screen.nextgen_btn.grid(row=2, column=0, padx=10, pady=3)

        self.experiment_art_screen.reset_btn = Tk.Button(self.experiment_art_screen.btn_frame, text="Reset scores", command=self.experiment_art_screen.reset_scores,
                                   activeforeground="black", activebackground="red", pady=0, width=self.btn_width)
        self.experiment_art_screen.reset_btn.grid(row=3, column=0, padx=10, pady=3, sticky=Tk.S)

        # set legend
        font_size = 15
        col = 0
        row = 4
        for _ in range(3):
            e = Tk.Label(self.experiment_art_screen.btn_frame, text=' ', background='white', font=('Arial', -5))
            e.grid(row=row, column=col, sticky=Tk.E + Tk.W)
            row += 1
        e = Tk.Label(self.experiment_art_screen.btn_frame, text='Legend:', background='white', font=('Arial', -font_size, 'italic'))
        e.grid(row=row, column=col, sticky=Tk.E + Tk.W)
        row += 1
        for i in range(self.view.art_screen.scoring_granularity):
            if i == 0:
                color = 'grey'
            else:
                color = self.view.art_screen.rectangle_colors[i]
            e = Tk.Label(self.experiment_art_screen.btn_frame, text=f'{i} points', background=color, font=('Arial', -font_size))
            e.grid(row=row, column=col, sticky=Tk.E + Tk.W)
            row += 1

        for _ in range(4):
            e = Tk.Label(self.experiment_art_screen.btn_frame, text=' ', background='white', font=('Arial', -5))
            e.grid(row=row, column=col, sticky=Tk.E + Tk.W)
            row += 1

        self.menu_btn = Tk.Button(self.experiment_art_screen.btn_frame, text="Quit experiment", command=self.experiment_quit_action,
                                  width=self.btn_width)
        self.menu_btn.grid(row=row, column=0, padx=10, pady=1, sticky=Tk.S)

    def experiment_update(self):
        if self.current_gen+1 == self.N_art_generations and self.current_art_round+1 == self.N_art_sessions:
            self.view.control.archive_manually()
            self.experiment_quit_art()
        elif self.current_gen+1 == self.N_art_generations:
            self.view.control.archive_manually()
            self.experiment_art_screen.send_scores()
            self.current_art_round += 1
            self.current_gen = 0
            self.experiment_art_screen.new_run()
            self.update_buttons()
        else:
            self.experiment_art_screen.send_scores()
            self.experiment_art_screen.view.archiver.make_record()
            self.experiment_art_screen.get_new_images()
            self.experiment_art_screen.update_images()
            self.current_gen += 1
            self.update_buttons()

    def experiment_quit_action(self):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        win = Tk.Toplevel()
        win.geometry(f'+{x + 400}+{y + 300}')
        win.wm_title("Quit experiment")

        l = Tk.Label(win, text="Are you sure?")
        l.grid(row=0, column=0)

        def move_on():
            win.destroy()
            self.experiment_art_screen.destroy()
            self.current_art_round = 0
            self.current_gen = 0
            self.view.show_startscreen()

        def go_back():
            win.destroy()

        def maybe():
            win.destroy()
            win2 = Tk.Toplevel()
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            win2.geometry(f'+{x + 500}+{y + 300}')
            win2.wm_title("MANTAAA")

            l = Tk.Label(win2, text="manta?")
            l.grid(row=0, column=0)
            l.grid(row=0, column=0)

            def set_imgs():
                for i in range(int(self.params['grid_dim'][0] ** 2)):
                    image = Image.open('data/mantas2.jpg')
                    image = np.array(
                        image.resize((self.experiment_art_screen.image_width, self.experiment_art_screen.image_height),
                                     Image.ANTIALIAS)) / 255
                    self.experiment_art_screen.imgs[i] += np.array(image)
                self.experiment_art_screen.update_images()
                win2.destroy()

            continue_btn = Tk.Button(win2, text="manta!", command=set_imgs, width=self.btn_width, height=2)
            continue_btn.grid(row=1, column=0)

        yes_btn = Tk.Button(win, text="Yes", command=move_on, width=self.btn_width, height=2)
        yes_btn.grid(row=1, column=0)
        no_btn = Tk.Button(win, text="No", command=go_back, width=self.btn_width, height=2)
        no_btn.grid(row=1, column=1)
        maybe_btn = Tk.Button(win, text="Maybe", command=maybe, width=self.btn_width, height=2)
        maybe_btn.grid(row=1, column=2)

    def experiment_quit_art(self):
        self.experiment_art_screen.destroy()
        self.current_art_round = 0
        self.current_gen = 0
        self.experiment_turing_test()

    def experiment_turing_test(self):
        # todo: make turing test appear here
        self.experiment_turing_screen = TuringScreen(self.root, self.view, self.params)

        self.experiment_turing_screen.update = self.update
        self.experiment_turing_screen.show()

        def end_turing_experiment():
            self.experiment_turing_screen.turing_image_frame.destroy()
            self.experiment_turing_screen.turing_btn_frame.destroy()
            self.view.show_startscreen()
        self.experiment_turing_screen.turing_btn_frame.menu_btn = Tk.Button(self.experiment_turing_screen.turing_btn_frame,
                                                                            text="Main menu", command=end_turing_experiment, width=self.btn_width)
        self.experiment_turing_screen.turing_btn_frame.menu_btn.grid(row=4, column=0, padx=10, pady=1, sticky=Tk.S)

        self.destroy()

    def update(self, i):
        self.view.control.set_turing_selection(i)
        self.experiment_turing_screen.turing_current_iter += 1
        self.imgs = []

        if self.experiment_turing_screen.turing_current_iter >= self.experiment_turing_screen.turing_total_iter:
            self.experiment_turing_screen.turing_image_frame.destroy()
            self.experiment_turing_screen.turing_btn_frame.destroy()
            self.score_window = Tk.Frame(self.root, width=self.view.image_frame_width, height=self.view.image_frame_height, background="white", bd=-2)
            self.score_window.grid(row=0, column=0, padx=0, pady=0)
            text_width = 40

            def end_turing_experiment():
                self.score_window.destroy()
                self.experiment_turing_screen.turing_image_frame.destroy()
                self.experiment_turing_screen.turing_btn_frame.destroy()
                self.view.show_startscreen()

            score = self.view.control.get_final_turing_score()
            percentage = int(score/self.experiment_turing_screen.turing_total_iter * 100)


            # labels
            information_label = Tk.Label(self.score_window, text="Thank you for participating.",
                                         width=text_width, anchor='center', background='white',
                                         font=('Arial', -15, "bold"))
            information_label.grid(row=0, column=0, padx=0, pady=5, sticky='news')

            l = Tk.Label(self.score_window, text=f"You answered {percentage}% correct", font=('Arial', -14), width=text_width)
            l.grid(row=1, column=0)
            l2 = Tk.Label(self.score_window, text="All artworks from the experiment are saved.", width=text_width, anchor='center',
                                         background='white', font=('Arial', -15))
            l2.grid(row=2, column=0, padx=0, pady=0)
            l3 = Tk.Label(self.score_window, text=" ", width=text_width,
                          anchor='center',
                          background='white', font=('Arial', -15))
            l3.grid(row=3, column=0, padx=0, pady=0)
            b = Tk.Button(self.score_window, text=f"Back to main menu", width=text_width, command=end_turing_experiment, pady=5)
            b.grid(row=4, column=0)

        else:
            e = Tk.Label(self.experiment_turing_screen.turing_btn_frame, text=f'Progress: {self.experiment_turing_screen.turing_current_iter + 1}/{self.experiment_turing_screen.turing_total_iter} ',
                         font=('Arial', -14))
            e.grid(row=0, column=0, sticky=Tk.E + Tk.W)

            self.experiment_turing_screen.imgs = self.view.control.get_turing_images()
            self.experiment_turing_screen.update_images()

    def destroy(self):
        self.destroy_start_screen()
        self.destroy_end_screen()

    def destroy_start_screen(self):
        if hasattr(self, 'start_text_frame'):
            self.start_text_frame.destroy()
            self.start_btn_frame.destroy()

    def destroy_end_screen(self):
        if hasattr(self, 'end_frame'):
            self.end_frame.destroy()



