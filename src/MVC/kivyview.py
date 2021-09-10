import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition

class MainMenu(Screen):
    pass


class SetupScreen(Screen):

    grid_layout = ObjectProperty(None)
    col1  = ObjectProperty(None)
    col2 = ObjectProperty(None)
    col3 = ObjectProperty(None)
    col4 = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(SetupScreen, self).__init__(**kwargs)

    def test_func(self):
        for i in self.grid_layout.children:
            for j in i.children:
                pass
                if not j == self.ids.mainbutton:
                    i.remove_widget(j)

        for i in self.grid_layout.children:
            for j in i.children:
                for k in range(np.random.randint(5)):
                    j.add_widget(Button(text=str(i)))


class TuringTestScreen(Screen):
    pass

class ParticipantScreen(Screen):
    pass

class WindowManager(ScreenManager):

    main_menu = ObjectProperty(None)
    setup_screen = ObjectProperty(None)

class ViewApp(App):

    def build(self):
        m = WindowManager(transition=NoTransition())
        return m
