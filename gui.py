__author__ = 'carles'

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

class Gui:

    def __init__(self):
        self.root = tk.Tk()
        #self.window = MainWindow(self.root)

        self.dialog = OptionDialog(self.root)

    def run(self):
        self.root.mainloop()





class OptionDialog(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        self.initUI()

    def initUI(self):
        lbl = tk.Label(self, text="Tweet weather predictor")
        lbl.grid(row=0, column=0, pady=4, padx=5)
        tweet = tk.Button(self, text="Tweet")
        tweet.grid(row=1, column=0)
        bck = tk.Button(self, text="CSV file")
        bck.grid(row=1, column=1)
        self.pack()
