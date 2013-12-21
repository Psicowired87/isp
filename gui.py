__author__ = 'carles'

import modelmock as model
import numpy as np
import tksimpledialog as spdialog
import tkMessageBox
import tkFileDialog
import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk


class Gui(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.model = model.Model()

        self.showdialog()

    def showdialog(self):
        OptionDialog(self)

    def chosentweet(self):
        """
        The user has selected to input a tweet -> show main window
        @return:
        """
        #self.window.show()

    def chosencsv(self):
        """
        The user has selected to use a csv -> show file selectors
        @return:
        """
        try:
            inp = tkFileDialog.askopenfile(parent=self, mode='r', title='Choose an input file')
            out = tkFileDialog.asksaveasfilename(parent=self, filetypes=("Comma separated value", "*.csv"),
                                                    title="Save output as...")
            if inp and out:
                predictions = self.model.predict(list(inp))
                inp.close()
                np.savetxt(out, predictions, delimiter=",")
        except Exception as err:
            tk.tkMessageBox.showwarning(
                "Error",
                "An error has occurred {0}".format(err)
            )

        tkMessageBox.showinfo(
                "Success",
                "The operation has been processed correctly"
            )



class OptionDialog(spdialog.Dialog):
    TWEET = "tweet"
    CSV = "csv"

    def __init__(self, parent, title=None):
        spdialog.Dialog.__init__(self, parent, title)
        self.result = None
        self.parent = parent

    def body(self, master):
        tk.Label(master, text="Tweet weather predictor")

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons

        box = tk.Frame(self)

        tweet = tk.Button(box, text="Tweet", width=10, command=self.chosentweet)
        tweet.pack(side=tk.LEFT, padx=5, pady=5)
        csv = tk.Button(box, text="CSV", width=10, command=self.chosencsv)
        csv.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Escape>", self.cancel)

        box.pack()

    def chosentweet(self):
        self.cancel()
        self.parent.chosentweet()

    def chosencsv(self):
        self.result = self.CSV
        self.parent.chosencsv()





