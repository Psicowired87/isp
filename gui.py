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
        self.parent = parent
        self.parent.geometry("800x600+300+100")
        self.model = model.Model()
        self.initUI()

    def initUI(self):

        self.parent.title("Buttons")

        frame = tk.Frame(self, relief=tk.RAISED, borderwidth=1)
        frame.pack(fill=tk.BOTH, expand=1)

        self.pack(fill=tk.BOTH, expand=1)

        self.buttonbox()


    def buttonbox(self):
        box = tk.Frame(self)

        tweet = tk.Button(box, text="Tweet", width=10, )
        tweet.pack(side=tk.RIGHT, padx=5, pady=5)
        csv = tk.Button(box, text="CSV", width=10, command=self.chosencsv)
        csv.pack(side=tk.RIGHT, padx=5, pady=5)

        box.pack()


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
            out = tkFileDialog.asksaveasfilename(parent=self, filetypes=[("Comma separated value", "*.csv")],
                                                 title="Save output as...")
            if inp and out:
                predictions = self.model.predict(list(inp))
                inp.close()
                np.savetxt(out, predictions, delimiter=",")
            else:
                raise ValueError("Invalid files")
        except Exception as err:
            tkMessageBox.showerror(
                "Error",
                "An error has occurred. {0}".format(err)
            )
            return

        tkMessageBox.showinfo(
            "Success",
            "The operation has been processed correctly"
        )




