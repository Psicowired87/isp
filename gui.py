__author__ = 'carles'

import modelmock as model
import numpy as np
import tksimpledialog as spdialog
import tkMessageBox
import tkFileDialog
import matplotlib

matplotlib.use('TkAgg')

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# implement the default mpl key bindings



from matplotlib.figure import Figure

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
        self.entry = None
        self.figure = Figure(figsize=(5, 4), dpi=75)
        self.axes = self.figure.add_subplot(111)
        self.canvas = None
        self.initUI()

    def initUI(self):
        self.parent.title("Tweet weather predictor")
        self.body()
        self.pack(fill=tk.BOTH, expand=1)
        self.buttonbox()

    def body(self):
        frame = tk.Frame(self, relief=tk.RAISED, borderwidth=1, padx=10, pady=10)
        self.initplot(frame, np.ones(24))

        self.entry = tk.Entry(frame)
        self.entry.pack()
        frame.pack(fill=tk.BOTH, expand=1)


    def buttonbox(self):
        box = tk.Frame(self)

        tweet = tk.Button(box, text="Tweet", width=10, command=self.chosentweet)
        tweet.pack(side=tk.RIGHT, padx=5, pady=5)
        csv = tk.Button(box, text="CSV", width=10, command=self.chosencsv)
        csv.pack(side=tk.RIGHT, padx=5, pady=5)

        box.pack()


    def initplot(self, frame, values):
        ind = range(24)

        self.axes.bar(ind, values, align='edge')
        self.axes.set_ylabel("Mass")

        self.axes.set_title('Prediction', fontstyle='italic')

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)


        lbls = ["s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8",
                 "k9", "k10", "k11", "k12", "k13", "k14", "k15"]

        # Set the x tick labels to the group_labels defined above.
        self.axes.set_xticks(ind)
        self.axes.set_xticklabels(lbls)

        # Extremely nice function to auto-rotate the x axis labels.
        # It was made for dates (hence the name) but it works
        # for any long x tick labels
        self.figure.autofmt_xdate()

        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X, expand=1)

        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.X, expand=1)

    def updateplot(self, values):
        ind = range(24)
        if self.axes:
            self.figure.delaxes(self.axes)
            self.axes = self.figure.add_subplot(111)
        self.axes.bar(ind, values, align='edge')
        self.axes.set_ylabel("Mass")

        self.axes.set_title('Prediction', fontstyle='italic')

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)


        lbls = ["s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8",
                 "k9", "k10", "k11", "k12", "k13", "k14", "k15"]

        # Set the x tick labels to the group_labels defined above.
        self.axes.set_xticks(ind)
        self.axes.set_xticklabels(lbls)

        # Extremely nice function to auto-rotate the x axis labels.
        # It was made for dates (hence the name) but it works
        # for any long x tick labels
        self.figure.autofmt_xdate()

        self.canvas.show()


    def chosentweet(self):
        """
        The user has selected to input a tweet -> show main window
        @return:
        """
        tweet = self.entry.get()
        mat = self.model.predict([tweet])
        self.updateplot(mat.squeeze())

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
                "Error", "An error has occurred. {0}".format(err)
            )
            return

        tkMessageBox.showinfo(
            "Success",
            "The operation has been processed correctly"
        )




