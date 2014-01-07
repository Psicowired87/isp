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
        self.parent.geometry("1024x500+150+100")
        self.model = model.Model()
        self.entry = None
        self.figure = Figure(figsize=(5, 4), dpi=75)
        self.axes = self.figure.add_subplot(111)
        self.canvas = None
        self.legend = None
        self.text = \
"""s1: I can't tell
s2: Negative
s3: Neutral
s4: Positive
s5: Tweet not related to weather
w1: current (same day) weather
w2: future (forecast)
w3: I can't tell
w4: past weather
k1: clouds
k2: cold
k3: dry
k4: hot
k5: humid
k6: hurricane
k7: I can't tell
k8: ice
k9: other
k10: rain
k11: snow
k12: storms
k13: sun
k14: tornado
k15: wind"""
        self.initUI()

    def initUI(self):
        self.parent.title("Tweet weather predictor")
        self.body()
        self.pack(fill=tk.BOTH, expand=1)
        self.buttonbox()

    def body(self):
        frame = tk.Frame(self)
        plotframe = tk.Frame(frame, relief=tk.RAISED, borderwidth=1, padx=10, pady=10)
        self.initplot(plotframe, np.ones(24))
        self.legend = tk.Label(plotframe, text=self.text, borderwidth=2, justify="left", relief=tk.RIDGE, padx=5)
        self.legend.pack(side=tk.RIGHT)
        plotframe.pack(fill=tk.BOTH, expand=1)
        self.entry = tk.Entry(frame)
        self.entry.pack(fill=tk.X, expand=1, padx=20)
        frame.pack(fill=tk.BOTH, expand=1)


    def buttonbox(self):
        box = tk.Frame(self)

        tweet = tk.Button(box, text="Tweet", width=10, command=self.chosentweet)
        tweet.pack(side=tk.RIGHT, padx=5, pady=5)
        csv = tk.Button(box, text="CSV", width=10, command=self.chosencsv)
        csv.pack(side=tk.RIGHT, padx=5, pady=5)

        box.pack()


    def initplot(self, frame, values):
        locs = np.arange(24) + 0.5
        self.axes.set_xlim([0, 24])
        self.axes.bar(locs, values, align='center')
        self.axes.set_ylabel("Mass")

        self.axes.set_title('Prediction', fontstyle='italic')

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)


        lbls = ["s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8",
                "k9", "k10", "k11", "k12", "k13", "k14", "k15"]

        # Set the x tick labels to the group_labels defined above.
        self.axes.set_xticks(locs)
        self.axes.set_xticklabels(lbls)

        # Extremely nice function to auto-rotate the x axis labels.
        # It was made for dates (hence the name) but it works
        # for any long x tick labels
        # self.figure.autofmt_xdate()

        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    def updateplot(self, values):
        if self.axes:
            self.figure.delaxes(self.axes)
            self.axes = self.figure.add_subplot(111)
        locs = np.arange(24) + 0.5
        self.axes.set_xlim([0, 24])
        self.axes.bar(locs, values, align='center')
        self.axes.set_ylabel("Mass")

        self.axes.set_title('Prediction', fontstyle='italic')

        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)


        lbls = ["s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8",
                "k9", "k10", "k11", "k12", "k13", "k14", "k15"]

        # Set the x tick labels to the group_labels defined above.
        self.axes.set_xticks(locs)
        self.axes.set_xticklabels(lbls)

        # Extremely nice function to auto-rotate the x axis labels.
        # It was made for dates (hence the name) but it works
        # for any long x tick labels
        #self.figure.autofmt_xdate()

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
            if not inp:
                return
            out = tkFileDialog.asksaveasfilename(parent=self, filetypes=[("Comma separated value", "*.csv")],
                                                 title="Save output as...")
            if out:
                predictions = self.model.predict(list(inp))
                inp.close()
                np.savetxt(out, predictions, delimiter=",")
                tkMessageBox.showinfo(
                    "Success",
                    "The operation has been processed correctly"
                )
            else:
                return

        except model.ModelException as err:
            tkMessageBox.showerror(
                "Error", "Error while predicting. {0}".format(err)
            )

        except IOError as err:
            tkMessageBox.showerror(
                "Error", "Error while opening the file. {0}".format(err)
            )

        except Exception as err:
            tkMessageBox.showerror(
                "Error", err
            )






