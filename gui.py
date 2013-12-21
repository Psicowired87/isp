__author__ = 'carles'

import modelmock as model
import numpy as np

import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk


class Gui:
    def __init__(self):
        self.droot = tk.Tk()
        self.wroot = tk.Tk()
        #self.window = MainWindow(self.root)

        self.dialog = OptionDialog(self.droot, self)
        #self.dialog.hide()

        self.window = MainWindow(self.wroot, self)
        #self.window.hide()

        self.model = model.Model()

    def run(self):
        self.window.hide()
        self.dialog.waitforoption()
        self.droot.mainloop()
        self.wroot.mainloop()


    def chosentweet(self):
        """
        The user has selected to input a tweet -> show main window
        @return:
        """
        self.window.show()

    def chosencsv(self):
        """
        The user has selected to use a csv -> show file selectors
        @return:
        """
        try:
            inp = tk.tkFileDialog.askopenfile(parent=self.wroot, mode='r', title='Choose an input file')
            out = tk.tkFileDialog.asksaveasfilename(parent=self.wroot, filetypes=("Comma separated value", "*.csv"),
                                                    title="Save output as...")
            if inp and out:
                predictions = self.model.predict(list(inp))
                inp.close()
                np.savetxt(out, predictions, delimiter=",")
        except (IOError, EnvironmentError) as err:
            tk.tkMessageBox.showwarning(
                "Error",
                "An error has occurred {0}".format(err)
            )
            self.dialog.waitforoption()

        tk.tkMessageBox.showinfo(
                "Success",
                "The operation has been processed correctly"
            )



class HideableWindow(tk.Toplevel):
    def __init__(self, parent, gui):
        tk.Toplevel.__init__(self, parent)
        self.parent = parent
        self.gui = gui

    def hide(self):
        self.parent.withdraw()

    def show(self):
        self.parent.update()
 

class OptionDialog(HideableWindow):
    def __init__(self, parent, gui):
        HideableWindow.__init__(self, parent, gui)
        self.initUI()

    def initUI(self):
        lbl = tk.Label(self, text="Tweet weather predictor")
        lbl.grid(row=0, column=0, pady=4, padx=5)
        tweet = tk.Button(self, text="Tweet", command=self.chosentweet)
        tweet.grid(row=1, column=0)
        bck = tk.Button(self, text="CSV file", command=self.chosencsv)
        bck.grid(row=1, column=1)
        self.pack()

    def chosentweet(self):
        self.hide()
        self.gui.chosentweet()

    def chosencsv(self):
        self.hide()
        self.gui.chosentweet()

    def waitforoption(self):
        self.show()


class MainWindow(HideableWindow):
    def __init__(self, parent, gui):
        HideableWindow.__init__(self, parent, gui)


