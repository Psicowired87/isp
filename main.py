__author__ = 'carles'

import gui
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk






def main():
    root = tk.Tk()
    window = gui.Gui(root)
    window.mainloop()











if __name__ == "__main__":
    main()