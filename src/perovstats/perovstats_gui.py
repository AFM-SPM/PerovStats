import tkinter as tk
from tkinter import ttk
import numpy as np

from .processing import run_process

WINDOW_SIZE = (600,400)
WINDOW_RESIZE = False

class PerovStatsGUI():
    def __init__(self, root):
        self.root = root
        self.root.title("PerovStats GUI")
        self.root.geometry("1000x600")

        # Configure the main window grid weights
        # Column 0 (Left), Column 2 (Right) weight = 0 (stays small)
        # Column 1 (Center) weight = 1 (expands to fill space)
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=1) # Make the row fill the height

        # --- 1. Left Sidebar ---
        self.left_bar = tk.Frame(self.root, width=200, pady=20, padx=10)
        self.left_bar.grid(row=0, column=0, sticky="nsew")

        self.upload_btn = tk.Button(self.left_bar, text="Upload scan(s)")
        self.upload_btn.grid(row=0, column=0, padx=25, pady=5, sticky="ew")
        ttk.Separator(self.left_bar, orient="horizontal").grid(row=1, column=0, pady=20, sticky="ew")

        self.analysis_mode = tk.StringVar(value="auto")

        tk.Label(self.left_bar, text="Segmentation method:").grid(row=2, column=0, sticky="w")

        # 3. First Radio Button
        self.radio_auto = tk.Radiobutton(
            self.left_bar,
            text="Traditional",
            variable=self.analysis_mode,
            value="traditional"
        )
        self.radio_auto.grid(row=3, column=0, columnspan=2, sticky="w")

        # 4. Second Radio Button
        self.radio_manual = tk.Radiobutton(
            self.left_bar,
            text="Cellpose ML",
            variable=self.analysis_mode,
            value="cellpose"
        )
        self.radio_manual.grid(row=4, column=0, columnspan=2, sticky="w")


        ttk.Separator(self.left_bar, orient="horizontal").grid(row=5, column=0, pady=20, sticky="ew")

        self.run_btn = tk.Button(self.left_bar, text="Frequency Splitting")
        self.run_btn.grid(row=6, column=0, padx=25, pady=5, sticky="ew")

        self.run_btn = tk.Button(self.left_bar, text="Image Segmentation")
        self.run_btn.grid(row=7, column=0, padx=25, pady=5, sticky="ew")

        self.run_btn = tk.Button(self.left_bar, text="Grain Processing")
        self.run_btn.grid(row=8, column=0, padx=25, pady=5, sticky="ew")

        # --- 2. Central Area ---
        self.center_area = tk.Frame(self.root, bg="white", borderwidth=2, relief="sunken")
        self.center_area.grid(row=0, column=1, sticky="nsew")

        # Centering a label inside the center area
        self.center_area.columnconfigure(0, weight=1)
        self.center_area.rowconfigure(0, weight=1)
        tk.Label(self.center_area, text="Upload a scan and configure/ run the steps on the left to generate data.", bg="white").grid(row=0, column=0)

        # --- 3. Right Sidebar (Settings) ---
        self.right_bar = tk.Frame(self.root, width=250, bg="#ecf0f1", padx=10, pady=10)
        self.right_bar.grid(row=0, column=2, sticky="nsew")

        tk.Label(self.right_bar, text="Settings", bg="#ecf0f1", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, pady=10)

        # Example of why Grid is better for settings:
        tk.Label(self.right_bar, text="Threshold:", bg="#ecf0f1").grid(row=1, column=0, sticky="w")
        tk.Entry(self.right_bar, width=10).grid(row=1, column=1, padx=5, pady=5)


    def frequency_settings(self):
        pass

    def segmentation_settings(self):
        pass

    def grain_settings(self):
        pass


    def upload_files(self):
        pass


    def run_splitting(self):
        pass


    def run_segmentation(self):
        pass


    def run_smears(self):
        pass


    def run_grains(self):
        pass


def run_gui(args):
    root = tk.Tk()
    app = PerovStatsGUI(root)
    root.mainloop()
