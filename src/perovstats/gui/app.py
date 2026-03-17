from threading import Thread
import sys
import re

from loguru import logger
import tkinter as tk
from tkinter import ttk, scrolledtext

class PerovStatsApp(tk.Tk):
    def __init__(self, process, img_files, config, output_dir):
        super().__init__()

        self.process = process
        self.img_files = img_files
        self.config = config
        self.output_dir = output_dir
        self.title("PerovStats GUI")
        self.geometry("1400x600")

        self.setup_widgets()


    def setup_widgets(self):
        self.label = ttk.Label(self, text="Welcome to the PerovStats GUI app")
        self.label.pack(pady=20)

        self.run_button = ttk.Button(
            self,
            text="Run PerovStats",
            command=self.start_process
        )
        self.run_button.pack(pady=10)

        self.terminal = scrolledtext.ScrolledText(self, state='disabled', height=10)
        self.terminal.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.redirector = OutputRedirector(self.terminal)
        sys.stdout = self.redirector
        self.log_sink_id = logger.add(
            self.redirector,
            level="DEBUG",
            format="<blue>{time:HH:mm:ss}</blue> | <level>{level: <7}</level> | <magenta>{file: <15}</magenta> | {message}",
            colorize=True
        )


    def start_process(self):
        t = Thread(target=self.process, args=(self.img_files, self.config, self.output_dir), daemon=True)
        t.start()


class OutputRedirector:

    COLOUR_MAP = {
        "31": "red", "91": "red",         # ERROR / CRITICAL
        "32": "green", "92": "green",     # SUCCESS
        "33": "yellow", "93": "yellow",   # WARNING
        "34": "blue", "94": "blue",       # DEBUG
        "35": "magenta", "95": "magenta",
        "36": "cyan", "96": "cyan",       # INFO
        "37": "white", "39": "white",     # Default
    }

    def __init__(self, widget):
        self.widget = widget
        for code, colour in self.COLOUR_MAP.items():
            self.widget.tag_config(f"color_{code}", foreground=colour)

    def write(self, message):
        self.widget.after(0, self.insert_text, str(message))

    def insert_text(self, str):
        self.widget.configure(state='normal')

        segments = re.split(r'\x1b\[(\d+)m', str)

        current_tag = None
        for i, part in enumerate(segments):
            if i % 2 == 1:
                current_tag = f"color_{part}" if part in self.COLOUR_MAP else None
            else:
                if part:
                    self.widget.insert(tk.END, part, current_tag)

        # self.widget.insert(tk.END, str)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass
