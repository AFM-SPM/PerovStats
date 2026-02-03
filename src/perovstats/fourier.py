from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import skimage as ski
from matplotlib import pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

from .freqsplit import frequency_split, find_cutoff
from .segmentation import create_grain_mask
from .segmentation import threshold_mad, threshold_mean_std


LOGGER = logging.getLogger(__name__)

image_params = {}
final_images = {}

def create_masks(perovstats_object) -> None:
    split_frequencies(perovstats_object)

    output_dir = Path(perovstats_object.config["output_dir"])

    for i, image in enumerate(perovstats_object.images):
        # For each image create and save a mask
        fname = image.filename
        im = image.high_pass
        pixel_to_nm_scaling = image.pixel_to_nm_scaling

        # Thresholding config options
        threshold = perovstats_object.config["mask"]["threshold"]
        threshold_func = perovstats_object.config["mask"]["threshold_function"]
        if threshold_func == "mad":
            threshold_func = threshold_mad
        elif threshold_func == "std":
            threshold_func = threshold_mean_std

        # Cleaning config options
        area_threshold = perovstats_object.config["mask"]["cleaning"]["area_threshold"]
        if area_threshold:
            area_threshold = area_threshold / (pixel_to_nm_scaling**2)
            disk_radius = perovstats_object.config["mask"]["cleaning"]["disk_radius_factor"] / pixel_to_nm_scaling
        else:
            disk_radius = None

        # Smoothing config options
        smooth_sigma = perovstats_object.config["mask"]["smoothing"]["sigma"]
        smooth_function = perovstats_object.config["mask"]["smoothing"]["smooth_function"]
        if smooth_function == "gaussian":
            smooth_function = ski.filters.gaussian

        np_mask = create_grain_mask(
            im,
            threshold_func=threshold_func,
            threshold=threshold,
            smooth_sigma=smooth_sigma,
            smooth_function=smooth_function,
            area_threshold=area_threshold,
            disk_radius=disk_radius
        )

        perovstats_object.images[i].mask = np_mask

        # Convert to image format and save
        plt.imsave(output_dir / fname / "images" / f"{fname}_mask.jpg", np_mask)


def split_frequencies(perovstats_object) -> list[np.real]:
    """
    Carry out frequency splitting on a batch of files.

    Parameters
    ----------
    args : list[str], optional
        Arguments.

    Raises
    ------
    ValueError
        If neither `cutoff` nor `cutoff_freq_nm` argument supplied.
    """
    cutoff_freq_nm = perovstats_object.config["freqsplit"]["cutoff_freq_nm"]
    edge_width = perovstats_object.config["freqsplit"]["edge_width"]
    output_dir = Path(perovstats_object.config["output_dir"])

    for image_data in perovstats_object.images:
        filename = image_data.filename

        file_output_dir = Path(output_dir / filename)
        file_output_dir.mkdir(parents=True, exist_ok=True)

        if image_data.image_flattened is not None:
            image = image_data.image_flattened
        else:
            image = image_data.image_original
        pixel_to_nm_scaling = image_data.pixel_to_nm_scaling
        LOGGER.debug("[%s] Image dimensions: ", image.shape)
        LOGGER.info("[%s] : *** Frequency splitting ***", filename)

        if cutoff_freq_nm:
            cutoff = 2 * pixel_to_nm_scaling / cutoff_freq_nm

        LOGGER.info("[%s] : pixel_to_nm_scaling: %s", filename, pixel_to_nm_scaling)

        cutoff, rmses, cutoffs = find_cutoff(
            image,
            edge_width,
            min_cutoff=0,
            max_cutoff=0.2,
            cutoff_step=0.005,
            min_rms=12,
        )

        cutoff_nm = 2 * pixel_to_nm_scaling / cutoff
        print(f"CHOSEN CUTOFF: {cutoff} ({round(cutoff_nm, 3)}nm)")

        # Update image class with chosen cutoff
        image_data.cutoff = cutoff
        image_data.cutoff_freq_nm = cutoff_nm

        high_pass, low_pass = frequency_split(
            image,
            cutoff=cutoff,
            edge_width=edge_width,
        )

        # high_pass, low_pass = open_freq_split_GUI(
        #     filename,
        #     image,
        #     cutoff=cutoff,
        #     edge_width=edge_width,
        # )

        image_data.high_pass = normalise_array(high_pass)
        image_data.low_pass = low_pass
        image_data.file_directory = file_output_dir


        # fig = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        # ((ax_orig, ax_hp), (ax_graph, ax_lp)) = fig[1]
        # ax_orig.imshow(image, cmap='gray')
        # ax_orig.set_title('Original image')
        # ax_orig.axis('off')

        # ax_hp.imshow(high_pass, cmap='gray')
        # ax_hp.set_title("High passed image")
        # ax_hp.axis('off')

        # ax_lp.imshow(low_pass, cmap='gray')
        # ax_lp.set_title("Low passed image")
        # ax_lp.axis('off')

        # ax_graph.plot(cutoffs, rmses, marker='.', color='royalblue', label='highpass filter rms')
        # ax_graph.axhline(y=10, color='red', linestyle='--', label='Hard coded min rms (10)')
        # ax_graph.axvline(x=cutoff, color='green', linestyle='-.', label='Chosen cutoff from hard coded RMS')
        # ax_graph.set_title("RMS roughness to cutoff frequency value")
        # ax_graph.set_xlabel("Cutoff frequency")
        # ax_graph.set_ylabel("RMS value")
        # ax_graph.grid(True, alpha=0.3)
        # ax_graph.legend()

        # plt.show()

        # Convert to image format
        arr = high_pass
        # arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = normalise_array(arr)
        img = Image.fromarray(arr * 255).convert("L")
        img_dir = Path(file_output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img.save(file_output_dir / "images" / f"{filename}_high_pass.jpg")

        arr = low_pass
        # arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = normalise_array(arr)
        img = Image.fromarray(arr * 255).convert("L")
        img.save(file_output_dir / "images" / f"{filename}_low_pass.jpg")

        arr = image_data.image_original
        # arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = normalise_array(arr)
        img = Image.fromarray(arr * 255).convert("L")
        img.save(file_output_dir / "images" / f"{filename}_original.jpg")


def normalise_array(arr):
    v_min, v_max = np.percentile(arr, [0.05, 99.95])

    clipped = np.clip(arr, v_min, v_max)
    normalised = (clipped - v_min) / (v_max - v_min)
    return normalised

# def open_freq_split_GUI(
#         filename,
#         image,
#         cutoff,
#         edge_width,
# ):
#     # Set parameters to default
#     image_params["image"] = image
#     image_params["display_mode"] = 0
#     image_params["cutoff"] = cutoff
#     image_params["edge_width"] = edge_width

#     final_images["high_pass"] = image
#     final_images["low_pass"] = image

#     # Set up window and frames
#     root = tk.Tk()
#     title_frame = tk.Frame(root)
#     title_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
#     image_frame = tk.Frame(root)
#     image_frame.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)
#     options_frame = tk.Frame(root)
#     options_frame.grid(row=1, column=1, sticky="nsew", padx=10)
#     footer_frame = tk.Frame(options_frame)
#     footer_frame.grid(row=7, column=0, sticky="s")

#     # Large label - frequency split - [image name]
#     title_text = f"Frequency Splitter\n({filename})"
#     title = tk.Label(title_frame, text=title_text, font=("Helvetica", 15)).grid(row=0, column=0, sticky="nesw")

#     # Small label - instructions
#     instruction_text = ("Select the best frequency to split the image into a low-pass and high-pass. Aim for the lowest"
#                         + " frequency value that does not show the large curves in the high-pass image.")
#     instructions = tk.Label(
#         title_frame, text=instruction_text, font=("Helvetica", 11), wraplength=600
#     ).grid(row=1, column=0, sticky="nesw")

#     # Matplotlib - image display
#     fig = Figure(figsize=(4,4), dpi=100)
#     plot = fig.add_subplot(111)
#     curr_image = plot.imshow(image, cmap="grey")
#     image_params["curr_image"] = curr_image
#     plot.axis("off")
#     fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     canvas = FigureCanvasTkAgg(fig, master=image_frame)
#     canvas.draw()
#     canvas.get_tk_widget().grid(sticky="nesw")
#     image_params["canvas"] = canvas

#     # Slider 1 - Select/ preview cutoff frequency
#     freq_slider_label = tk.Label(options_frame, text="Cutoff frequency:").grid(row=0, column=0, sticky="nesw")
#     freq_value = tk.DoubleVar(value = image_params["cutoff"])
#     freq_slider = ttk.Scale(
#         options_frame,
#         from_=0,
#         to=0.3,
#         orient="horizontal",
#         command=freq_slider_changed,
#         variable=freq_value
#     ).grid(row=1, column=0, sticky="nesw")
#     freq_slider_value_label = tk.Label(options_frame, textvariable=freq_value).grid(row=0, column=1, sticky="e")

#     # Slider 2 - Select/ preview edge-width
#     edge_width_slider_label = tk.Label(options_frame, text="Edge width:").grid(row=2, column=0, sticky="nesw")
#     edge_width_value = tk.DoubleVar(value = image_params["edge_width"])
#     edge_width_slider = ttk.Scale(
#         options_frame,
#         from_=0,
#         to=0.2,
#         orient="horizontal",
#         command=edge_width_slider_changed,
#         variable=edge_width_value
#     ).grid(row=3, column=0, sticky="nesw", pady=(0, 10))
#     freq_slider_value_label = tk.Label(options_frame, textvariable=edge_width_value).grid(row=2, column=1, sticky="e")

#     # Radio button 1 - display original
#     # Radio button 2 - display high-pass
#     # Radio button 3 - display low-pass
#     v = tk.IntVar(value=0)
#     radio_values = {
#         "Original image": 0,
#         "High-pass": 1,
#         "Low-pass": 2
#     }
#     for (text, value) in radio_values.items():
#         row = 3 + int(value)
#         tk.Radiobutton(options_frame, text=text, variable=v, command=lambda: radio_changed(v.get()), value=value).grid(row=row, column=0, sticky="nw")


#     # Button - Confirm
#     confirm_button = tk.Button(footer_frame, text="Confirm", command=lambda: confirm_split(root)).grid(row=0, column=1, sticky="se")

#     root.columnconfigure(0, weight=1)
#     root.rowconfigure(0, weight=1)
#     title_frame.columnconfigure(0, weight=1)
#     title_frame.rowconfigure(0, weight=1)
#     image_frame.columnconfigure(0, weight=1)
#     image_frame.rowconfigure(0, weight=1)
#     options_frame.columnconfigure(0, weight=1, minsize=250)
#     options_frame.rowconfigure(0, weight=1)
#     footer_frame.columnconfigure(0, weight=1)
#     footer_frame.rowconfigure(0, weight=1)

#     root.mainloop()

#     return final_images["high_pass"], final_images["low_pass"]


# def freq_slider_changed(value):
#     image_params["cutoff"] = float(value)
#     print(image_params.items())
#     update_image(**image_params)


# def edge_width_slider_changed(value):
#     image_params["edge_width"] = float(value)
#     update_image(**image_params)


# def radio_changed(value):
#     image_params["display_mode"] = int(value)
#     update_image(**image_params)


# def update_image(image, curr_image, cutoff, edge_width, display_mode, canvas):
#     high_pass, low_pass = frequency_split(
#         image,
#         cutoff=cutoff,
#         edge_width=edge_width,
#     )
#     img = None
#     if display_mode == 0:
#         img = image
#     elif display_mode == 1:
#         img = high_pass
#     elif display_mode == 2:
#         img = low_pass
#     img = (img - img.min()) / (img.max() - img.min())
#     curr_image.set_data(img)
#     canvas.draw_idle()


# def confirm_split(root):
#     root.destroy()
