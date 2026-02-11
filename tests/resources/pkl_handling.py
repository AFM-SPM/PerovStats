import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# grain mask

def save_data_pkl():
    # Get data and prepare for pkl
    image_to_slice = Image.open("./output/4_c60_perovonsil_ref_10um.PFQNM/images/4_c60_perovonsil_ref_10um.PFQNM_mask.jpg")
    array_to_slice = np.asarray(image_to_slice)

    sliced_array = array_to_slice[0:50, 0:50]

    # Save .pkl to tests/resources/
    with open('./tests/resources/small_mask.pkl', 'wb') as file:
        pickle.dump(sliced_array, file)


def display_arr_pkl():
    with open('tests/resources/small_mask.pkl', 'rb')as file:
        arr = pickle.load(file)
        plt.imshow(arr, cmap="gray")
        plt.show()

display_arr_pkl()
