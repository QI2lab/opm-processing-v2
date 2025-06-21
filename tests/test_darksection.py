from opm_processing.imageprocessing.darksection import dark_sectioning
from tifffile import imread
from pathlib import Path
import numpy as np
import napari

def main():
    rawdata_path = Path("/home/qi2lab/Downloads/Mito/Mito_525nm_65nm_1.49NA.tif")
    data = imread(rawdata_path)
    darksection_data = np.zeros((9,512,512),dtype=np.uint16)
    for i in range(9):
        darksection_data[i,:] = dark_sectioning(
            input_image = data[i,:],
            emwavelength = 525,
            na = 1.49,
            pixel_size = 65,
            factor = 2
        )
    processed_path = Path("/home/qi2lab/Downloads/Mito/Result/Mito_Darkraw.tif")
    processed_data = imread(processed_path)
    viewer = napari.Viewer()
    viewer.add_image(darksection_data,name='qi2lab')
    viewer.add_image(processed_data,name='paper')
    viewer.add_image(data,name='raw')
    napari.run()

if __name__ == "__main__":
    main()