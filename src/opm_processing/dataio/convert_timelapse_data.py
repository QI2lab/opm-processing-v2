import json
from pathlib import Path

import numpy as np
import tensorstore as ts
import yaml
from tifffile import TiffWriter

from opm_processing.dataio.metadata import find_key


def save_raw_with_yaml(data_array, output_path):
    print("Saving Yaml")
    if output_path.suffix != '.raw':
        print('Output path is not a .raw, creating one now...')
        output_path = output_path.parent / (output_path.stem + '.raw')
    yml_path = output_path.parent / (output_path.stem + '.yaml')
    print(data_array.shape)
    data_array.tofile(output_path)

    meta = {
        "Frames": data_array.shape[0],
        "Data Type": str(np.dtype('uint16')),
        "Height": data_array.shape[1],
        "Width": data_array.shape[2],
        "Byte Order": "<"
    }
    with open(yml_path, "w") as yf:
        yaml.safe_dump(meta, yf, sort_keys=False)


def save_time_projection(data_array, pixel_size_um, output_path):
    # time_projection = np.max(data_array, axis=0)
    # Convert to photon counts
    data_array = (data_array.astype(np.float32) - 100) * 0.24
    data_array = np.clip(data_array, 0, 2**16 -1)
    time_projection = np.mean(data_array, axis=0)
    time_projection = time_projection.astype(np.uint16)

    # save time projection as tiff
    if output_path.suffix != '.tiff':
        print('Output path is not a .tiff, creating one now...')
        output_path = output_path.parent / (output_path.stem + '.tiff')

    axes = 'YX'
    with TiffWriter(output_path, bigtiff=True) as tif:
            metadata={
                'axes': axes,
                'SignificantBits': 16,
                'PhysicalSizeX': pixel_size_um,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': pixel_size_um,
                'PhysicalSizeYUnit': 'µm'
            }
            options = dict(
                compression='zlib',
                compressionargs={'level': 8},
                predictor=True,
                photometric='minisblack',
                resolutionunit='CENTIMETER',
            )
            tif.write(
                time_projection,
                resolution=(
                    1e4 / pixel_size_um,
                    1e4 / pixel_size_um
                ),
                **options,
                metadata=metadata
            )
    

def save_as_tiff(data_array, pixel_size_um, output_path):
    
    # save time projection as tiff
    if output_path.suffix != '.tiff':
        print('Output path is not a .tiff, creating one now...')
        output_path = output_path.parent / (output_path.stem + '.tiff')

    axes = 'TYX'
    with TiffWriter(output_path, bigtiff=True) as tif:
        metadata={
            'axes': axes,
            'SignificantBits': 16,
            'PhysicalSizeX': pixel_size_um,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixel_size_um,
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZ': 1.0,
            'PhysicalSizeZUnit': 'µm',
        }
        options = dict(
            compression='zlib',
            compressionargs={'level': 8},
            predictor=True,
            photometric='minisblack',
            resolutionunit='CENTIMETER',
        )
        tif.write(
            data_array,
            resolution=(
                1e4 / pixel_size_um,
                1e4 / pixel_size_um
            ),
            **options,
            metadata=metadata
        )

data_dirs = [
    Path(r"E:\20251106_DNApaint\20251106_204843_50pM_timelapse_restarted"), 
    Path(r"E:\20251106_DNApaint\20251107_112433_80pM_timelapse") 
]

zarr_dir = Path(r'e:\20251218_beads_glyc\20251218_091258_testrun\testrun.zarr')
big_tiff_path = zarr_dir.parent / 'timelapse.tiff'
batch_processing = False

# zarr_dir = None
mirror_pos_range = None
stage_pos_range = None 
fov_range = [300, 1000]
time_range = None
create_raw = False
create_time_projection = False
create_tiff = True

try:
    ds_spec = {
        "driver" : "zarr3",
        "kvstore" : {
            "driver" : "file",
            "path" : str(zarr_dir)
        }
    }
    datastore = ts.open(ds_spec).result()
except ValueError:
    try:
        ds_spec = {
            "driver" : "zarr",
            "kvstore" : {
                "driver" : "file",
                "path" : str(zarr_dir)
            }
        }
        datastore = ts.open(ds_spec).result()  
    except Exception as e:
        print(f"Error opening datastore: {e}")
        raise

# Read metadata
zattrs_path = zarr_dir / Path(".zattrs")
with open(zattrs_path, "r") as f:
    zattrs = json.load(f)

pixel_size_um = float(find_key(zattrs,"pixel_size_um"))

data_array = np.squeeze(
    datastore[:, :, :, :, :, fov_range[0]:fov_range[1]].read().result()
).astype(np.uint16)

if create_tiff:
    print("Saving timelapse as TIFF")
    save_as_tiff(data_array, pixel_size_um, big_tiff_path)


if batch_processing and data_dirs:
    zarr_dirs = []
    # Batch processing over all directories
    if data_dirs:
        for data_dir in data_dirs:
            # Directory parsing for batch applications
            for dir in data_dir.iterdir():
                if dir.is_dir():
                    if dir.is_dir() and dir.name.endswith("zarr"):
                        zarr_dirs.append(dir)
    # Process single directory
    elif zarr_dir is not None:
        zarr_dirs.append(zarr_dir)
    else:
        raise ValueError("No data directories or zarr directory specified.")

    for zarr_dir in zarr_dirs:
        print(f"Processing {zarr_dir}...")
        
        # Read metadata
        zattrs_path = zarr_dir / Path(".zattrs")
        with open(zattrs_path, "r") as f:
            zattrs = json.load(f)

        pixel_size_um = float(find_key(zattrs,"pixel_size_um"))
        try:
            output_dir_path = zarr_dir.parent / Path("converted_files")
        except Exception:
            print("Error creating tiff directory path.")

        output_dir_path.mkdir(exist_ok=True)

        # Create new file paths 
        raw_file_fname = 'timelapse.raw'
        time_proj_fname = 'time_mean_projection.tiff'
        big_tiff_fname = 'timeplapse.tiff'

        # Try opening datastore
        try:
            ds_spec = {
                "driver" : "zarr3",
                "kvstore" : {
                    "driver" : "file",
                    "path" : str(zarr_dir)
                }
            }
            datastore = ts.open(ds_spec).result()
        except ValueError:
            try:
                ds_spec = {
                    "driver" : "zarr",
                    "kvstore" : {
                        "driver" : "file",
                        "path" : str(zarr_dir)
                    }
                }
                datastore = ts.open(ds_spec).result()  
            except Exception as e:
                print(f"Error opening datastore: {e}")
                raise
        
        # Define the axis order based on datastore shape
        num_scan_position = datastore.shape[3]
        num_channels = datastore.shape[2]
        num_timepoints = datastore.shape[0]
        num_stage_positions = datastore.shape[1]
        print(
            f"Number of scan positions: {num_scan_position}\n"
            f"Number of channels: {num_channels}\n"
            f"Number of timepoints: {num_timepoints}\n"
            f"Number of stage positions: {num_stage_positions}\n"
            f"FOV size in pixels: {datastore.shape[-2]}, {datastore.shape[-1]}\n"
        )   

        if num_scan_position > 1:
            if num_channels == 1:
                axes = "TZYX"
            else:
                axes = "TCZYX"
        elif num_scan_position == 1:
            if num_channels == 1:
                axes = "TYX"
            else:
                axes = "TCYX"

        # Save each position as a separate file
        if mirror_pos_range is not None:
            pos_iterator = range(mirror_pos_range[0], mirror_pos_range[1])
        else:
            pos_iterator = range(num_scan_position)

        if stage_pos_range is not None:
            stage_pos_iterator = range(stage_pos_range[0], stage_pos_range[1])
        else:
            stage_pos_iterator = range(num_stage_positions)
        
        for pos_idx in stage_pos_iterator:
            for scan_idx in pos_iterator:
                print(f"Processing position {pos_idx}, scan {scan_idx}...")
                # create unique file path
                raw_file_path = output_dir_path / ('pos_' + str(pos_idx) + 'scan_' + str(scan_idx) + '_' + raw_file_fname)
                time_proj_path = output_dir_path / ('pos_' + str(pos_idx) + 'scan_' + str(scan_idx) +  '_' + time_proj_fname)
                big_tiff_path = output_dir_path / ('pos_' + str(pos_idx) + 'scan_' + str(scan_idx) +  '_' + big_tiff_fname)

                # load current position as an array
                if time_range is None:
                    time_range = [0, num_timepoints]
                if fov_range is None:
                    fov_range = [0, datastore.shape[-1]]

                current_pos_arr = np.squeeze(
                    datastore[time_range[0]:time_range[1], pos_idx, :, scan_idx, :, fov_range[0]:fov_range[1]].read().result()
                ).astype(np.uint16)

                # Save raw file
                if create_raw:
                    print("Creating RAW")
                    save_raw_with_yaml(current_pos_arr, raw_file_path)

                # Save time proejection
                if create_time_projection:
                    print("Creating time-projection")
                    save_time_projection(current_pos_arr, pixel_size_um, time_proj_path)

                # Save timelapse as tiff
                if create_tiff:
                    print("Saving timelapse as TIFF")
                    save_as_tiff(current_pos_arr, pixel_size_um, big_tiff_path)
else:
    print("Single directory processing...")