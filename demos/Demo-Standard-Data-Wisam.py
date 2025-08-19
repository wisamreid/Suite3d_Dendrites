import os
from datetime import date
from matplotlib import pyplot as plt
import numpy as np

# # we need to set the current path to the directory
# # containing the suite3d repository, this hack should
# # do the trick
# os.chdir(os.path.dirname(os.path.abspath("")))

from suite3d.job import Job
from suite3d import io
from suite3d import plot_utils as plot


def main():
    """
        Run the demo!
    """

    """
        Paths:
    """

    demo_data_base = 'Z:\\\\Wisam\\Z_Python\\Suite3D Data\\'
    demo_data_dir = demo_data_base + 'standard-2p\\'
    output_dir = demo_data_base + 'runs\\'
    # output_dir = r"Z:\Wisam\Z_Python\Suite3D Data\runs"

    """
        Import Data:
    """

    # update this to point to the demo data!
    tifs = io.get_tif_paths(demo_data_dir)[:10]
    for tif in tifs: print(tif)

    """
        Set Parameters:
    """

    # Set the mandatory parameters
    params = {
        # volume rate
        'fs': io.get_vol_rate(tifs[0]),

        # planes to analyze. 0 is typically the flyback, so we exclude it here
        'planes': np.array([1, 2, 3, 4]),
        # number of planes recorded by scanimage, including the flyback
        'n_ch_tif': 5,

        # Decay time of the Ca indicator in seconds. 1.3 for GCaMP6s. This example is for GCamP8m
        'tau': 1.3,
        'lbm': False,
        'num_colors': 2,  # how many color channels were recorded by scanimage
        'functional_color_channel': 0,  # which color channel is the functional one
        # voxel size in z,y,x in microns
        'voxel_size_um': (20, 1.5, 1.5),

        # number of files to use for the initial pass
        # usually, ~500 frames is a good rule of thumb
        # we will just use 200 here for speed
        'n_init_files': 2,

        # 3D GPU registration - fast!
        '3d_reg': True,
        'gpu_reg': True,

        # note : 3D CPU is not supported yet
        'subtract_crosstalk': False,  # turn off some lbm-only features
        'fuse_strips': False,  # turn off some lbm-only features

        # ——— Reference‐image smoothing/blocking ———
        # how big each block is when computing masks & FFTs
        # smaller → less peak memory, but more boundary overhead
        # 'block_size': [128, 128],
        'block_size': [64, 64],

        # how many planes to batch through the GPU when building the reference
        # smaller → less VRAM, but more kernel launch overhead
        # 'gpu_reference_batch_size': 20,
        'gpu_reference_batch_size': 5,

        # require exact frame‐counts so preregistration can’t mis-fire
        'tif_preregistration_safe_mode': True,

        # limit how many TIFFs get loaded at once (reduces host‐RAM spikes)
        # 'tif_batch_size': 2
        'tif_batch_size': 1
    }

    """
        Setup Job
    """

    # (1) Pick a base folder for all your runs.
    #     This can be anywhere you have write access:
    output_base = output_dir

    # (2) Give your run a name (this will create: <output_base>\<job_name>):
    job_name   = "demo-std"

    # (3) tifs is your list of .tif paths; params is the dict of parameters you set up.

    # (4) create=True tells it to mkdir the folder if it doesn’t exist,
    #     overwrite=True lets it delete an old run of the same name,
    #     verbosity=3 turns on detailed logging to the console.
    # Create the job
    job = Job(
        output_base,
        job_name,
        tifs=tifs,
        params=params,
        create=True,
        overwrite=True,
        verbosity=3
    )

    """
       Initialize the job
    """

    # (5) Kick the job off
    job.run_init_pass()
    print("✔︎ init_pass complete; ref_img_3d.shape =", job.summary['ref_img_3d'].shape)

    # NOTE: If you have large tiffs, split the large tiffs into files of size 100 after registration
    job.params['split_tif_size'] = 100

    # OPTIONAL: load and take a look at the reference image
    summary = job.load_summary()
    ref_img = summary['ref_img_3d']

    # # view 1 plane at a time
    # plot.show_img(ref_img[3], figsize=(3,4))

    # # interactive 3D viewer
    # plot.VolumeViewer(ref_img)

    """
        Register Images:
    """

    job.register()

    """
        Compute Correlation Map:
    """

    corr_map = job.calculate_corr_map()
    res = job.load_corr_map_results()
    vmap = res['vmap']

    """
        Segment ROIs:
    """

    job.params['patch_size_xy'] = (550, 550)
    # for speed, only segment a single patch
    job.segment_rois()

    job.compute_npil_masks()

    """
        Extract Traces and Deconvolve:
    """

    traces = job.extract_and_deconvolve()

    """
        Export Results:
    """

    job.export_results(output_dir, result_dir_name='rois')

    # NOTE: To take a look at the outputs in napari, navigate to the suite3d directory in a command shell and run the following:
    #   python curation.py curation --output_dir /path/to/output/rois

if __name__ == "__main__":
    main()
    print("\n")
    print("Done!")