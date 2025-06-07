#!/usr/bin/env python
'''
QI2lab OPM suite
Reconstruction tools

Generate theoretical OPM in skewed coordinates

Last updated: Shepherd 05/25
'''

import psfmodels as psfm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d


# ROI tools
def get_skewed_roi_size(sizes, theta, dc, dstep, ensure_odd=True):
    """
    Calculate the ROI size in the OPM matrix to include sufficient xy and z points.

    Parameters
    ----------
    sizes : list of float
        [z-size, y-size, x-size] in the same units as `dc` and `dstep`.
    theta : float
        Angle in radians.
    dc : float
        Camera pixel size.
    dstep : float
        Step size.
    ensure_odd : bool, optional
        Whether to ensure the ROI dimensions are odd. Default is True.

    Returns
    -------
    list of int
        [n0, n1, n2]: Integer size of ROI in skewed coordinates.
    """

    # x-size determines n2 size
    n2 = int(np.ceil(sizes[2] / dc))

    # z-size determines n1
    n1 = int(np.ceil(sizes[0] / dc / np.sin(theta)))

    # set so that @ top and bottom z-points, ROI includes the full y-size
    n0 = int(np.ceil((0.5 * (n1 + 1)) * dc * np.cos(theta) + sizes[1]) / dstep)

    if ensure_odd:
        if np.mod(n2, 2) == 0:
            n2 += 1

        if np.mod(n1, 2) == 0:
            n1 += 1

        if np.mod(n0, 2) == 0:
            n0 += 1

    return [n0, n1, n2]

# coordinate transformations between OPM and coverslip frames
def get_skewed_coords(sizes, dc, ds, theta, scan_direction="lateral"):
    """
    Get laboratory coordinates (i.e., coverslip coordinates) for a stage-scanning OPM set.

    Parameters
    ----------
    sizes : tuple of int
        A tuple (n0, n1, n2) where:
        - n0: Number of images (nimgs).
        - n1: Number of camera pixels in the y-direction (ny_cam).
        - n2: Number of camera pixels in the x-direction (nx_cam).
    dc : float
        Camera pixel size.
    ds : float
        Stage step size.
    theta : float
        Angle in radians.
    scan_direction : {'lateral', 'axial'}, optional
        Direction of the scan. Must be either 'lateral' or 'axial'. Default is 'lateral'.

    Returns
    -------
    x : ndarray
        X-coordinates of the laboratory frame.
    y : ndarray
        Y-coordinates of the laboratory frame.
    z : ndarray
        Z-coordinates of the laboratory frame.

    Raises
    ------
    ValueError
        If `scan_direction` is not 'lateral' or 'axial'.

    Notes
    -----
    - For `scan_direction='lateral'`, the stage moves laterally, and the y-coordinates are computed
      based on the stage position and the camera pixel size.
    - For `scan_direction='axial'`, the stage moves axially, and the z-coordinates are computed
      based on the stage position and the camera pixel size.
    """
   
    nimgs, ny_cam, nx_cam = sizes

    if scan_direction == "lateral":
        x = dc * np.arange(nx_cam)[None, None, :]
        # y = stage_pos[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        y = ds * np.arange(nimgs)[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        z = dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
    elif scan_direction == "axial":
        x = dc * np.arange(nx_cam)[None, None, :]
        y = dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        z = ds * np.arange(nimgs)[:, None, None] + dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
    else:
        raise ValueError("scan_direction must be `lateral` or `axial` but was `%s`" % scan_direction)

    return x, y, z

def create_psf_silicone_100x(
    dxy: float, 
    dz: float, 
    nxy: float, 
    nz: float, 
    em_wvl: float, 
    pz: float
) -> np.ndarray:
    """
    Create OPM PSF in coverslip coordinates.

    Parameters
    ----------
    dxy : float
        Spacing of xy pixels.
    dz : float
        Spacing of z planes.
    nxy : int
        Number of xy pixels on a side.
    nz : int
        Number of z planes.
    em_wvl : float
        Emission wavelength in microns.
    pz : float
        Microns above coverslip.

    Returns
    -------
    tot_psf : ndarray
        Theoretical PSF in coverslip coordinates.
    """

    silicone_lens = {

        'ni0': 1.4, # immersion medium RI design value
        'ni': 1.4,  # immersion medium RI experimental value
        'tg0': 170, # microns, coverslip thickness design value
        'tg': 170,  # microns, coverslip thickness
        'ns': 1.38,  # specimen refractive index
        'ti0': 300,
        #'nxy': nxy,
        #'dxy': dxy,
        #'wvl': em_wvl,
        #'pz': pz
    }
    #ex_lens = {**silicone_lens, 'NA': ex_NA}
    em_lens = {**silicone_lens, 'NA': 1.35}

    # # The psf model to use
    # # can be any of {'vectorial', 'scalar', or 'microscpsf'}
    # func = 'vectorial'

    # # the main function
    # _, _, tot_psf = psfm._core.tot_psf(nx=nxy, nz=nz, dxy=dxy, dz=dz, 
    #                                     pz = pz, x_offset=0, z_offset=0,
    #                                     ex_wvl = ex_wvl, em_wvl = em_wvl,
    #                                     ex_params=ex_lens, em_params=em_lens,
    #                                     psf_func=func)

    lim = (nz - 1) * dz / 2
    zv = np.linspace(-lim + pz, lim + pz, nz)
    
    psf = psfm.vectorial_psf(zv=zv,nx=nxy,dxy=dxy,pz=pz,wvl=em_wvl,params=em_lens)
    psf = psf / np.sum(psf)
    x = np.linspace(1, psf.shape[1], psf.shape[1])
    x = x - psf.shape[1] / 2
    y = np.linspace(1, psf.shape[2], psf.shape[2])
    y = y - psf.shape[2] / 2
    x, y = np.meshgrid(x, y)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    sigma = 10
    filter = np.exp(-np.power(r, 2) / (2 * sigma**2))
    filter = filter / np.max(filter)

    psf = psf * filter
    
    return psf

def create_psf_water_40x(
    dxy: float, 
    dz: float, 
    nxy: float, 
    nz: float, 
    em_wvl: float, 
    pz: float
) -> np.ndarray:
    """
    Create OPM PSF in coverslip coordinates.

    Parameters
    ----------
    dxy : float
        Spacing of xy pixels.
    dz : float
        Spacing of z planes.
    nxy : int
        Number of xy pixels on a side.
    nz : int
        Number of z planes.
    em_wvl : float
        Emission wavelength in microns.
    pz : float
        Microns above coverslip.

    Returns
    -------
    tot_psf : ndarray
        Theoretical PSF in coverslip coordinates.
    """

    water_lens = {

        'ni0': 1.33, # immersion medium RI design value
        'ni': 1.33,  # immersion medium RI experimental value
        'tg0': 170, # microns, coverslip thickness design value
        'tg': 170,  # microns, coverslip thickness
        'ns': 1.33,  # specimen refractive index
        'ti0': 620,
        #'nxy': nxy,
        #'dxy': dxy,
        #'wvl': em_wvl,
        #'pz': pz
    }
    #ex_lens = {**silicone_lens, 'NA': ex_NA}
    em_lens = {**water_lens, 'NA': 1.05}

    # # The psf model to use
    # # can be any of {'vectorial', 'scalar', or 'microscpsf'}
    # func = 'vectorial'

    # # the main function
    # _, _, tot_psf = psfm._core.tot_psf(nx=nxy, nz=nz, dxy=dxy, dz=dz, 
    #                                     pz = pz, x_offset=0, z_offset=0,
    #                                     ex_wvl = ex_wvl, em_wvl = em_wvl,
    #                                     ex_params=ex_lens, em_params=em_lens,
    #                                     psf_func=func)

    lim = (nz - 1) * dz / 2
    zv = np.linspace(-lim + pz, lim + pz, nz)
    
    psf = psfm.vectorial_psf(zv=zv,nx=nxy,dxy=dxy,pz=pz,wvl=em_wvl,params=em_lens)
    psf = psf / np.sum(psf)
    x = np.linspace(1, psf.shape[1], psf.shape[1])
    x = x - psf.shape[1] / 2
    y = np.linspace(1, psf.shape[2], psf.shape[2])
    y = y - psf.shape[2] / 2
    x, y = np.meshgrid(x, y)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    sigma = 10
    filter = np.exp(-np.power(r, 2) / (2 * sigma**2))
    filter = filter / np.max(filter)

    psf = psf * filter
    
    return psf

def generate_proj_psf(
    em_wvl: float,
    pixel_size_um: float = 0.115,
    pz : float = 15.0,
    plot=False
):
    
    silicone_lens = {

        'ni0': 1.4, # immersion medium RI design value
        'ni': 1.4,  # immersion medium RI experimental value
        'tg0': 170, # microns, coverslip thickness design value
        'tg': 170,  # microns, coverslip thickness
        'ns': 1.38,  # specimen refractive index
        'ti0': 300,
    }
    #ex_lens = {**silicone_lens, 'NA': ex_NA}
    em_lens = {**silicone_lens, 'NA': 1.35}

    psf = psfm.vectorial_psf(zv=0,nx=51,dxy=pixel_size_um,pz=pz,wvl=em_wvl,params=em_lens)
    psf = psf / np.sum(psf)
    if psf.ndim == 2:
        psf = psf[np.newaxis, :, :]
    x = np.linspace(1, psf.shape[1], psf.shape[1])
    x = x - psf.shape[1] / 2
    y = np.linspace(1, psf.shape[2], psf.shape[2])
    y = y - psf.shape[2] / 2
    x, y = np.meshgrid(x, y)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    sigma = 10
    filter = np.exp(-np.power(r, 2) / (2 * sigma**2))
    filter = filter / np.max(filter)

    psf = psf * filter
    
    return psf

def generate_skewed_psf(
    em_wvl: float,
    pixel_size_um: float = 0.115,
    scan_axis_step_um: float = 0.4,
    pz : float = 0.0,
    plot=False):
    """
    Create OPM PSF in skewed coordinates.
    
    Parameters
    ----------
    em_wvl: float
        emission wavelength in microns 
    pixel_size_um: float
        pixel size in microns
    scan_axis_step_um: float
        step size in microns
    pz: float
        distance above coverslip in microns
    plot: bool
        whether to plot the PSF
    
    Returns
    -------
    skewed_psf: np.ndarray
        OPM PSF in skewed coordinates
    """

    dc = pixel_size_um
    na = 1.35
    ni = 1.4
    dstage = scan_axis_step_um
    theta = 30 * np.pi/180

    xy_res = 1.6163399561827614 / np.pi * em_wvl / na
    z_res = 2.355*(np.sqrt(6) / np.pi * ni * em_wvl / na ** 2)
    z_res_mod = z_res + np.divide(pz+1e-12,15) / 5
    
    roi_skewed_size = get_skewed_roi_size([z_res_mod * 12, xy_res * 12, xy_res * 12],
                                                          theta, dc, dstage, ensure_odd=True)
    # make square
    roi_skewed_size[2]= roi_skewed_size[1]

    # get tilted coordinates
    x, y, z = get_skewed_coords(roi_skewed_size, dc, dstage, theta)
    dx = x[0, 0, 1] - x[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    dz = z[0, 1, 0] - z[0, 0, 0]

    z -= z.mean()
    x -= x.mean()
    y -= y.mean()

    # get on grid of coordinates
    dxy = 0.5 * np.min([dx, dy])
    dz = 0.5 * dz
    
    nxy = np.max([int(2 * ((x.max() - x.min()) // dxy) + 1),
                  int(2 * ((y.max() - y.min()) // dxy) + 1)])
    nz = z.size

    xg = np.arange(nxy) * dxy
    xg -= xg.mean()
    yg = np.arange(nxy) * dxy
    yg -= yg.mean()


    psf_grid = create_psf_silicone_100x(dxy, dz, nxy, nz, em_wvl, pz)

    if plot:
    # plot gridded psf
        figh1 = plt.figure()
        ax11 = plt.subplot(2, 2, 1) # noqa: F841
        plt.imshow(psf_grid[psf_grid.shape[0]//2])

        ax12 = plt.subplot(2, 2, 2) # noqa: F841
        plt.imshow(psf_grid[:, psf_grid.shape[1]//2, :])

        ax13 = plt.subplot(2, 2, 3) # noqa: F841
        plt.imshow(psf_grid[:, :, psf_grid.shape[2]//2])

    # get value from interpolation
    skewed_psf = np.zeros(roi_skewed_size)
    for ii in range(nz):
        skewed_psf[:, ii, :] = interp2d(xg, yg, psf_grid[ii], kind="linear")(x.ravel(), y[:, ii].ravel())
    
    if plot:
        # plot gridded psf
        figh2 = plt.figure()
        ax21 = plt.subplot(2, 2, 1) # noqa: F841
        plt.imshow(skewed_psf[skewed_psf.shape[0]//2])

        ax22 = plt.subplot(2, 2, 2) # noqa: F841
        plt.imshow(skewed_psf[:, skewed_psf.shape[1]//2, :])

        ax23 = plt.subplot(2, 2, 3) # noqa: F841
        plt.imshow(skewed_psf[:, :, skewed_psf.shape[2]//2])

        figh1.show()
        figh2.show()
    plt.show()

    return skewed_psf

def ASI_generate_skewed_psf(
    em_wvl: float,
    pixel_size_um: float = 0.195,
    scan_axis_step_um: float = 0.4,
    theta_deg = 38,
    pz : float = 0.0,
    plot=False):
    """
    Create OPM PSF in skewed coordinates.
    
    Parameters
    ----------
    em_wvl: float
        emission wavelength in microns 
    pixel_size_um: float
        pixel size in microns
    scan_axis_step_um: float
        step size in microns
    pz: float
        distance above coverslip in microns
    plot: bool
        whether to plot the PSF
    
    Returns
    -------
    skewed_psf: np.ndarray
        OPM PSF in skewed coordinates
    """

    dc = pixel_size_um
    na = 1.1
    ni = 1.33
    dstage = scan_axis_step_um
    theta = theta_deg * np.pi/180

    xy_res = 1.6163399561827614 / np.pi * em_wvl / na
    z_res = 2.355*(np.sqrt(6) / np.pi * ni * em_wvl / na ** 2)
    z_res_mod = z_res + np.divide(pz+1e-12,15) / 5
    
    roi_skewed_size = get_skewed_roi_size([z_res_mod * 12, xy_res * 12, xy_res * 12],
                                                          theta, dc, dstage, ensure_odd=True)
    # make square
    roi_skewed_size[2]= roi_skewed_size[1]

    # get tilted coordinates
    x, y, z = get_skewed_coords(roi_skewed_size, dc, dstage, theta)
    dx = x[0, 0, 1] - x[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    dz = z[0, 1, 0] - z[0, 0, 0]

    z -= z.mean()
    x -= x.mean()
    y -= y.mean()

    # get on grid of coordinates
    dxy = 0.5 * np.min([dx, dy])
    dz = 0.5 * dz
    
    nxy = np.max([int(2 * ((x.max() - x.min()) // dxy) + 1),
                  int(2 * ((y.max() - y.min()) // dxy) + 1)])
    nz = z.size

    xg = np.arange(nxy) * dxy
    xg -= xg.mean()
    yg = np.arange(nxy) * dxy
    yg -= yg.mean()


    psf_grid = create_psf_water_40x(dxy, dz, nxy, nz, em_wvl, pz)

    if plot:
    # plot gridded psf
        figh1 = plt.figure()
        ax11 = plt.subplot(2, 2, 1) # noqa: F841
        plt.imshow(psf_grid[psf_grid.shape[0]//2])

        ax12 = plt.subplot(2, 2, 2) # noqa: F841
        plt.imshow(psf_grid[:, psf_grid.shape[1]//2, :])

        ax13 = plt.subplot(2, 2, 3) # noqa: F841
        plt.imshow(psf_grid[:, :, psf_grid.shape[2]//2])

    # get value from interpolation
    skewed_psf = np.zeros(roi_skewed_size)
    for ii in range(nz):
        skewed_psf[:, ii, :] = interp2d(xg, yg, psf_grid[ii], kind="linear")(x.ravel(), y[:, ii].ravel())
    
    if plot:
        # plot gridded psf
        figh2 = plt.figure()
        ax21 = plt.subplot(2, 2, 1) # noqa: F841
        plt.imshow(skewed_psf[skewed_psf.shape[0]//2])

        ax22 = plt.subplot(2, 2, 2) # noqa: F841
        plt.imshow(skewed_psf[:, skewed_psf.shape[1]//2, :])

        ax23 = plt.subplot(2, 2, 3) # noqa: F841
        plt.imshow(skewed_psf[:, :, skewed_psf.shape[2]//2])

        figh1.show()
        figh2.show()
    plt.show()

    return skewed_psf