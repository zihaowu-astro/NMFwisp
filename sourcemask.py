#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""sourcemask.py

Stack images and create a segmentation map, then reproject the segmentation map
to the WCS of each individual frame. This is used to create source masks for
each frame, which can be used for background estimation and other purposes. The
segmentation map can be created on the fly from a list of (LW) image names, or
it can be read from an existing file. The script uses drizzle to mosaic the
images and photutils to detect sources in the mosaic. The resulting segmentation
map is then reprojected to the WCS of each individual frame and saved as a new
FITS file.
"""

# --------------
# import modules
# --------------

import os, glob, logging
import argparse
import numpy as np
from scipy.ndimage import binary_dilation

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Gaussian2DKernel
from photutils.segmentation import detect_sources

from drizzle.drizzle import Drizzle
from reproject import reproject_interp

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

from jwst import datamodels
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
#DO_NOT_USE = 1<<0

# --------------
# methods for finding optimal WCS for a group of images.
# this is adapted from code in reproject.
# --------------

def find_optimal_celestial_wcs(image_names, hdu_in="SCI", **extras):

    # find the corners of each image in sky coordinates,
    # and flatten to get lists of ras and decs
    boxes = []
    for f in image_names:
        hdr = fits.getheader(f, extname=hdu_in)
        boxes.append(header_box(hdr))
    boxes = np.array(boxes)
    ras = boxes[:, 0,:].flatten()
    decs = boxes[:, 1,:].flatten()

    # make dummy tangent plane with 60mas pixels, centered on the median position of the corners
    ram = np.median(ras)
    decm = np.median(decs)
    tp = WCS(info_to_header(ra=ram, dec=decm, PS=0.06))

    # get pixel coordinates of corners in this plane,
    xx, yy = tp.world_to_pixel_values(ras, decs)
    # compute minimum bounding rotated rectange in TP coordinates and mosaic coordinates.
    # also get rotation matrix and angle of rotation
    pixbox, rot, bbox, theta = minimum_bounding_rectangle(np.array([xx, yy]).T)
    # get size of the of the rectangle
    nx = bbox[0] - bbox[1]
    ny = bbox[2] - bbox[3]
    # convert to sky coordinates
    skybox = np.array(tp.pixel_to_world_values(pixbox))

    # center of the bounding box, in TP coordinates
    crpix = np.dot([bbox[1] + nx/2, bbox[3] + ny/2], rot)
    # center of the box in sky coordinates
    crval = np.array(tp.pixel_to_world_values(*crpix))
    # make a dummy rotated header
    hdr = info_to_header(ra=crval[0], dec=crval[1],
                         PA=theta, PS=0.06)
    # roughly set where the center is and how big the image is;
    # this is not really necessary and will be refined below
    hdr["CRPIX1"] = nx/2
    hdr["CRPIX2"] = ny/2
    hdr["NAXIS1"] = int(nx)
    hdr["NAXIS2"] = int(ny)

    # Translate and expand to be sure to cover the hull, even in the case
    # of a mistake in rotation angle or an axis flip
    wnew = WCS(hdr)
    c = np.array(wnew.world_to_pixel_values(skybox))
    xmin, xmax = int(np.floor(c[:, 0].min()))-1, int(np.ceil(c[:, 0].max()))+1
    ymin, ymax = int(np.floor(c[:, 1].min()))-1, int(np.ceil(c[:, 1].max()))+1
    dx, dy = xmax - xmin, (ymax - ymin)
    dx = int(np.ceil(dx))
    dy = int(np.ceil(dy))

    hdr["CRPIX1"] -= xmin
    hdr["CRPIX2"] -= ymin
    hdr["NAXIS1"] = dx
    hdr["NAXIS2"] = dy

    # Check that the all the corners are covered.
    wnew = WCS(hdr)
    c = np.array(wnew.world_to_pixel_values(ras, decs)).T
    assert np.all(c > 0)
    assert np.all(c[:, 0] < hdr["NAXIS1"])
    assert np.all(c[:, 1] < hdr["NAXIS2"])

    return wnew, (hdr["NAXIS1"], hdr["NAXIS2"])


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    Parameters
    ----------
    points: an nx2 matrix of coordinates
        Coordinates to bound.

    Returns
    -------
    rval : an nx2 matrix of coordinates
        Coordinates of the corners of the bounding box
     r : 2x2 rotation matrix
        The rotation matrix that was applied to the points to get the bounding box
     bbox : list of floats
        The bounding box in the rotated, axis-aligned space [x_max, x_min, y_max, y_min]
     theta : float
        The angle of the rotation in degrees
    """
    from scipy.spatial import ConvexHull
    pi2 = np.pi/2.

    # get the convex hull for the points; these are in CCW order
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2)) # 90 degree rotations aren't interesting
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull; these coordinates are in the rotated space
    # with the box axes now aligned with the coordinate axes.
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding coordinates for each rotation
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # get the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]
    a = angles[best_idx]

    # de-rotate the corners of the bounding box to get the coordinates in the original space
    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, r, [x1, x2, y1, y2], np.rad2deg(a)


def header_box(hdr):
    """Get the bbox from the WCS and NAXIS in the header

    Returns
    -------
    box : ndarray of floats of shape ()
    """
    wcs = WCS(hdr)
    corners = [(-0.5, -0.5),
               (hdr["NAXIS1"]+0.5, -0.5),
               (hdr["NAXIS1"]+0.5, hdr["NAXIS2"]+0.5),
               (-0.5, hdr["NAXIS2"]+0.5)]
    corners = np.array(corners)
    sky_corners = wcs.pixel_to_world_values(corners)
    return np.array(sky_corners).T


def rotation_matrix(theta):
    costheta, sintheta = np.cos(theta), np.sin(theta)
    return np.array([[costheta, -sintheta],
                     [sintheta, costheta]])


def info_to_header(ra=None, dec=None,
                   PA=0, PS=0.06,
                   **kwargs):
    """Make a simple header with some info.
    """
    hdr = {"CTYPE1":'RA---TAN-SIP',
           "CTYPE2":'DEC--TAN-SIP',
           "CRPIX1": 1024.5,
           "CRPIX2": 1024.5,
           "NAXIS1": 2048,
           "NAXIS2": 2048}

    reflection = np.array([[-1, 0],[0, 1]]) # because N is up, E to the left
    theta = PA
    T = rotation_matrix(np.deg2rad(theta))
    cd = np.eye(2) * PS / 3600
    cd = np.matmul(reflection, np.matmul(T, cd))
    hdr["CD1_1"] = cd[0, 0]
    hdr["CD1_2"] = cd[0, 1]
    hdr["CD2_2"] = cd[1, 1]
    hdr["CD2_1"] = cd[1, 0]
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec

    return hdr


# --------------
# methods for making segmap and projecting to individual frames
# --------------

def drizzle_images(images, wcs=None, wt_scl="exptime",
                   shape=None):
    """Use drizzle to mosaic; by default weights by exposure time, but pixel
    weights are included if VAR_RNOISE or VAR_BKG extension is available.

    Parameters
    ----------
    images : list of str
        List of image names (full paths) to stack. These should be background homogenized,
        astrometrically aligned, and have a celestial WCS.  Bad pixels may be
        marked in a DQ extension, or by NaNs in the weight extensions.
    wcs : astropy.wcs.WCS, optional
        If None, find the optimal WCS that covers all the input images.
        Otherwise, use the provided WCS for the output mosaic.
    wt_scl : str, optional
        The keyword in the header to use for scaling the weights. Default is "exptime".
    shape : tuple of int, optional
        The shape of the output mosaic. If None, it will be determined from the WCS.
        This is only used if wcs is provided; if wcs is None, the shape will be determined
        from the optimal WCS.

    Returns
    -------
    outsci : 2D numpy array
        The drizzled mosaic image.
    outwht : 2D numpy array
        The drizzled weight map.
    wcs : astropy.wcs.WCS
        The WCS of the output mosaic.
    """
    if wcs is None:
        wcs, shape = find_optimal_celestial_wcs(images,
                                                hdu_in="SCI",
                                                auto_rotate=True)
        wcs.pixel_shape = shape

    assert wcs is not None
    driz = Drizzle(outwcs=wcs, wt_scl=wt_scl)
    for infile in images:
        im = fits.open(infile)
        sci = im["SCI"].data
        iwcs = WCS(im["SCI"].header)
        if "VAR_BKG" in im:
            wht = 1/ im["VAR_BKG"].data
            wht[np.isnan(wht)] = 0
        elif "VAR_RNOISE" in im:
            wht = 1/ im["VAR_RNOISE"].data
            wht[np.isnan(wht)] = 0
        else:
            wht = np.ones(im.shape)

        if "DQ" in im:
            bad = (im["DQ"].data & DO_NOT_USE) == DO_NOT_USE
            wht[bad] = 0.0

        driz.add_image(sci, iwcs, wht)
        im.close()

    return driz.outsci, driz.outwht, wcs


def make_segmentation(data, mask=None, sigma=3.0):
    """Make a basic segmentation map from the input data, using sigma-clipped
    stats to estimate the background and noise, and then convolving with a
    Gaussian kernel to smooth the data before source detection.
    """
    mean, _, std = sigma_clipped_stats(data, mask=mask)
    kernel = Gaussian2DKernel(1.0, x_size=7, y_size=7)
    convolved_data = convolve(data-mean, kernel)
    seg = detect_sources(convolved_data, sigma * std, npixels=10)

    return seg


def images_to_seg(det_images):
    """In the case a full segmap does not exist, make one on the fly from a list
    of (LW) image names.

    Parameters
    ----------
    det_images : list of str
        List of image names (full paths) to stack and make the segmap from.
        These should be (all) the LW images, which are not as affected by stray
        light and thus better for source detection. Images are assumed to be
        background homogenized, astrometrically aligned, and have a celestial
        WCS.

    Returns
    -------
    segmap_hdul : astropy.io.fits.HDUList
        HDUList containing the segmentation map in an extension named "SEGMAP",
        and the stacked image and weight map in extensions named "SCI" and "WHT".
    stack : 2D numpy array
        The stacked image used to make the segmap, for diagnostic purposes.
    """
    stack, wht, wcs = drizzle_images(det_images)
    seg = make_segmentation(stack)


    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(seg, name="SEGMAP"),
                         fits.ImageHDU(stack.astype(np.float32), name="SCI"),
                         fits.ImageHDU(wht.astype(np.float32), name="WHT")])
    [hdu.header.update(wcs.to_header(relax=True)) for hdu in hdul]

    return hdul, stack


def get_wcs(imname, use_gwcs=True):
    if use_gwcs:
        im = datamodels.ImageModel(imname)
        gwcs = im.meta.wcs
        hdr = gwcs.to_fits_sip(max_pix_error=0.04,
                               max_inv_pix_error=0.04,
                               npoints=32)
    else:
        hdr = fits.getheader(imname, "SCI")

    wcs = WCS(hdr)
    shape = hdr["NAXIS1"], hdr["NAXIS2"]
    return wcs, shape


def project_segmap(segmap_hdu, iname, ndilate=0):

    if "SEGMAP" in segmap_hdul:
        hdu_in = "SEGMAP"
    else:
        hdu_in = 0

    wcs, shape = get_wcs(iname)
    seg = reproject_interp(segmap_hdu, wcs, hdu_in=hdu_in,
                           shape_out=shape, return_footprint=False)
    if ndilate > 0:
        seg = binary_dilation(seg, iterations=ndilate)
    seg[seg > 0.0] = 1
    seg = seg.astype(int)
    hdu = fits.PrimaryHDU(seg)
    return hdu


# --------------
# parse input
# --------------
def get_parser():
    parser = argparse.ArgumentParser()
    # ims
    parser.add_argument("--exp_dir", type=str, default="./",
                        help="directory containing the input exposures")
    parser.add_argument("--lw_dir", type=str, default="./",
                        help="directory containing the LW exposures for making the segmap")
    parser.add_argument("--lw_pattern", type=str, default="*long*.fits",
                        help="pattern to match the LW exposures for making the segmap (e.g. *long*.fits)")
    parser.add_argument("--exp_pattern", type=str, default="*.fits",
                        help="pattern to match the exposures to mask (e.g. *cal.fits)")
    # seg
    parser.add_argument("--make_segmap", type=int, default=0,
                        help="if 1, make the segmap on the fly from lw_pattern; if 0, read from segname")
    parser.add_argument("--segname", type=str, default=None,
                        help="if make_segmap=0, read the segmap from this file.")
    # out
    parser.add_argument("--regenerate", type=int, default=1)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--mask_suffix", type=str, default="_smask.fits")
    return parser

if __name__ == "__main__":

    # MPI
    rank = 0
    n_child = 1
    size = 1
    HASMPI = False

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"masker-{rank}")
    logger.setLevel(logging.INFO)
    #logger.info(f"HASMPI={HASMPI}, {size} tasks")

    parser = get_parser()
    args = parser.parse_args()
    if args.mask_dir is None:
        args.mask_dir = args.exp_dir

    # --- make the segmap ---
    if args.make_segmap & (rank == 0):
        long_pattern = f"{args.lw_dir}/{args.lw_pattern}"
        det_images = glob.glob(long_pattern)
        assert len(det_images) > 0, long_pattern
        segmap_hdul, stack = images_to_seg(det_images)
        segmap_hdul.writeto(args.segname, overwrite=True)
        print(f"wrote {args.segname}")
        #segtype = "otf"

    if not bool(args.make_segmap):
        segmap_hdul = fits.open(args.segname)
        exp_pattern = f"{args.exp_dir}/{args.exp_pattern}"
        files = glob.glob(exp_pattern)
        if len(files) == 0:
            logger.warning(f"no files at {exp_pattern}")

        # loop over exposures for this worker
        wfiles = files[rank:None:n_child]
        for iname in wfiles:
            mname = os.path.basename(iname).replace(".fits", args.mask_suffix)
            mname = os.path.join(args.mask_dir, mname)
            if os.path.exists(mname) & (bool(args.regenerate) is False):
                continue
            seg = project_segmap(segmap_hdul, iname)
            seg.header["SEGN"] = args.segname
            seg.writeto(mname, overwrite=True)
            print(f"wrote {mname}")

