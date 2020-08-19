from __future__ import print_function, division, absolute_import

import numpy as np
from numba import jit


def make_bins(wavs):
    """ Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins. """
    edges = np.zeros(wavs.shape[0]+1)
    widths = np.zeros(wavs.shape[0])
    edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
    widths[-1] = (wavs[-1] - wavs[-2])
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
    edges[1:-1] = (wavs[1:] + wavs[:-1])/2
    widths[:-1] = edges[1:-1] - edges[:-2]

    return edges, widths


def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None,
             verbose=True):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.

    Parameters
    ----------

    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.

    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.

    Returns
    -------

    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins

    old_edges, old_widths = make_bins(old_wavs)
    new_edges, new_widths = make_bins(new_wavs)

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0
    warned = False

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill

            if (j == 0 or j == new_wavs.shape[0]-1) and verbose and not warned:
                warned = True
                print("\nSpectres: new_wavs contains values outside the range "
                      "in spec_wavs, new_fluxes and new_errs will be filled "
                      "with the value set in the 'fill' keyword argument. \n")
            continue

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

            if old_errs is not None:
                e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

                new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        return new_fluxes


@jit(nopython=True)
def _resample_single_with_err_jit(new_wavs, spec_wavs, spec_fluxes, spec_errs, fill, verbose):

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins

    # old_edges, old_widths = make_bins(old_wavs)
    # new_edges, new_widths = make_bins(new_wavs)
    old_edges = np.zeros(old_wavs.shape[0]+1)
    old_widths = np.zeros(old_wavs.shape[0])
    old_edges[0] = old_wavs[0] - (old_wavs[1] - old_wavs[0])/2
    old_widths[-1] = (old_wavs[-1] - old_wavs[-2])
    old_edges[-1] = old_wavs[-1] + (old_wavs[-1] - old_wavs[-2])/2
    old_edges[1:-1] = (old_wavs[1:] + old_wavs[:-1])/2
    old_widths[:-1] = old_edges[1:-1] - old_edges[:-2]

    new_edges = np.zeros(new_wavs.shape[0]+1)
    new_widths = np.zeros(new_wavs.shape[0])
    new_edges[0] = new_wavs[0] - (new_wavs[1] - new_wavs[0])/2
    new_widths[-1] = (new_wavs[-1] - new_wavs[-2])
    new_edges[-1] = new_wavs[-1] + (new_wavs[-1] - new_wavs[-2])/2
    new_edges[1:-1] = (new_wavs[1:] + new_wavs[:-1])/2
    new_widths[:-1] = new_edges[1:-1] - new_edges[:-2]

    # Generate output arrays to be populated
    # new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)
    new_fluxes = np.zeros(new_wavs.shape)
    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0
    warned = False

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[j] = fill

            new_errs[j] = fill

            if (j == 0 or j == new_wavs.shape[0]-1) and verbose and not warned:
                warned = True
                print("\nSpectres: new_wavs contains values outside the range "
                      "in spec_wavs, new_fluxes and new_errs will be filled "
                      "with the value set in the 'fill' keyword argument. \n")
            continue

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[j] = old_fluxes[start]
            new_errs[j] = old_errs[start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[start:stop+1]
            new_fluxes[j] = np.sum(f_widths, axis=-1)
            new_fluxes[j] /= np.sum(old_widths[start:stop+1])

            e_wid = old_widths[start:stop+1]*old_errs[start:stop+1]

            new_errs[j] = np.sqrt(np.sum(e_wid**2, axis=-1))
            new_errs[j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    return new_fluxes, new_errs


@jit(nopython=True)
def _resample_single_no_err_jit(new_wavs, spec_wavs, spec_fluxes, fill, verbose):

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes

    # Make arrays of edge positions and widths for the old and new bins

    # old_edges, old_widths = make_bins(old_wavs)
    # new_edges, new_widths = make_bins(new_wavs)
    old_edges = np.zeros(old_wavs.shape[0]+1)
    old_widths = np.zeros(old_wavs.shape[0])
    old_edges[0] = old_wavs[0] - (old_wavs[1] - old_wavs[0])/2
    old_widths[-1] = (old_wavs[-1] - old_wavs[-2])
    old_edges[-1] = old_wavs[-1] + (old_wavs[-1] - old_wavs[-2])/2
    old_edges[1:-1] = (old_wavs[1:] + old_wavs[:-1])/2
    old_widths[:-1] = old_edges[1:-1] - old_edges[:-2]

    new_edges = np.zeros(new_wavs.shape[0]+1)
    new_widths = np.zeros(new_wavs.shape[0])
    new_edges[0] = new_wavs[0] - (new_wavs[1] - new_wavs[0])/2
    new_widths[-1] = (new_wavs[-1] - new_wavs[-2])
    new_edges[-1] = new_wavs[-1] + (new_wavs[-1] - new_wavs[-2])/2
    new_edges[1:-1] = (new_wavs[1:] + new_wavs[:-1])/2
    new_widths[:-1] = new_edges[1:-1] - new_edges[:-2]

    # Generate output arrays to be populated
    # new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)
    new_fluxes = np.zeros(new_wavs.shape)

    start = 0
    stop = 0
    warned = False

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[j] = fill

            if (j == 0 or j == new_wavs.shape[0]-1) and verbose and not warned:
                warned = True
                print("\nSpectres: new_wavs contains values outside the range "
                      "in spec_wavs, new_fluxes and new_errs will be filled "
                      "with the value set in the 'fill' keyword argument. \n")
            continue

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[j] = old_fluxes[start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[start:stop+1]
            new_fluxes[j] = np.sum(f_widths, axis=-1)
            new_fluxes[j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # Otherwise just return the new_fluxes spectrum array
    return new_fluxes


def spectres_single_fast(new_wave, spec_wave, spec_flux, spec_err=None, fill=np.nan,
                         verbose=True):

    """
    Function for resampling a spectrum (and optionally associated
    uncertainties) onto a new wavelength basis with JIT acceleration. It is 
    faster than `spectres()` if you prefer to resample spectra one by one and 
    call the resampling function many times.

    Parameters
    ----------

    new_wave : numpy.ndarray
        1D array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wave : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_flux : numpy.ndarray
        1D array containing spectral flux at the wavelengths specified in
        spec_wave, the dimension must correspond to the shape of
        spec_wave. 

    spec_err : numpy.ndarray (optional)
        1D array of the same shape as spec_flux containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wave extends outside the wavelength range in spec_wave
        this value will be used as a filler in new_flux and new_err. By
        default, `fill = np.nan`.

    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wave extending outside spec_wave and "fill" being used.

    Returns
    -------

    new_flux : numpy.ndarray
        1D array of resampled flux values, the dimension is the same
        length as new_wave.

    new_err : numpy.ndarray
        1D array of uncertainties associated with flux in new_flux.
        Only returned if spec_err was specified.

    """

    if spec_err is not None:
        new_flux, new_err = _resample_single_with_err_jit(new_wave, spec_wave, spec_flux, spec_err, fill,
                                                          verbose)
        return new_flux, new_err

    # Otherwise just return the new_flux spectrum array
    else:
        new_flux = _resample_single_no_err_jit(
            new_wave, spec_wave, spec_flux, fill, verbose)
        return new_flux
