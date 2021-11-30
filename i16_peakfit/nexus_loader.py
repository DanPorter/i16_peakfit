"""
i16_peakfit.nexus_loader
Functions copied from BabelScan
"""

import numpy as np
import h5py


"----------------------------LOAD FUNCTIONS---------------------------------"


def load(filename):
    """Load a hdf5 or nexus file"""
    try:
        return h5py.File(filename, 'r')
    except OSError:
        raise Exception('File does not exist: %s' % filename)


def hdf_addresses(filename, addresses='/', recursion_limit=100, get_size=None, get_ndim=None):
    """Load hdf5 file and return list of addresses"""
    with load(filename) as hdf:
        addresses = dataset_addresses(hdf, addresses, recursion_limit, get_size, get_ndim)
    return addresses


def array_addresses(filename):
    """Load hdf5 file and return addresses of arrays"""
    return hdf_addresses(filename, get_ndim=1)


def load_hdf_array(files, address, default=None):
    """
    Load single dataset array (scandata) from hdf files
      Will return array as per dataset.
    :param files: str or list of str file names
    :param address: str hdf dataset address
    :param default: value to return if dataset not in file
    :return: list of arrays (or Nones if address is not array or doesn't exit)
    """
    if default is None:
        default = np.array([0], dtype=float)
    files = np.asarray(files, dtype=str).reshape(-1)
    output = []
    for n, file in enumerate(files):
        with load(file) as hdf:
            if address in hdf:
                dataset = hdf.get(address)
                if dataset.ndim > 0:
                    output += [hdf.get(address)[()]]
                else:
                    output += [default]
            else:
                output += [default]
    return output


def load_hdf_values(files, address, default=None):
    """
    Load single dataset value (metadata) from hdf files
      Will return str or float value as per dataset. Array datsets will be averaged to return a single float.
    :param files: str or list of str file names
    :param address: str hdf dataset address
    :param default: value to return if dataset not in file
    :return: array of floats or strings
    """
    files = np.asarray(files, dtype=str).reshape(-1)
    values = np.empty(len(files), dtype=object)
    for n, file in enumerate(files):
        with load(file) as hdf:
            if address in hdf:
                dataset = hdf.get(address)
                if dataset.ndim > 0:
                    values[n] = np.mean(dataset)
                else:
                    values[n] = hdf.get(address)[()]
            else:
                values[n] = default
    return values


"-------------------------HDF ADDRESS FUNCTIONS-------------------------------"


def dataset_addresses(hdf_group, addresses='/', recursion_limit=100, get_size=None, get_ndim=None):
    """
    Return list of addresses of datasets, starting at each address
    :param hdf_group: hdf5 File or Group object
    :param addresses: list of str or str : time_start in this / these addresses
    :param recursion_limit: Limit on recursivley checking lower groups
    :param get_size: None or int, if int, return only datasets with matching size
    :param get_ndim: None or int, if int, return only datasets with matching ndim
    :return: list of str
    """
    addresses = np.asarray(addresses, dtype=str).reshape(-1)
    out = []
    for address in addresses:
        data = hdf_group.get(address)
        if data and hasattr(data, 'size'):
            # address is dataset
            if (get_size is None and get_ndim is None) or (get_size is not None and data.size == get_size) or (
                    get_ndim is not None and data.ndim == get_ndim):
                out += [address]
        elif data and recursion_limit > 0:
            # address is Group
            new_addresses = ['/'.join([address, d]).replace('//', '/') for d in data.keys()]
            out += dataset_addresses(hdf_group, new_addresses, recursion_limit - 1, get_size, get_ndim)
        #elif recursion_limit > 0:
        #    # address is None, search for group address and iterate
        #    new_address = get_address(hdf_group, address, return_group=True)  # this goes forever if a group fails to load
        #    if new_address:
        #        out += dataset_addresses(hdf_group, new_address, recursion_limit - 1, get_size, get_ndim)
    return out

