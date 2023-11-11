#
# Helper functions
#
import json
import os

import numpy as np

# When dumping tensors, take every nth channel to reduce output size
DATASET_TO_SUBSAMPLE = {
    'hieve' : 90,
    'sfu' : 9
}

def conv_to_numpy(element, max_channels):
    """
    Handle ragged Python list-of-lists (make into regular 2D numpy array)
    (for JSON deserialisation)
    """
    if not isinstance(element, list):
        return np.array(element)

    if not isinstance(element[0], list):
        return np.array(element)

    element1 = []
    for subelement in element:
        padding = (max_channels - len(subelement)) * [0.]
        element1.append(subelement + padding)

    return np.array(element1)

def conv_from_numpy(element, layer_cnt, channel_cnts):
    """
    Make uneven-length 2D numpy quant data into ragged Python list-of-lists
    (for JSON serialisation)
    """
    if element is None:
        return None
    if isinstance(element, float):
        return element
    if len(element.shape) < 2:
        return element.tolist()

    element1 = []
    for idx in range(layer_cnt):
        subelement = element[idx, :]
        element1.append(subelement[0:channel_cnts[idx]].tolist())
    return element1

def get_merged_dict(path, prefix):
    """
    Return a dictionary containing merged contents of parsed JSON files in 'path' beginning with
    'prefix'
    """
    fullpath = os.path.expandvars('${VCM_STAGE_DIR}') + '/' + path
    all_files = os.listdir(fullpath)
    json_files = [f for f in all_files if f.endswith('.json')]
    prefix_files = [f for f in json_files if f.startswith(prefix)]
    result = {}
    for prefix_file in prefix_files:
        with open(f'{fullpath}/{prefix_file}', 'r') as f:
            result.update(json.load(f))
    return result


class FeatureDump():
    """
    Dump tensor summary to disk for verifying data exists at the VCM feature decoder output
    (dumping actual tensors is prohibitive in terms of disk space).
    """
    def __init__(self, dataset=None, subsample=None):
        self._fptr = None
        self._offset = 0
        if subsample is not None:
            self._subsample = subsample
        elif dataset in DATASET_TO_SUBSAMPLE:
            self._subsample = DATASET_TO_SUBSAMPLE[dataset]
        else:
            self._subsample = 9
        print(f'Feature dump subsample ratio: {self._subsample}')

    def set_fptr(self, fptr):
        self._fptr = fptr

    def write_layers(self, layers):
        """
        Dump summary of input tensors (e.g. FPN layers) for one frame/image.
        Summary is feature map mean/variance subsampled on channelwise basis.
        Input is a list of numpy arrays.
        """
        if self._fptr is None or not self._fptr.writable():
            return

        summary = []
        for layer in layers:
            # dump output
            if layer is None:
                continue
            channel_cnt = layer.shape[1]
            means = np.mean(layer, axis=(2,3)).tolist()[0]
            variances = np.var(layer, axis=(2,3)).tolist()[0]

            means = means[self._offset::self._subsample]
            variances = variances[self._offset::self._subsample]
            self._offset = (self._offset + channel_cnt) % self._subsample

            summary.append({
                'means' : means,
                'variances' : variances
            })

        self._fptr.write(json.dumps(summary) + '\n')
