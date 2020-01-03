from __future__ import absolute_import
from __future__ import print_function

from .dataset import Dataset, ImageDataset
from .kesicreid import KesciReID


__image_datasets = {
    'kescireid': KesciReID,
}



def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(name, avai_datasets))
    return __image_datasets[name](**kwargs)



def register_image_dataset(name, dataset):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        data = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        data = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError('The given name already exists, please choose '
                         'another name excluding {}'.format(curr_datasets))
    __image_datasets[name] = dataset
