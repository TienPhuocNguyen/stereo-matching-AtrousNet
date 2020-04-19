import json

from dataset.CustomDataLoader import CustomDataLoader

def get_loader(name):

    return {
        'custom': CustomDataLoader
    }[name]


