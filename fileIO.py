import csv
import numpy as np
from tqdm import tqdm

import csv
import numpy as np
from tqdm import tqdm
#import datatable as dt
import pandas as pd


def openMNIST(filename):
    labels = data.iloc[:,0].values
    images = data.iloc[:,1:].values
    return images, labels
            