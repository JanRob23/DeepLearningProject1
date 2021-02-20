import csv
import numpy as np
from tqdm import tqdm

def openMNIST(filename):
    labels = np.array(list())
    images = np.array(list())
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in tqdm(csv_reader):
            if line_count == 0:
                line_count += 1
            else:
                labels = np.append(labels, row[0])
                images = np.append(images, row[1:])
                line_count += 1
    return images, labels
            