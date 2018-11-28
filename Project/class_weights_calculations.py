import pandas as pd
import numpy as np
import math


def get_class_weights():
    label_names = {
        0:  "Nucleoplasm",
        1:  "Nuclear membrane",
        2:  "Nucleoli",
        3:  "Nucleoli fibrillar center",
        4:  "Nuclear speckles",
        5:  "Nuclear bodies",
        6:  "Endoplasmic reticulum",
        7:  "Golgi apparatus",
        8:  "Peroxisomes",
        9:  "Endosomes",
        10:  "Lysosomes",
        11:  "Intermediate filaments",
        12:  "Actin filaments",
        13:  "Focal adhesion sites",
        14:  "Microtubules",
        15:  "Microtubule ends",
        16:  "Cytokinetic bridge",
        17:  "Mitotic spindle",
        18:  "Microtubule organizing center",
        19:  "Centrosome",
        20:  "Lipid droplets",
        21:  "Plasma membrane",
        22:  "Cell junctions",
        23:  "Mitochondria",
        24:  "Aggresome",
        25:  "Cytosol",
        26:  "Cytoplasmic bodies",
        27:  "Rods & rings"
    }

    train_labels = pd.read_csv("train.csv")
    reverse_train_labels = dict((v,k) for k,v in label_names.items())

    def fill_targets(row):
        row.Target = np.array(row.Target.split(" ")).astype(np.int)
        for num in row.Target:
            name = label_names[int(num)]
            row.loc[name] = 1
        return row

    for key in label_names.keys():
        train_labels[label_names[key]] = 0
    train_labels = train_labels.apply(fill_targets, axis=1)

    # Compute count for each class present in the dictionary.
    count_dictionary = {}
    for k,v in label_names.items():
        count_dictionary[k] = train_labels[label_names[k]].sum()

    d2 = dict((k, 1/math.log(v)) for k, v in count_dictionary.items())

    return d2
