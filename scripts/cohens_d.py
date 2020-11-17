import csv
import numpy as np
from ext.lab2im.utils import add_axis
from hypothalamus_seg.evaluate import cohens_d

path_subjects_info = '/home/benjamin/data/hypothalamus/test_data/ADNI_norm/subjects.npy'
path_subjects_volumes = '/home/benjamin/data/hypothalamus/evaluations/ADNI/ADNI_volumes_new.csv'

# ----------------------------------------------------------------------------------------------------------------------

# read subjects info
available_subjects_info = np.load(path_subjects_info)[1:, :]
diagnosis = np.int32(available_subjects_info[:, -1])
available_subjects_info = available_subjects_info[(diagnosis == 1) | (diagnosis == 6), :]
available_subjects = np.apply_along_axis(lambda d: d[0] + '_' + d[2], 1, available_subjects_info)

# read csv file
with open(path_subjects_volumes, 'r') as csvFile:
    data_iter = csv.reader(csvFile, delimiter=',')
    data = [data for data in data_iter]
data_array = np.asarray(data)

# extract data from csv file
label_names = data_array[0, 1:]
volumes = np.float32(data_array[1:, 1:])
processed_subjects = np.apply_along_axis(lambda d: d[0].replace('ADNI2_', ''), 1, data_array[1:, :])
processed_subjects = np.apply_along_axis(lambda d: d[0][:10] + '_' + d[0][15:], 0, add_axis(processed_subjects))

# find corresponding subjects
is_processed = np.in1d(available_subjects, processed_subjects)
available_subjects = available_subjects[is_processed]
available_subjects_sort_indices = np.argsort(available_subjects, kind='mergesort')
is_available = np.in1d(processed_subjects, available_subjects)
processed_subjects = processed_subjects[is_available]
processed_subjects_sort_indices = np.argsort(processed_subjects, kind='mergesort')

# find corresponding volumes and diagnosis
volumes = volumes[is_available, :][processed_subjects_sort_indices]
diagnosis = np.int32(available_subjects_info[is_processed, -1])[available_subjects_sort_indices]

# separate ad and control subjects
is_control = diagnosis == 1
is_ad = diagnosis == 6
volumes_ad = volumes[is_ad, :]
volumes_control = volumes[is_control, :]

# compute cohen's d
cohensd = np.around(cohens_d(volumes_control, volumes_ad), 3)

# print results
label_sort = [10, 1, 0, 4, 3, 2, 11, 6, 5, 9, 8, 7]
label_names = label_names[label_sort]
cohensd = cohensd[label_sort]
for label, d in zip(label_names, cohensd):
    print('\n' + label)
    print("cohen's d: %.3f" % d)
print('')
