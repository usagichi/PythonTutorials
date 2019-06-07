import pandas as pd
import os


def assemble(dataset_name="boston"):
    base_dataset_dir = os.path.abspath(os.path.join(os.curdir, '..', dataset_name))
    target_dir = os.path.join(base_dataset_dir, 'target')
    # target_string = "This is a string with a %s, %d and a %.2f parameter"
    target_files = [file for file in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, file))]
    target_files = []
    for filename in os.listdir(target_dir):
        file_url = os.path.join(target_dir, filename)
        if os.path.isfile(file_url):
            target_files.append(filename)
    n = len(target_files)
    target_values = [None] * n
    for observation_index in range(len(target_files)):
        with open(os.path.join(target_dir, '%d.txt' % observation_index), 'r') as file:
            observation_value = float(file.readline().strip())
            target_values[observation_index] = observation_value

    feature_names = []
    for dir in os.listdir(base_dataset_dir):
        dir_path = os.path.join(base_dataset_dir, dir)
        if dir != 'target' and os.path.isdir(dir_path):
            feature_names.append(dir)

    feature_dataset = {}

    for feature_name in feature_names:
        feature_dir = os.path.join(base_dataset_dir, feature_name)
        feature_values = [None] * n
        for observation_index in range(len(target_files)):
            with open(os.path.join(feature_dir, '%d.txt' % observation_index), 'r') as file:
                observation_value = float(file.readline().strip())
                feature_values[observation_index] = observation_value
        feature_dataset[feature_name] = feature_values
    feature_dataframe = pd.DataFrame(feature_dataset)
    return None, None

