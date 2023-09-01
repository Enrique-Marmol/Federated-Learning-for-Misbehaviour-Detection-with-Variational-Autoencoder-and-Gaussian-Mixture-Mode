import pandas as pd
import numpy as np
import os
import json
import re


import json

def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        # Read each line as a separate JSON object
        json_data = [json.loads(line) for line in json_file]

    return json_data

def merge_json_files(file_path1, file_path2):
    # Read the JSON data from the first file
    json_data1 = read_json_file(file_path1)

    # Read the JSON data from the second file
    json_data2 = read_json_file(file_path2)

    # Merge the JSON data from both files into a single list
    merged_data = json_data1 + json_data2

    return merged_data

def flatten_list_attributes(data):
    # Function to flatten list attributes into separate dictionaries
    flattened_data = []
    for item in data:
        flattened_item = {}
        for key, value in item.items():
            if isinstance(value, list):
                for i, elem in enumerate(value):
                    flattened_item[f"{key}_{i}"] = elem
            else:
                flattened_item[key] = value
        flattened_data.append(flattened_item)
    return flattened_data

def pasar_csv(datos, url, zeros):
    flattened_data = flatten_list_attributes(datos)
    df = pd.DataFrame(flattened_data)
    if zeros:
        df['Label'] = np.zeros(len(df))
    else:
        df['Label'] = np.ones(len(df))
    df.to_csv(url, index=False)

if __name__ == '__main__':
    # Replace 'file_with_different_attributes.json' with the path to your JSON file
    #file_path = '/home/enrique/Datasets/Veremi extension/ConstPos_0709/VeReMi_25200_28800_2022-9-11_20.45.32/VeReMi_25200_28800_2022-9-11_20:45:32/traceJSON-9-7-A0-25202-7.json'

    # Read the JSON data from the file
    #json_data = read_json_file(file_path)
    """
    csv1b = read_json_file("/home/enrique/Datasets/Veremi extension/ConstPos_0709/VeReMi_25200_28800_2022-9-11_20.45.32/VeReMi_25200_28800_2022-9-11_20:45:32/traceJSON-21-19-A0-25208-7.json")
    csv2b = read_json_file("/home/enrique/Datasets/Veremi extension/ConstPos_0709/VeReMi_25200_28800_2022-9-11_20.45.32/VeReMi_25200_28800_2022-9-11_20:45:32/traceJSON-9-7-A0-25202-7.json")
    pasar_csv(csv1b+csv2b, "/home/enrique/Datasets/ve_jsons/benigno.csv", True)

    csv1m = read_json_file("/home/enrique/Datasets/Veremi extension/ConstPos_0709/VeReMi_25200_28800_2022-9-11_20.45.32/VeReMi_25200_28800_2022-9-11_20:45:32/traceJSON-33-31-A1-25210-7.json")
    csv2m = read_json_file("/home/enrique/Datasets/Veremi extension/ConstPos_0709/VeReMi_25200_28800_2022-9-11_20.45.32/VeReMi_25200_28800_2022-9-11_20:45:32/traceJSON-15-13-A1-25207-7.json")
    pasar_csv(csv1m+csv2m, "/home/enrique/Datasets/ve_jsons/maligno.csv", False)
    """


    directory = "/home/enrique/Datasets/Veremi extension/ConstPos_0709/menos"
    listado = os.listdir(directory)
    nombre = "-A0-"
    patron = re.compile(f"{nombre}.*\.json$")
    primero = True
    for archivo in listado:
        if patron.search(archivo):
            if primero:
                print(archivo)
                file_path = os.path.join(directory, archivo)
                total = read_json_file(file_path)
                primero = False
            else:
                print(archivo)
                file_path = os.path.join(directory, archivo)
                aux = read_json_file(file_path)
                total = total + aux
    #with open('/home/enrique/Datasets/ve_jsons/ConstPos_0709_benigno.json', 'w') as output_json_file:
    #    json.dump(total, output_json_file, indent=2)
    print("Pasando a csv...")
    #df = pd.json_normalize(total, sep='_')
    pasar_csv(total, "/home/enrique/Datasets/ve_jsons/benigno.csv", True)

    listado = os.listdir(directory)
    nombre = "-A1-"
    patron = re.compile(f"{nombre}.*\.json$")
    primero = True
    for archivo in listado:
        if patron.search(archivo):
            if primero:
                print(archivo)
                file_path = os.path.join(directory, archivo)
                total = read_json_file(file_path)
                primero = False
            else:
                print(archivo)
                file_path = os.path.join(directory, archivo)
                aux = read_json_file(file_path)
                total = total + aux
    # with open('/home/enrique/Datasets/ve_jsons/ConstPos_0709_benigno.json', 'w') as output_json_file:
    #    json.dump(total, output_json_file, indent=2)
    print("Pasando a csv...")
    # df = pd.json_normalize(total, sep='_')
    pasar_csv(total, "/home/enrique/Datasets/ve_jsons/maligno.csv", False)

    training_dataset = pd.read_csv("/home/enrique/Datasets/ve_jsons/benigno.csv")
    training_dataset = training_dataset[training_dataset['type'] > 2]

    training_dataset2 = pd.read_csv("/home/enrique/Datasets/ve_jsons/maligno.csv")
    training_dataset2 = training_dataset2[training_dataset2['type'] > 2]

    training_dataset = pd.concat([training_dataset, training_dataset2])
    training_dataset.to_csv("/home/enrique/Datasets/Veremi extension/csvs/prueba.csv")




"""
nombre = "-A1-"
patron = re.compile(f"{nombre}.*\.json$")
merged_data = []
for archivo in listado:
    if patron.search(archivo):
        print(archivo)
        file_path = os.path.join(directory, archivo)
        with open(file_path, 'r') as json_file:
            # Load the JSON data from the file
            data = json.load(json_file)
            merged_data.append(data)

with open('/home/enrique/Datasets/ve_jsons/ConstPos_0709_maligno.json', 'w') as output_json_file:
    json.dump(merged_data, output_json_file, indent=2)
    
"""
