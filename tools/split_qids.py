import argparse
import csv
import os
import json
from tqdm import tqdm
import math
import requests
import glob
import numpy
import multiprocessing



def load_search_content(search_folder):
    """
    Load search content from subfolders within the specified search folder.
    Depending on the search_type, load different fields into a dictionary.
    """
    search_queries = {}
    for subfolder in tqdm(os.listdir(search_folder)):

        subfolder_path = os.path.join(search_folder, subfolder)
        if os.path.isdir(subfolder_path):
            jsonl_files = glob.glob(os.path.join(subfolder_path + "/*.jsonl"))
            if len(jsonl_files)==0:
                print(f"JSONL file not found in {subfolder_path}")
                continue
            for json_file in jsonl_files:
                # Check if the JSON file exists in the subfolder
                if os.path.isfile(json_file):
                    with open(json_file, 'r') as file:
                        data = json.load(file)
                    for key, item in data.items():
                        k_id = key.split("_")[0]
                        search_queries[k_id] = item['query']
    return search_queries

def main(input_folder, input_folder_2, out_folder, n):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    search_queries = load_search_content(input_folder)
    search_queries_2 = load_search_content(input_folder_2)
    search_queries.update(search_queries_2)
    print(len(search_queries))

    #split into n parts
    num = math.ceil(len(search_queries) / n)
    search_query_id_list = list(search_queries.keys())

    for i in range(n):
        with open(os.path.join(out_folder, f"qids_{i}.txt"), 'w') as f:
            for j in range(num):
                if i * num + j >= len(search_query_id_list):
                    break
                f.write(search_query_id_list[i * num + j] + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--input_folder_2', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--out_folder', type=str, required=True, help='output file')
    #how many split
    parser.add_argument('--n', type=int, default=10, help='how many split')

    args, _ = parser.parse_known_args()
    main(args.input_folder, args.input_folder_2, args.out_folder, args.n)

