import argparse
import json
import csv
from tqdm import tqdm
import glob
import os


def load_engine(engines_file, type):
    engines = {}
    # Load engines
    with open(engines_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            engine_id = row['engineID']
            if type not in engine_id:
                engine_id = type + "-" + row['engineID']

            engines[engine_id] = row

    return engines



def load_queries(search_folder):
    search_queries = {}
    for subfolder in tqdm(os.listdir(search_folder)):
        subfolder_path = os.path.join(search_folder, subfolder)
        if os.path.isdir(subfolder_path):
            jsonl_files = glob.glob(os.path.join(subfolder_path + "/*.jsonl"))
            if len(jsonl_files) == 0:
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


def main(input_file, out_file, type, queries_folder, engine_file):
    engine_dict = load_engine(engine_file, type)
    #we have three types, for scores > 50, or in top 10 (if the score > 0), we we store "Yes"
    #then we have ones with score > 10 and < 50, we have two training samples, one is Yes, another is No
    #then we have ones with score < 10, we have one training sample, which is No
    queries = load_queries(queries_folder)
    #input_file is a trec like file
    input_dict = {}
    #first read this in a dictionary with qid, then in it, it stores, engine_id with scores, sequentially
    with open(input_file) as f:
        for line in f:
            items = line.strip().split()
            qid = items[0]
            engine_id = items[2].replace("FW13", type)
            score = float(items[4])
            if qid not in input_dict:
                input_dict[qid] = []
            input_dict[qid].append((engine_id, score))

    all_engines = set(engine_dict.keys())
    train_lines = []

    for qid in input_dict:
        query = queries[qid]
        current_ranks = input_dict[qid]
        parsed_engine_ids = set()
        #first check if this qid is in the training set
        for engine_rank, engine_info in enumerate(current_ranks):
            engine_id = engine_info[0]
            score = engine_info[1]
            if engine_id not in engine_dict:
                print(f"Engine {engine_id} not found in engine_dict")
                continue

            engine_url = engine_dict[engine_id]["URL"]
            engine_name = engine_dict[engine_id]["name"]
            if (score >= 50) or (engine_rank < 10 and score > 0):
                train_out = "Yes"
                train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
                train_lines.append(train_dict)
            elif score >= 10 and score < 50:
                train_out = "Yes"
                train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
                train_out = "No"
                train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
            else:
                train_out = "No"
                train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
                train_lines.append(train_dict)
            parsed_engine_ids.add(engine_id)
        #then check if there are engines not in the current rank, if so, we add them as No
        for engine_id in all_engines:
            if engine_id not in parsed_engine_ids:


                engine_url = engine_dict[engine_id]["URL"]
                engine_name = engine_dict[engine_id]["name"]
                train_out = "No"
                train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
                train_lines.append(train_dict)
    with open(out_file, 'w') as fw:
        json.dump(train_lines, fw, indent=4)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the ranking')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    parser.add_argument('--type', type=str, required=True, help='FW13 or FW14')
    parser.add_argument('--queries_folder', type=str, default="/scratch/project/neural_ir/dylan/LLM_FS/data/fedwebgh/search_data/fedweb13/FW13-realquery-snippet", help='folder to get all the queries')
    parser.add_argument('--engine_file', type=str, required=True, help='engine_file')
    args, _ = parser.parse_known_args()
    main(args.input_file, args.out_file, args.type, args.queries_folder, args.engine_file)