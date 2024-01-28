import argparse
import json
import csv
from tqdm import tqdm
import glob
import os


def load_engine(engines_file):
    engines = {}
    # Load engines
    with open(engines_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            engine_id = row['engineID'].replace("FW13-", "").replace("FW14-", "")

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

def load_example_result(example_result_file, n):
    # this is a dictionary
    # key is the query id
    print(example_result_file)
    initial_dict = json.load(open(example_result_file, "r"))
    result_dict = {}
    for qid in initial_dict:
        result_dict[qid] = {}
        for engine_id in initial_dict[qid]:
            result_dict[qid][engine_id] = [tem_id for tem_id, score in initial_dict[qid][engine_id][:n]]
    return result_dict

def load_search_content(search_folder):
    """
    Load search content from subfolders within the specified search folder.
    Depending on the search_type, load different fields into a dictionary.
    """
    search_snippets = {}

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
                        for snippet in item['snippets']:
                            for snippet_id, snippet_details in snippet.items():
                                tem_content = "Snippet Title: \n"
                                tem_content += snippet_details['title']  if snippet_details['title'] is not None else 'None'
                                tem_content += '\nSnippet Description: \n'
                                tem_content += snippet_details['description'] if snippet_details['description'] is not None else 'None'
                                search_snippets[snippet_id] = tem_content
    return search_snippets

def main(input_file, out_file, queries_folder, engine_file, query_representation, source_representation, description_file, snippet_dict=None):
    engine_dict = load_engine(engine_file)
    out_folder = os.path.dirname(out_file)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if query_representation == "description":
        description_dict = {}
        with open(description_file, 'r') as f:
            for line in f:
                current_json = json.loads(line)
                qid = current_json['query_id']
                description = current_json['description']
                description_dict[qid] = description


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
            engine_id = items[2].replace("FW13-", "").replace("FW14-", "")
            score = float(items[4])
            if qid not in input_dict:
                input_dict[qid] = []
            input_dict[qid].append((engine_id, score))

    all_engines = set(engine_dict.keys())
    train_lines = []

    for qid in input_dict:
        query = queries[qid]
        if query_representation == "description":
            if qid not in description_dict:
                print(f"Query {qid} not found in description_dict")
                continue
            query = description_dict[qid]

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
            if source_representation == "name_description":
                description = engine_dict[engine_id]["Description"]
            if (args.source_representation=="example_snippet") or (args.source_representation=="name_snippet"):
                description = engine_dict[engine_id]["Description"]
                snippets = snippet_dict[qid][engine_id]
                snippet_combined = ""
                for snippet_rank, snippet in enumerate(snippets):
                    snippet_splited = snippet.split(" ")
                    if len(snippet_splited) >= 20:
                        truncated_snippet = " ".join(snippet_splited[:20])
                    else:
                        truncated_snippet = snippet
                    snippet_combined += f"Snippet {snippet_rank + 1}:\n{truncated_snippet}\n"

            if (score >= 50):
                train_out = "Yes"
                if source_representation=="name_description":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description, "output": train_out}
                elif source_representation=="example_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description, "snippet": snippet_combined, "output": train_out}
                elif source_representation == "name_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "snippet": snippet_combined, "output": train_out}
                else:
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
                train_lines.append(train_dict)
            elif score >= 25 and score < 50:
                train_out = "Yes"
                if source_representation == "name_description":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description,
                                  "output": train_out}
                elif source_representation == "example_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description,
                                  "snippet": snippet_combined, "output": train_out}
                elif source_representation == "name_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "snippet": snippet_combined, "output": train_out}
                else:
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
                train_out = "No"
                if source_representation == "name_description":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description,
                                  "output": train_out}
                elif source_representation == "example_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description,
                                  "snippet": snippet_combined, "output": train_out}
                elif source_representation == "name_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "snippet": snippet_combined, "output": train_out}
                else:
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
            else:
                train_out = "No"
                if source_representation == "name_description":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description,
                                  "output": train_out}
                elif source_representation == "example_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description,
                                  "snippet": snippet_combined, "output": train_out}
                elif source_representation == "name_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "snippet": snippet_combined, "output": train_out}
                else:
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
                train_lines.append(train_dict)
            parsed_engine_ids.add(engine_id)
        #then check if there are engines not in the current rank, if so, we add them as No
        for engine_id in all_engines:
            if engine_id not in parsed_engine_ids:
                engine_url = engine_dict[engine_id]["URL"]
                engine_name = engine_dict[engine_id]["name"]
                if source_representation=="name_description":
                    description = engine_dict[engine_id]["Description"]
                if source_representation == "example_snippet":
                    description = engine_dict[engine_id]["Description"]
                    snippets = snippet_dict[qid][engine_id]
                    snippet_combined = ""
                    for snippet_rank, snippet in enumerate(snippets):
                        snippet_splited = snippet.split(" ")
                        if len(snippet_splited) >= 20:
                            truncated_snippet = " ".join(snippet_splited[:20])
                        else:
                            truncated_snippet = snippet
                        snippet_combined +=  f"Snippet {snippet_rank + 1}:\n{truncated_snippet}\n"

                train_out = "No"
                if source_representation == "name_description":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description,
                                  "output": train_out}
                elif source_representation == "example_snippet":
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "description": description,
                                  "snippet": snippet_combined, "output": train_out}
                else:
                    train_dict = {"query": query, "name": engine_name, "url": engine_url, "output": train_out}
                train_lines.append(train_dict)
                train_lines.append(train_dict)
    with open(out_file, 'w') as fw:
        json.dump(train_lines, fw, indent=4)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the ranking')
    parser.add_argument('--snippet_file', type=str, default="name", help="representation of the source, query or description")
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    parser.add_argument('--queries_folder', type=str, default="data/fedwebgh/search_data/fedweb14/FW14-realquery-snippet", help='folder to get all the queries')
    parser.add_argument('--snippet_folder', type=str,
                        default="data/fedwebgh/search_data/fedweb14/FW14-sample-snippet",
                        help='folder to get all the queries')
    parser.add_argument('--engine_file', type=str, default='data/fedwebgh/meta_data/engines/FW14-engines.csv')
    parser.add_argument("--query_representation", type=str, default="name",
                        help="representation of the source, query or description")
    parser.add_argument("--query_description_input", type=str, default="generated_descriptions/description.jsonl", help="the input file for the query description")
    parser.add_argument("--source_representation", type=str, default="name", help="representation of the source, name, or name_description")
    args, _ = parser.parse_known_args()
    final_dict = None
    if (args.source_representation=="example_snippet") or (args.source_representation=="name_snippet"):
        snippet_dict = load_search_content(args.snippet_folder)
        example_result_dict = load_example_result(args.snippet_file, 3)
        #create a final dict that is instead not only id, but find the id in the snippet dict
        final_dict = {}
        for qid in example_result_dict:
            final_dict[qid] = {}
            for engine_id in example_result_dict[qid]:
                engine_id_2 = engine_id.replace("FW13-", "").replace("FW14-", "")
                final_dict[qid][engine_id_2] = []
                for tem_id in example_result_dict[qid][engine_id]:
                    final_dict[qid][engine_id_2].append(snippet_dict[tem_id])


    main(args.input_file, args.out_file, args.queries_folder, args.engine_file, args.query_representation, args.source_representation, args.query_description_input, final_dict)