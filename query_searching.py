import argparse
import csv
import os
import json
from FlagEmbedding import FlagModel
import torch
import numpy as np
from tqdm import tqdm
import glob
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
def load_model(search_type, model_name='BAAI/bge-large-en-v1.5'):
    """
    Load the embedding model.

    :param model_name: The name of the model to load.
    :return: The loaded model.
    """
    if search_type == 'query':
        instruction_retrieval = "Represent this query for searching other similar queries:"
    else:
        instruction_retrieval = "Represent this sentence for searching relevant passages::"
    model = FlagModel(model_name, use_fp16=True, query_instruction_for_retrieval=instruction_retrieval)
    print(model.device)
    return model


def predict_for_single_query(model, query_id, query, contents):
    """
    Process a single query and compute similarity scores with search contents.

    :param model: The embedding model.
    :param query_id: The ID of the query.
    :param query: The query text.
    :param search_contents: A dictionary of search content items.
    :return: A dictionary of similarity scores for the query against search contents.
    """
    # Embed the query
    q_embeddings = model.encode_queries([query])

    # Prepare to collect similarity scores
    similarity_scores = {}


    # Convert search contents to list and keep track of original keys
    content_list = list(contents.values())
    content_keys = list(contents.keys())

    # Embed the search contents
    p_embeddings = model.encode(content_list)

    # Compute dot product to get similarity scores
    scores = q_embeddings @ p_embeddings.T

    # Map back scores to original content keys
    content_scores = dict(zip(content_keys, scores[0]))
    similarity_scores = content_scores

    return similarity_scores

def predict_for_all_queries(q_embeddings, p_embeddings, query_keys , content_keys):
    """
    Process all queries and compute similarity scores with search contents.

    :param model: The embedding model.
    :param queries: A dictionary of all query texts.
    :param search_contents: A dictionary of search content items for a specific engine.
    :return: A dictionary of dictionaries containing similarity scores for each query against search contents.
    """
    scores = q_embeddings @ p_embeddings.T

    # Prepare to collect similarity scores for all queries
    all_similarity_scores = {}

    for query_id, query_embedding in zip(query_keys, scores):
        # Map back scores to original content keys for each query
        content_scores = dict(zip(content_keys, query_embedding))
        all_similarity_scores[query_id] = content_scores

    return all_similarity_scores

def load_search_content(search_folder, search_type, qid_set):
    """
    Load search content from subfolders within the specified search folder.
    Depending on the search_type, load different fields into a dictionary.
    """
    all_content = {}
    prefile_data = ""
    if "FW13" in search_folder:
        prefile_data = "FW13-"
    elif "FW14" in search_folder:
        prefile_data = "FW14-"
    # Iterate over subfolders in the search folder
    for subfolder in tqdm(os.listdir(search_folder)):
        search_content = {}
        subfolder_path = os.path.join(search_folder, subfolder)
        if os.path.isdir(subfolder_path):
            jsonl_files = glob.glob(os.path.join(subfolder_path + "/*.jsonl"))
            if len(jsonl_files)==0:
                continue
            for json_file in jsonl_files:
                # Check if the JSON file exists in the subfolder
                if os.path.isfile(json_file):
                    with open(json_file, 'r') as file:
                        data = json.load(file)
                    for key, item in data.items():
                        if (search_type == 'query') or (search_type=='realquery'):
                            # Extract the 'query' field
                            if key.split("_")[0] in qid_set:
                                continue
                            search_content[key] = item['query']
                        elif search_type == 'snippet':
                            # Extract the 'title' from each snippet
                            for snippet in item['snippets']:
                                for snippet_id, snippet_details in snippet.items():
                                    tem_content = snippet_details['title']  if snippet_details['title'] is not None else 'None'
                                    tem_content += ' '
                                    tem_content += snippet_details['description'] if snippet_details['description'] is not None else 'None'
                                    search_content[snippet_id] = tem_content
                        else:
                            raise ValueError("Invalid search type. Must be 'query' or 'snippet'.")
                else:
                    print(f"JSON file not found in {subfolder_path}")
        else:
            print(f"JSONL file not found in {subfolder_path}")
        print(len(search_content))
        all_content[prefile_data + subfolder.replace('/', "")] = search_content

    return all_content



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


def search(queries_file, query_representation, search_folder, search_type, write_file, qids=None):
    """
    Process queries from the queries file based on the query representation and search type.
    """
    queries = {}
    if queries_file.endswith(".csv"):
        with open(queries_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                query_id = row['qid']
                queries[query_id] = row[query_representation]
    else:
        queries = load_queries(queries_file)
        with open(qids, 'r') as f:
            qids = f.read().splitlines()
        queries = {k: v for k, v in queries.items() if k not in qids}
    qid_set = set(queries.keys())
    search_contents = load_search_content(search_folder, search_type, qid_set)

    model = load_model(search_type=search_type)
    query_keys = list(queries.keys())
    query_values = list(queries.values())
    queries_embeddings = model.encode_queries(query_values, max_length=32)

    engine_embedding_dict = {}


    search_content_keys_dict = {}
    search_contents_values_dict = {}

    for engine_id, search_content in tqdm(search_contents.items()):
        search_content_keys_dict[engine_id] = list(search_content.keys())
        search_contents_values_dict[engine_id] = list(search_content.values())
        engine_embedding_dict[engine_id] = model.encode(search_contents_values_dict[engine_id], max_length=256)

    write_dict={}
    for engine_id, search_content in tqdm(search_contents.items()):
        similarity_scores = predict_for_all_queries(queries_embeddings, engine_embedding_dict[engine_id], query_keys, search_content_keys_dict[engine_id])
        for query_id, scores in similarity_scores.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
            converted_list = []
            #convert all scores to float instead of float16
            for key, value in sorted_scores:
                converted_list.append((key, float(value)))


            if query_id not in write_dict:
                write_dict[query_id] = {}
            write_dict[query_id][engine_id] = converted_list

    # Write the converted dictionary to a JSON file
    with open(write_file, 'w') as file:
        json.dump(write_dict, file, indent=4)
        #for engine_id, search_content in search_contents.items():






# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process query arguments')
    parser.add_argument('--queries', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--qids', type=str, help='Path to the queries file')
    parser.add_argument('--search_folder', type=str, required=True, help='Path to the search_folder')

    parser.add_argument('--query_representation', type=str, choices=['query', 'description'], required=True,
                        help='Type of query representation')
    parser.add_argument('--search_type', type=str, choices=['query', 'snippet', 'realquery'], required=True,
                        help='Type of query representation')
    parser.add_argument('--write_file', type=str, required=True, help='Path to the write_file')
    #parser.add_argument('--embedding_model', type=str, required=True,
                        #help='which embedding model to use')
    args = parser.parse_args()
    search(args.queries, args.query_representation, args.search_folder, args.search_type, args.write_file, args.qids)




