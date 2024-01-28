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


api_key_path = "api_key"
pseudo_output = '{"M": 2, "T": 1,"O": 1}'


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

example_query = "protege pizza tutorial"

snippet_examples="Snippet 1:\nSnippet Title: \nIn annual tradition, advertisers cowed by NFL trademark bullying.\n" \
        "Snippet Info:\n... to sell a variety of products\u2014televisions, pizzas, soda\u2014in conjunction with the game... \"Super Sale XLVI,\" using a football for a logo. Pizza Hut is offering a \"Big Deal…\n"

example_description = "You are looking for a tutorial or guide related to making or preparing pizza, possibly with a focus on specific techniques or styles. This could include step-by-step instructions, tips, or video demonstrations."

assistant_prompt = "Given a user query and relevant snippets about the query, a description of the query describes the user information need with respect to the relevant snippets.\n"

user_prompt = 'An example query is:\n' \
    f'{example_query}\n' \
    'then the relevant snippets are:\n' \
    f'{snippet_examples}' \
    f"The description of the user query is:\n" \
    f"{example_description}\n" \
    'Now, given the following user query:\n' \
    '{query}\n' \
    'the relevant snippets are:\n' \
    '{snippets}' \
    'Please write a description for the user query above.\n'

labeling_prompt_llama = f"{BOS}{B_INST} {B_SYS}\n" \
                f"{assistant_prompt}\n" \
                f"{E_SYS}\n\n" \
                 f"{user_prompt}\n" \
                 f"{E_INST}The description of the user query is:\n"



def main(input_folder, out_file, label_folder, model_path):
    # Load API key and other initial setup



    out_folder = os.path.dirname(out_file)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    combined_snippets = load_labeled_folder_snippets(label_folder)

    out_files_now = glob.glob(os.path.join(out_folder, "*.jsonl"))

    qid_labeled = set()
    for out_file_now in out_files_now:
        qid_labeled.update(check_already_labeled(out_file_now))
    queries, snippets = load_search_content(input_folder)

    output = open(out_file, 'a')
    from models.solar_models import SOLARModel
    import random
    random.seed(0)
    model = SOLARModel()
    model.load_model(model_path=model_path)
    print("Model loaded")
    process_data_opensource(queries, labeling_prompt_llama, qid_labeled, snippets, model, output, combined_snippets)



import random

def process_data_opensource(queries, prompt, qid_labeled, snippets, model, output, combined_snippets_dict):
    for query_id in tqdm(queries):
        if query_id in qid_labeled:
            continue
        query = queries[query_id]
        if query_id not in combined_snippets_dict:
            print("No snippets for query", query_id)
            continue
        if 'nav' not in combined_snippets_dict[query_id]:
            best_snippet_ids = []
        else:
            best_snippet_ids = combined_snippets_dict[query_id]['nav']

        if "key" in combined_snippets_dict[query_id]:
            if len(best_snippet_ids) <= 3:
                if len(combined_snippets_dict[query_id]['key']) >= 3-len(best_snippet_ids):
                    best_snippet_ids.extend(combined_snippets_dict[query_id]['key'][:3-len(best_snippet_ids)])
                else:
                    best_snippet_ids.extend(combined_snippets_dict[query_id]['key'])

        if len(best_snippet_ids) == 0:
            if "hr" in combined_snippets_dict[query_id]:
                if len(combined_snippets_dict[query_id]['hr']) >= 3:
                    best_snippet_ids.extend(combined_snippets_dict[query_id]['hr'][:3])
                else:
                    best_snippet_ids.extend(combined_snippets_dict[query_id]['hr'])


        snippets_combined = ""

        for snippet_rank, snippet_id in enumerate(best_snippet_ids):
            snippet = snippets[snippet_id]
            snippets_combined += "Snippet " + str(snippet_rank + 1) + ":\n" + snippet + "\n"
        description_generated = model.generate_description(prompt, query, snippets_combined)
        print(query, description_generated)
        output.write(json.dumps({"query_id": query_id, "query": query, "description": description_generated }) + "\n")
    output.close()



def check_already_labeled(out_file):
    check_already_labeled = set()
    with open(out_file, 'r') as file:
        for line in file:
            current_json = json.loads(line)
            representation = current_json['query_id']
            check_already_labeled.add(representation)
    return check_already_labeled

def load_labeled_folder_snippets(labeled_folder):
    labeled_data = {}
    for file in tqdm(os.listdir(labeled_folder)):
        if file.endswith(".jsonl"):
            with open(os.path.join(labeled_folder, file), 'r') as f:
                for line in f:

                    current_json = json.loads(line)
                    qid = current_json['query_id']
                    snippet_id = current_json['snippet_id']
                    label = current_json['label']
                    if label in ["key", "nav", "hr"]:
                        if qid not in labeled_data:
                            labeled_data[qid] = {}
                        if label not in labeled_data[qid]:
                            labeled_data[qid][label] = []
                        labeled_data[qid][label].append(snippet_id)
    return labeled_data



def load_search_content(search_folder):
    """
    Load search content from subfolders within the specified search folder.
    Depending on the search_type, load different fields into a dictionary.
    """
    search_queries = {}
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
                        k_id = key.split("_")[0]
                        search_queries[k_id] = item['query']
                        for snippet in item['snippets']:
                            for snippet_id, snippet_details in snippet.items():
                                tem_content = "Snippet Title: \n"
                                tem_content += snippet_details['title']  if snippet_details['title'] is not None else 'None'
                                tem_content += '\nSnippet Description: \n'
                                tem_content += snippet_details['description'] if snippet_details['description'] is not None else 'None'
                                search_snippets[snippet_id] = tem_content


    return search_queries, search_snippets



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    parser.add_argument('--snippet_labeled_folder', type=str, default='llama2', help='the name of the model')
    parser.add_argument('--model_path', type=str, default='llama2-7b-chat', help='the path of the model')
    args, _ = parser.parse_known_args()
    main(args.input_folder, args.out_file, args.snippet_labeled_folder, args.model_path)
