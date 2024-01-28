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

labeling_prompt = "Given a query and a web snippet, you must provide a score on an integer scale of 0 to 4 with the following meanings:\n\n" \
    "4 = navigational, this page represents a home page of an entity directly named by the query; the user may be searching for this specific page or site.\n" \
    "3 = top relevance, this page or site is dedicated to the topic; authoritative and comprehensive, it is worthy of being a top result in a web search engine.\n" \
    "2 = highly relevant, the content of this page provides substantial information on the topic.\n" \
    "1 = minimal relevance, the content of this page provides some information on the topic, which may be minimal; the relevant information must be on that page, not just promising-looking anchor text pointing to a possibly useful page.\n" \
    "0 = not relevant, the content of this page does not provide useful information on the topic, but may provide useful information on other topics, including other interpretations of the same query.\n" \
    'Assume that you are writing a report on the subject of the topic. mark the web snippet according to the previous scale description.\n' \
    'Query:\n' \
    'A person has typed {query} into a search engine.\n' \
    'Result:\n' \
    'Consider the following web snippet:\n' \
    '—BEGIN WEB Snippet CONTENT—\n' \
    '{snippet}\n' \
    '—END WEB Snippet CONTENT—\n' \
    'Instructions:\n' \
    'Split this problem into steps:\n' \
    'Consider the underlying intent of the search.\n' \
    'Measure how well the content matches a likely intent of the query (M)\n' \
    'Measure how trustworthy the web page is (T).\n' \
    'Consider the aspects above and the relative importance of each, and decide on a final score (O).\n' \
    'Produce a JSON of scores without providing any reasoning. Example:{"M": 2, "T": 1,"O": 1}\n' \
    'Results:\n'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"
assistant_prompt = "Given a query and a web snippet, you must provide a score on an integer scale of 0 to 4 with the following meanings:\n\n" \
    "4 = navigational, this page represents a home page of an entity directly named by the query; the user may be searching for this specific page or site.\n" \
    "3 = top relevance, this page or site is dedicated to the topic; authoritative and comprehensive, it is worthy of being a top result in a web search engine.\n" \
    "2 = highly relevant, the content of this page provides substantial information on the topic.\n" \
    "1 = minimal relevance, the content of this page provides some information on the topic, which may be minimal; the relevant information must be on that page, not just promising-looking anchor text pointing to a possibly useful page.\n" \
    "0 = not relevant, the content of this page does not provide useful information on the topic, but may provide useful information on other topics, including other interpretations of the same query.\n" \
    'Assume that you are writing a report on the subject of the topic. mark the web snippet according to the previous scale description.\n'

user_prompt = 'Query:\n' \
    'A person has typed {query} into a search engine.\n' \
    'Result:\n' \
    'Consider the following web snippet:\n' \
    '—BEGIN WEB Snippet CONTENT—\n' \
    '{snippet}\n' \
    '—END WEB Snippet CONTENT—\n' \
    'Instructions:\n' \
    'Split this problem into steps:\n' \
    'Consider the underlying intent of the search.\n' \
    'Measure how well the content matches a likely intent of the query (M)\n' \
    'Measure how trustworthy the web page is (T).\n' \
    'Consider the aspects above and the relative importance of each, and decide on a final score (O).\n' \
    'Produce a JSON of scores without providing any reasoning. Example:{"M": 2, "T": 1,"O": 1}' \

labeling_prompt_llama = f"{BOS}{B_INST} {B_SYS}\n" \
                f"{assistant_prompt}\n" \
                f"{E_SYS}\n\n" \
                 f"{user_prompt}\n" \
                 f"{E_INST}Results:\n"



def process_data(subset, out_file, qid_labeled, api_key, snippets, queries):
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, AIMessage
    model_kwargs = {'response_format': {"type": "json_object"}}

    chat_model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, max_tokens=1024, api_key=api_key,
                            model_kwargs=model_kwargs)

    output = open(out_file, 'a')
    for query_id in tqdm(subset):
        query = queries[query_id]
        for engine_id, search_content in tqdm(snippets.items()):
            current_representation = engine_id + "_" + query_id
            if current_representation in qid_labeled:
                continue
            if query_id not in search_content:
                continue
            snippet_list = search_content[query_id]
            if len(snippet_list) == 0:
                continue
            overall_score = 0
            for snippet_id, snippet in enumerate(snippet_list):
                prompt_input = HumanMessage(
                    content=labeling_prompt.replace("{query}", query).replace("{snippet}", snippet))
                output_text = chat_model.generate([[prompt_input]]).generations[0][0].text
                output_json = json.loads(output_text)
                converted_score = convert_score(output_json['O'])
                overall_score += converted_score

            overall_score = overall_score * 100
            output.write(json.dumps({"engine_id": engine_id, "query_id": query_id, "score": overall_score}) + "\n")


def main(input_folder, out_file, llm, model_path, query_id_file=None, num_processes=50):
    # Load API key and other initial setup
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()

    query_ids_set = set()
    if query_id_file is not None:
        with open(query_id_file, 'r') as file:
            for line in file:
                query_ids_set.add(line.strip())
    out_folder = os.path.dirname(out_file)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_files_now = glob.glob(os.path.join(out_folder, "*.jsonl"))
    qid_labeled = set()
    for out_file_now in out_files_now:
        qid_labeled.update(check_already_labeled(out_file_now))
    queries, snippets = load_search_content(input_folder)
    if llm == "gpt3.5":
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage, AIMessage
        subsets = numpy.array_split(list(query_ids_set), num_processes)

        # Create a pool of processes
        pool = multiprocessing.Pool(processes=num_processes)

        # Start the processes
        for i, subset in enumerate(subsets):
            process_out_file = f"{out_folder}/part_{i}.jsonl"
            pool.apply_async(process_data, args=(subset, process_out_file, qid_labeled, api_key, snippets, queries))

        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()
    else:
        output = open(out_file, 'a')
        if "flan" in llm:
            from models.flan_models import FLANModel
            import random
            random.seed(0)
            model = FLANModel()
            model.load_model(model_path=model_path)
            process_data_opensource(query_ids_set, queries, labeling_prompt, qid_labeled, snippets, model, output)
        elif "solar" in llm:
            from models.solar_models import SOLARModel
            import random
            random.seed(0)
            model = SOLARModel()
            model.load_model(model_path=model_path)
            print("Model loaded")
            process_data_opensource(query_ids_set, queries, labeling_prompt_llama, qid_labeled, snippets, model, output)




def process_data_opensource(qid_set, queries, prompt, qid_labeled, snippets, model, output):
    for query_id in tqdm(qid_set):
        query = queries[query_id]
        for engine_id, search_content in tqdm(snippets.items()):
            if query_id not in search_content:
                continue
            snippet_dict = search_content[query_id]
            if len(snippet_dict) == 0:
                continue
            for snippet_id in snippet_dict:
                current_representation = engine_id + "_" + query_id + "_" + snippet_id
                if current_representation in qid_labeled:
                    continue
                snippet = snippet_dict[snippet_id]
                text = model.generate(prompt, query, snippet)
                try:
                    current_json = json.loads(text)
                except:
                    current_json = extract_first_json(text)
                    if current_json is None:
                        print("Error occurred on the following text:")
                        print(f"{engine_id}, {query_id}, {snippet_id}")
                        print(text)
                        continue
                converted_score = convert_score_to_label(current_json['O'])
                output.write(json.dumps({"engine_id": engine_id, "query_id": query_id, "snippet_id": snippet_id, "label": converted_score}) + "\n")
    output.close()


import re
def extract_first_json(text):
    """
    Extract the first JSON object from a given string.

    :param text: The string containing the JSON object.
    :return: The first JSON object if found, otherwise None.
    """
    try:
        # Regex pattern to find the first JSON object
        pattern = r'\{.*?\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def convert_score_to_label(score):
    # The following weights are given to the relevance levels of documents: wNon = 0, wRel = 0.25, wHRel =0.5, wKey = rNav = 1
    if score == 0:
        return "nr"
    elif score == 1:
        return "r"
    elif score == 2:
        return "hr"
    elif score == 3:
        return "key"
    elif score == 4:
        return "nav"

def check_already_labeled(out_file):
    check_already_labeled = set()
    with open(out_file, 'r') as file:
        for line in file:
            current_json = json.loads(line)
            representation = current_json['engine_id'] + "_" +  current_json['query_id']+ "_" + current_json['snippet_id']
            check_already_labeled.add(representation)
    return check_already_labeled





def load_search_content(search_folder):
    """
    Load search content from subfolders within the specified search folder.
    Depending on the search_type, load different fields into a dictionary.
    """
    search_queries = {}
    search_snippets = {}
    for subfolder in tqdm(os.listdir(search_folder)):
        search_snippets[subfolder] = {}
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
                        search_snippets[subfolder][k_id] = {}
                        for snippet in item['snippets']:
                            for snippet_id, snippet_details in snippet.items():
                                tem_content = "Snippet Title: \n"
                                tem_content += snippet_details['title']  if snippet_details['title'] is not None else 'None'
                                tem_content += '\nSnippet Description: \n'
                                tem_content += snippet_details['description'] if snippet_details['description'] is not None else 'None'
                                search_snippets[subfolder][k_id][snippet_id] = tem_content


    return search_queries, search_snippets



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    parser.add_argument('--qid_file', type=str, default=None, help='qid_file')
    parser.add_argument('--llm', type=str, help='llm')
    parser.add_argument('--model_path', type=str, default='llama2-7b-chat', help='the path of the model')
    args, _ = parser.parse_known_args()
    main(args.input_folder, args.out_file, args.llm,args.model_path, args.qid_file)
