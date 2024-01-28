from models.openai_models import GPT3_5Model, GPT4Model
from models.llama_models import LLAMA2_CHATMODEL
from models.flan_models import FLANModel
from models.mixtral_models import MIXTRALModel
from models.falcon_models import FALCONModel
from models.solar_models import SOLARModel
from source.base_source import BaseSource
import argparse
import csv
import os
import json
from tqdm import tqdm
api_key_path = "api_key"


def main(queries_file, engines_file, query_representation, llm_type, source_representation, batch, model_path, out_file, search_query_result=None, snippet_folder=None, search_top_n=None):
    queries, sources = load_dataset(queries_file, engines_file)
    # Initialize LLM based on the type
    api_key = open(api_key_path, "r").read()
    if llm_type == 'gpt-4':
        model = GPT4Model(api_key=api_key)
        model.load_model()
    # Add other models here based on llm_type
    elif llm_type == 'gpt-3.5':
        model = GPT3_5Model(api_key=api_key)
        model.load_model()
    elif "llama2" in llm_type:
        import random
        random.seed(0)
        model = LLAMA2_CHATMODEL()
        model.load_model(model_path=model_path)
    elif "flan" in llm_type:
        import random
        random.seed(0)
        model = FLANModel()
        model.load_model(model_path=model_path)
    elif "mixtral" in llm_type:
        import random
        random.seed(0)
        model = MIXTRALModel()
        model.load_model(model_path=model_path)
    elif "falcon" in llm_type:
        import random
        random.seed(0)
        model = FALCONModel()
        model.load_model(model_path=model_path)
    elif "solar" in llm_type:
        import random
        random.seed(0)
        model = SOLARModel()
        model.load_model(model_path=model_path)
    else:
        raise ValueError("Unsupported LLM type")

    representation_dict = {}

    for source_id in sources:
        # Initialize source based on the representation type
        source_info_dict = sources[source_id]
        source_info = BaseSource(source_info_dict)
        representation = source_info.get_representation(source_representation, llm_type)
        print(representation)
        representation_dict[source_id] = representation


    if source_representation == "example_snippet" or source_representation == "name_snippet":
        real_docs_dict = {}
        search_query_dict = load_example_result(search_query_result, search_top_n)
        search_content_dict = load_search_content(snippet_folder, "example_doc")
        for qid in search_query_dict:
            real_docs_dict[qid] = {}
            for engine_id in search_query_dict[qid]:
                real_docs_dict[qid][engine_id] = []
                for tem_id in search_query_dict[qid][engine_id]:
                    real_docs_dict[qid][engine_id].append(search_content_dict[engine_id][tem_id])
    elif source_representation == "example_querydoc" or source_representation == "name_querydoc" or source_representation=="realquery_querydoc":
        real_docs_dict = {}
        search_query_dict = load_example_result(search_query_result, 1)
        search_content_dict = load_search_content(snippet_folder, "example_querydoc")
        for qid in search_query_dict:
            real_docs_dict[qid] = {}
            for engine_id in search_query_dict[qid]:
                real_docs_dict[qid][engine_id] = {}
                tem_id = search_query_dict[qid][engine_id][0]
                real_docs_dict[qid][engine_id]["query"] = search_content_dict[engine_id][tem_id]["query"]
                if len(search_content_dict[engine_id][tem_id]["snippets"])>search_top_n:
                    real_docs_dict[qid][engine_id]["snippets"] = search_content_dict[engine_id][tem_id]["snippets"][0:search_top_n]
                else:
                    real_docs_dict[qid][engine_id]["snippets"] = search_content_dict[engine_id][tem_id]["snippets"]
    already_finished_qids = check_already_finished_qids(out_file)
    chunked_representation_dict = chunk_dict(representation_dict, int(batch))

    for qid in tqdm(queries):
        # Decide source for each query versus source representation
        if qid in already_finished_qids:
            continue

        query = queries[qid][query_representation]
        score_dict = {}

        for chunk in chunked_representation_dict:
            if (llm_type == 'gpt-4') or (llm_type == 'gpt-3.5'):
                for source_id in chunk:

                    decision_score = model.predict(query, representation_dict[source_id])
                    #print(f"Query: {query}\nSource: {representation_dict[source_id]}\nDecision Score: {decision_score}\n")
                    score_dict[source_id] = decision_score
                # Sort sources based on the decision score
            else:
                if source_representation == "example_snippet" or source_representation == "name_snippet":
                    result_dict = model.batch_predict(query, chunk, real_docs_dict[qid])
                elif source_representation == "example_querydoc" or source_representation == "name_querydoc" or source_representation == "realquery_querydoc":
                    result_dict = model.batch_predict(query, chunk, real_docs_dict[qid], True)
                else:
                    result_dict = model.batch_predict(query, chunk)
                for source_id in result_dict:
                    score_dict[source_id] = result_dict[source_id]

        sorted_sources = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            # Write the results to a file
        write_trec_results(qid, sorted_sources, llm_type, out_file)

def chunk_dict(input_dict, chunk_size):
    # Sort the dictionary based on the length of the value
    sorted_items = sorted(input_dict.items(), key=lambda x: len(x[1].split()))
    # Create chunks
    chunks = []
    current_chunk = {}
    current_size = 0

    for key, value in sorted_items:
        if current_size + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = {}
            current_size = 0
        current_chunk[key] = value
        current_size += 1

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def check_already_finished_qids(result_file):
    # This function get all qids from the result filem in trec format
    # You can use this function to check if the query has been processed
    qids = []
    if not os.path.exists(result_file):
        return set(qids)
    with open(result_file, "r") as f:
        for line in f:
            qid = line.split()[0]
            qids.append(qid)
    return set(qids)


def write_trec_results(qid, sorted_sources, label, out_file):
    # This function writes the results to a file
    # The results should be in TREC format
    # The file name should be {qid}.trec
    # The format of each line should be {qid} Q0 {source_id} {rank} {decision_score} {run_name}

    if not os.path.exists(out_file):
        #check if the folder exist first
        out_folder = os.path.dirname(out_file)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        #create the file
        fw = open(out_file, "w")
    else:
        fw = open(out_file, "a")
    rank = 1
    for source_id, decision_score in sorted_sources:
        # if decision_score <= 0:
        #     continue
        fw.write(f"{qid} Q0 {source_id} {rank} {decision_score} {label}\n")
        rank += 1

    fw.close()

def load_dataset(queries_file, engines_file):
    queries = {}
    engines = {}

    # Load queries
    with open(queries_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            query_id = row['qid']
            queries[query_id] = row

    # Load engines
    with open(engines_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            engine_id = row['engineID']
            engines[engine_id] = row

    return queries, engines


def load_example_result(example_result_file, n):
    # this is a dictionary
    # key is the query id
    initial_dict = json.load(open(example_result_file, "r"))
    result_dict = {}
    for qid in initial_dict:
        result_dict[qid] = {}
        for engine_id in initial_dict[qid]:
            result_dict[qid][engine_id] = [tem_id for tem_id, score in initial_dict[qid][engine_id][:n]]
    return result_dict

import glob
def load_search_content(search_folder, search_type):
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
    for subfolder in os.listdir(search_folder):
        search_content = {}
        subfolder_path = os.path.join(search_folder, subfolder)
        if os.path.isdir(subfolder_path):
            jsonl_files = glob.glob(os.path.join(subfolder_path + "/*.jsonl"))
            if len(jsonl_files) == 0:
                continue
            for json_file in jsonl_files:

                if os.path.isfile(json_file):
                    with open(json_file, 'r') as file:
                        data = json.load(file)

                    for key, item in data.items():
                        if (search_type == 'example_querydoc'):
                            # Extract the 'query' field
                            search_content[key] = {}
                            search_content[key]['query'] = item['query']
                            search_content[key]['snippets'] = []
                            for snippet in item['snippets']:
                                for snippet_id, snippet_details in snippet.items():
                                    tem_content = snippet_details['title']  if snippet_details['title'] is not None else 'None'
                                    tem_content += ' '
                                    tem_content += snippet_details['description'] if snippet_details['description'] is not None else 'None'
                                    search_content[key]['snippets'].append(tem_content)


                        elif search_type == 'example_doc':
                            # Extract the 'title' from each snippet
                            for snippet in item['snippets']:
                                for snippet_id, snippet_details in snippet.items():
                                    tem_content = "Title: "
                                    tem_content += snippet_details['title'].strip() if snippet_details[
                                                                                           'title'] is not None else 'None'
                                    tem_content += '\n'
                                    tem_content = "Description: "
                                    tem_content += snippet_details['description'] if snippet_details[
                                                                                         'description'] is not None else 'None'
                                    search_content[snippet_id] = tem_content
                        else:
                            raise ValueError("Invalid search type. Must be 'query' or 'snippet'.")

        all_content[prefile_data + subfolder.replace('/', "")] = search_content

    return all_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--queries', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--engines', type=str, required=True, help='Path to the engines file')
    parser.add_argument('--llm', type=str, choices=['gpt-4', 'gpt-3.5', 'llama2-7b-chat', 'llama2-13b-chat', "flan-xxl", "flan-large", "flan-xl", "mixtral-13b", "falcon-7b", "solar-11b", "flan-large-tuned","flan-xl-tuned", "llama2-7b-chat-tuned"], required=True, help='Type of Large Language Model to use')
    parser.add_argument('--query-representation', type=str, choices=['query', 'description','narrative'], required=True,
                        help='Type of query representation')
    parser.add_argument('--source-representation', type=str, choices=['name', 'name_description', "example_querydoc", "example_snippet", "name_querydoc", "name_snippet", "realquery_querydoc", "querylog"], required=True, help='Type of source representation')
    parser.add_argument('--batch', type=int, default=1, help='the batch size')
    parser.add_argument('--model_path', type=str, default='llama2-7b-chat', help='the path of the model')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    args, _ = parser.parse_known_args()

    # Conditional arguments based on the value of source-representation
    if (args.source_representation == "example_snippet") or (args.source_representation == "example_querydoc") or (args.source_representation == "name_snippet") or (args.source_representation == "name_querydoc") or (args.source_representation == "realquery_querydoc"):
        parser.add_argument('--search_query_result', type=str, required=True,
                            help='Path to the search query result file')
        parser.add_argument('--snippet_folder', type=str, required=True, help='Path to the snippet folder')
        parser.add_argument('--search_top_n', type=int, required=True, help='top n example in the search query result file')
    # Parse again including the new arguments
    args = parser.parse_args()
    main(args.queries, args.engines, args.query_representation, args.llm, args.source_representation, args.batch,
         args.model_path, args.out_file, getattr(args, 'search_query_result', None),
         getattr(args, 'snippet_folder', None), getattr(args, 'search_top_n', None))

