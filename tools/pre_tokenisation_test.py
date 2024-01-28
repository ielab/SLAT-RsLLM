import argparse
import os
import json
from tqdm import tqdm
import glob
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
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
    'Produce a JSON of scores without providing any reasoning. Example:{"M": 2, "T": 1,"O": 1}]\n' \
    'Results:\n'

def plot_distribution(token_length_list, title):
    """
    Plot a distribution of token lengths.

    :param token_length_list: List of token lengths.
    :param title: Title for the plot.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(token_length_list, kde=True)
    plt.title(title)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.savefig(f"{title}.png")

def main(input_folder):
    tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0")
    prompt_len = len(tokenizer.tokenize(labeling_prompt))
    pseudo_output_len = len(tokenizer.tokenize(pseudo_output))
    print(pseudo_output_len)
    print(prompt_len)
    queries, snippets = load_search_content(input_folder)
    query_token_len_list = []
    for query in tqdm(queries):
        query_token_len = len(tokenizer.tokenize(queries[query]))
        query_token_len_list.append(query_token_len)
    print(f"Average query token length: {sum(query_token_len_list)/len(query_token_len_list)}")
    print(f"Max query token length: {max(query_token_len_list)}")

    plot_distribution(query_token_len_list, "query")
    snippet_token_len_list = []
    for snippet_id in tqdm(snippets):
        snippet_token_len = len(tokenizer.tokenize(snippets[snippet_id]))
        if snippet_token_len > 1000:
            continue
        snippet_token_len_list.append(snippet_token_len)
    print(f"Average snippet token length: {sum(snippet_token_len_list)/len(snippet_token_len_list)}")
    print(f"Max snippet token length: {max(snippet_token_len_list)}")

    plot_distribution(snippet_token_len_list, "snippet")



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
                                if snippet_id not in search_snippets:
                                    search_snippets[snippet_id] = tem_content
    return search_queries, search_snippets



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Search Source Selection')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the queries file')
    args, _ = parser.parse_known_args()
    main(args.input_folder)
