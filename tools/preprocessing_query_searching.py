import os
import tarfile
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
def extract_tgz_files(folder_path):
    for item in os.listdir(folder_path):
        if item.endswith(".tgz"):
            file_path = os.path.join(folder_path, item)
            tar = tarfile.open(file_path, "r:gz")
            tar.extractall(path=folder_path)
            tar.close()

def parse_xml_and_create_json(folder_path):
    for current_dir, dirs, files in tqdm(os.walk(folder_path)):
        for file in files:
            file_dict = {}
            if file.endswith(".xml"):
                print(file)
                file_path = os.path.join(current_dir, file)
                tree = ET.parse(file_path)
                xml_root = tree.getroot()

                for search_results in xml_root.findall('search_results'):
                    query_id = search_results.find('query').get('id')
                    query_text = search_results.find('query').text
                    engine_id = search_results.find('engine').get('id')[5:]
                    snippets = []

                    for snippet in search_results.findall('.//snippet'):
                        snippet_title = snippet.find('title').text if snippet.find('title') is not None else 'None'
                        snippet_description = snippet.find('description').text if snippet.find('description') is not None else 'None'
                        snippet_data = {
                            'title': snippet_title,
                            'description': snippet_description
                        }
                        snippets.append({snippet.get('id'): snippet_data})

                    data = {
                        'query': query_text,
                        'engine_id': engine_id,
                        'snippets': snippets
                    }
                    new_query_id = f"{query_id}_{engine_id}"
                    file_dict[new_query_id] = data

                # Write to a file in the current directory
                with open(os.path.join(current_dir, f"{file.split('.')[0]}.jsonl"), 'w') as json_file:
                    json.dump(file_dict, json_file, indent=4)


def main(folder_path):
    #extract_tgz_files(folder_path)
    parse_xml_and_create_json(folder_path)

if __name__ == "__main__":
    folder_path = 'data/fedwebgh/search_data/fedweb14/FW14-sample-search'
    main(folder_path)
