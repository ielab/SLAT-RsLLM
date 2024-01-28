#this python file reads a xml format file that contains query and generate a csv format file that is qid and query
#sample of xml file
#  <topic evaluation="TREC" id="7001" official="FedWeb13" origin="new">
#   <query>LHC collision publications</query>
#   <description>You want to find scientific publications on LHC (Large Hadron Collider (LHC)) collisions. Books or scientific video lectures are accepted, too.</description>
#   <narrative>You want to study the most recent knowledge on the topic in depth. You expect scientific publications to be most helpful, including books (you might consider buying one), but scientific video lectures or encyclopedia might be interesting as well.</narrative>
# </topic>

#sample of csv file
#qid,query,description,narrative
#7001,LHC collision publications,You want to find scientific publications on LHC (Large Hadron Collider (LHC)) collisions. Books or scientific video lectures are accepted, too.,You want to study the most recent knowledge on the topic in depth. You expect scientific publications to be most helpful, including books (you might consider buying one), but scientific video lectures or encyclopedia might be interesting as well.


import xml.etree.ElementTree as ET
import csv
import argparse
import os

def main(args):
    input_id_file = open(args.input_id_file, 'r')
    input_id_list = []
    for line in input_id_file:
        input_id_list.append(line.strip())
    input_id_file.close()
    tree = ET.parse(args.input_file)
    root = tree.getroot()
    csv_file = open(args.output_file, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['qid', 'query', 'description', 'narrative'])
    total_len = 0
    for topic in root.findall('topic'):
        qid = topic.get('id')
        total_len +=1
        if qid not in input_id_list:
            continue
        query = topic.find('query').text
        description = topic.find('description').text
        narrative = topic.find('narrative').text

        csv_writer.writerow([qid, query, description, narrative])
    csv_file.close()
    print(total_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument("--input_id_file", type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args)
