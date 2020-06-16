import os
import spacy
# from mspan_roberta_gcn.drop_roberta_dataset import name_entity_recog
from datetime import datetime
import json
import re
def load_json_file(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def test_name_entities_speed(sentences: [list, str]):
    start = datetime.now()
    nlp = spacy.load('en_core_web_sm')
    entities = []
    for sent in sentences:
        doc = nlp(sent)
        entities.append(doc.ents)

    end  = datetime.now()

    if isinstance(sentences, list) and len(sentences)>0:
        per_time = (end - start).seconds/len(sentences)
    else:
        per_time = (end-start).seconds
    print('NER speed', per_time)


if __name__ == '__main__':
    # input_file = "../../numnet_plus_data/drop_dataset/sample_bad_case.json"
    # data = load_json_file(input_file)
    # print(data)
    # passages = []
    # for ins in data:
    #     passage = ins['passage']
    #     passages.append(passage)
    #
    # print(len(passages))
    # test_name_entities_speed(passages)
    words = '我，来。上海？吃？上海菜'
    wordlist = re.split('，|。|？', words)
    print(wordlist)