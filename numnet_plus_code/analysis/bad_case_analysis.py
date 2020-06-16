import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import os
import sys

sys.path.append(os.getcwd())
print(os.getcwd())
import random
from drop_eval import answer_json_to_strings, get_metrics


def load_json_file(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
        print('data has been load')
    f.close()
    return data


def dump_json_file(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
        print('data has been dumped')
    f.close()


def extract_bad_case(annotations, predicted_answers):
    bad_instance = []
    type_bad_instance: Dict[str, List[Any]] = defaultdict(list)

    for _, annotation in annotations.items():
        passage_info = annotation['passage']
        for qa_pair in annotation['qa_pairs']:
            query_id = qa_pair["query_id"]
            question = qa_pair["question"]
            max_em_score = 0.0
            max_f1_score = 0.0
            max_type = None
            if query_id in predicted_answers:
                predicted = predicted_answers[query_id]['predicted_answer']
                answer_type = predicted_answers[query_id]['answer_type']
                if 'arithmetic_expression' in predicted_answers[query_id].keys():
                    arithmetic_expression = predicted_answers[query_id]['arithmetic_expression']
                else:
                    arithmetic_expression = None
                candidate_answers = [qa_pair["answer"]]
                if "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                    candidate_answers += qa_pair["validated_answers"]
                for answer in candidate_answers:
                    gold_answer, gold_type = answer_json_to_strings(answer)
                    em_score, f1_score = get_metrics(predicted, gold_answer)
                    if gold_answer[0].strip() != "":
                        max_em_score = max(max_em_score, em_score)
                        max_f1_score = max(max_f1_score, f1_score)
                        if max_em_score == em_score or max_f1_score == f1_score:
                            max_type = gold_type
            else:
                print("Missing prediction for question: {}".format(query_id))
                if qa_pair and qa_pair["answer"]:
                    gold_answer, max_type = answer_json_to_strings(qa_pair["answer"])[1]
                else:
                    gold_answer = ""
                    max_type = "number"
                predicted = ""
                max_em_score = 0.0
                max_f1_score = 0.0

            if max_em_score == 0:
                instance = {}
                instance.update(
                    {'passage': passage_info, 'query_id': query_id, 'question': question, 'gold_type': max_type,
                     'answer_type': answer_type, 'arithmetic_expression': arithmetic_expression,
                     'gold_answer': gold_answer, 'predict_answer': predicted})
                bad_instance.append(instance)

                type_bad_instance[max_type].append(instance)

    return bad_instance, type_bad_instance


def sample(input_file, output_file, sample_num):
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    f.close()
    if sample_num <= len(input_data):
        sample_data = random.sample(input_data, sample_num)
    else:
        sample_data = input_data

    dump_json_file(sample_data, output_file)
    print('sample data has finished')


def statics(input_file):
    '''

    '''
    with open(input_file, 'r') as f:
        data = json.load(f)
    f.close()

    answer_type_num = defaultdict(list)
    for item in data:
        answer_type = item['answer_type']
        if answer_type in answer_type_num.keys():
            answer_type_num[answer_type] += 1
        else:
            answer_type_num[answer_type] = 1
    print(answer_type_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str, default='../numnet_plus_data', help=' root path for store data')
    parser.add_argument('--data_dir', type=str, default='drop_dataset', help='dataset used')
    parser.add_argument('--gold_file', type=str, default='drop_dataset_dev.json')
    parser.add_argument('--predict_file', type=str, default='prediction_drop_dataset_dev.json')
    parser.add_argument('--bad_case_file', type=str, default='bad_case_dev.json')

    parser.add_argument('--sample_num', type=int, default=50)
    parser.add_argument('--is_sample', default=True, action='store_true')
    parser.add_argument('--sample_file', type=str, default='sample_bad_case.json')

    args = parser.parse_args()

    gold_file = os.path.join(args.DATA_ROOT, args.data_dir, args.gold_file)
    predict_file = os.path.join(args.DATA_ROOT, args.data_dir, args.predict_file)
    bad_case_file = os.path.join(args.DATA_ROOT, args.data_dir, args.bad_case_file)

    print(bad_case_file)
    annotations = load_json_file(gold_file)
    prediction = load_json_file(predict_file)

    bad_instance, type_bad_instance = extract_bad_case(annotations=annotations, predicted_answers=prediction)

    print(len(bad_instance))
    dump_json_file(bad_instance, bad_case_file)

    for type, bad_case in type_bad_instance.items():
        type_bad_case_file = os.path.join(args.DATA_ROOT, args.data_dir, type + '_' + args.bad_case_file)
        dump_json_file(bad_instance, type_bad_case_file)
    print('finished bad case extract')

    if args.is_sample:
        sample_file = os.path.join(args.DATA_ROOT, args.data_dir, args.sample_file)
        print(sample_file)
        sample(bad_case_file, sample_file, args.sample_num)
        print('sample bad case finished')

    statics(bad_case_file)
    sample_bad_case_file = os.path.join(args.DATA_ROOT, args.data_dir, args.sample_file)
    statics(sample_bad_case_file)
