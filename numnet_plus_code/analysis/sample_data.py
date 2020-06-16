import json
import os
import random
import argparse


def load_json_file(input_file):
    assert os.path.exists(input_file)
    with open(input_file, 'r') as f:
        data = json.load(f)
    f.close()
    print('finish load json file {}'.format(input_file))
    return data


def dumps_json_file(data, output_file):
    assert os.path.exists(output_file)
    with open(output_file, 'w') as f:
        json.dump(data, f)
    f.close()
    print('finished dumps json file {}'.format(output_file))


def sample(input_data, sample_num):
    data_size = len(input_data)
    assert sample_num <= data_size
    output_data = random.sample(list(input_data), sample_num)
    print('sample finished')
    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data dir', default='../numnet_plus_data')
    parser.add_argument('--input_file', type=str, help='input file path',
                        default='drop_dataset/drop_dataset_train.json')
    parser.add_argument('--sample_file', type=str, help='sample file path',
                        default='drop_dataset_sample/drop_dataset_train.json')
    parser.add_argument('--sample_num', type=int, default=50, help='sample num of passage')

    args = parser.parse_args()

    input_file = os.path.join(args.data_path, args.input_file)
    sample_file = os.path.join(args.data_path, args.sample_file)
    sample_num = args.sample_num

    input_data = load_json_file(input_file)
    sample_data = sample(input_data, sample_num)
    dumps_json_file(sample_data, sample_file)
