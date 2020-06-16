import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--data_root_dir', type=str, default='numnet_plus_data')
    parser.add_argument('--data_dir', type=str, default='drop_dataset', help='')
    parser.add_argument('--code_dir', type=str, default='.')
    parser.add_argument('--model_dir', type=str, default='numnet_plus_345_LR_0.0005_BLR_1.5e-05_WD_5e-05_BWD_0.01_False')
    parser.add_argument('--max_epoch', type=str, default=10)
    parser.add_argument('--batch_size', type=str, default=16)
    parser.add_argument('--eval_batch_size', type=str, default=16)
    parser.add_argument('--TMSPAN', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    DATA_ROOT_DIR = os.path.join('..', args.data_root_dir)
    DATA_DIR = os.path.join(DATA_ROOT_DIR, args.data_dir)

    DATA_PATH = DATA_DIR # data path
    DUMP_PATH = os.path.join(DATA_DIR, 'prediction_drop_dataset_dev.json' )# result path
    INF_PATH = os.path.join(DATA_DIR, 'drop_dataset_dev.json') # origin data path
    PRE_PATH = os.path.join(DATA_DIR, args.model_dir, 'checkpoint_best.pt') # pretained model path

    BERT_CONFIG = "--roberta_model {}/roberta.large".format(DATA_PATH)
    if args.TMSPAN:
        "Use tag_mspan model..."
        MODEL_CONFIG = "--gcn_steps 3 --use_gcn --tag_mspan"
    else:
        "Use mspan model..."
        MODEL_CONFIG = "--gcn_steps 3 --use_gcn"

    print('start to evaluation')
    TEST_CONFIG = "--eval_batch_size {} --pre_path {} --data_mode dev --dump_path {} \
                 --inf_path {}".format(args.eval_batch_size, PRE_PATH, DUMP_PATH, INF_PATH)

    eval_cmd = ' '.join(['python', 'roberta_predict.py', TEST_CONFIG, BERT_CONFIG, MODEL_CONFIG])
    os.system(eval_cmd)

