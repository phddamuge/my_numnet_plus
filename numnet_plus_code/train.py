import json
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--data_root_dir', type=str, default='numnet_plus_data')
    parser.add_argument('--data_dir', type=str, default='drop_dataset', help='')
    parser.add_argument('--code_dir', type=str, default='.')
    parser.add_argument('--max_epoch', type=str, default=10)
    parser.add_argument('--batch_size', type=str, default=16)
    parser.add_argument('--eval_batch_size', type=str, default=16)

    parser.add_argument('--SEED', type=int, default=345)
    parser.add_argument('--LR', type=float, default=5e-4)
    parser.add_argument('--BLR', type=float, default=1.5e-5)
    parser.add_argument('--WD', type=float, default=5e-5)
    parser.add_argument('--BWD', type=float, default=0.01)
    parser.add_argument('--TMSPAN', default=False, action='store_true')

    args, _ = parser.parse_known_args()
    # os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'
    DATA_ROOT_DIR = os.path.join('..', args.data_root_dir)
    DATA_DIR = os.path.join(DATA_ROOT_DIR, args.data_dir)
    CODE_DIR = args.code_dir
    if args.TMSPAN:
        CACHED_TRAIN = os.path.join(DATA_DIR, 'tmspan_cached_roberta.pkl')
        CACHED_DEV = os.path.join(DATA_DIR, 'tmspan_cached_roberta_dev.pkl')
        MODEL_CONFIG = "--gcn_steps 3 --use_gcn --tag_mspan"
        if not os.path.exists(CACHED_TRAIN) or not os.path.exists(CACHED_DEV):
            print('Preparing cached data.')
            cmd = 'python prepare_roberta_data.py --input_path {} --output_dir {} --tag_mspan'.format(DATA_DIR,
                                                                                                      DATA_DIR)
            os.system(cmd)
    else:
        CACHED_TRAIN = os.path.join(DATA_DIR, 'cached_roberta_train.pkl')
        CACHED_DEV = os.path.join(DATA_DIR, 'cached_roberta_dev.pkl')
        MODEL_CONFIG = "--gcn_steps 3 --use_gcn"
        if not os.path.exists(CACHED_TRAIN) or not os.path.exists(CACHED_DEV):
            print('Preparing cached data.')
            cmd = 'python prepare_roberta_data.py --input_path {} --output_dir {}'.format(DATA_DIR, DATA_DIR)
            os.system(cmd)

    SAVE_DIR = "{}/numnet_plus_{}_LR_{}_BLR_{}_WD_{}_BWD_{}_{}".format(DATA_DIR, args.SEED, args.LR, args.BLR, args.WD,
                                                                       args.BWD, str(args.TMSPAN))
    DATA_CONFIG = "--data_dir {} --save_dir {}".format(DATA_DIR, SAVE_DIR)

    TRAIN_CONFIG = "--batch_size {} --eval_batch_size {} --max_epoch {} --warmup 0.06 --optimizer adam \
                  --learning_rate {} --weight_decay {} --seed {} --gradient_accumulation_steps 4 \
                  --bert_learning_rate {} --bert_weight_decay {} --log_per_updates 10 --eps 1e-6".format(
        args.batch_size, args.eval_batch_size, args.max_epoch, args.LR, args.WD, args.SEED, args.BLR, args.BWD)

    BERT_CONFIG = "--roberta_model {}/roberta.large".format(DATA_DIR)

    print('start to train')
    train_cmd = ' '.join(
        ['python', os.path.join(CODE_DIR, 'roberta_gcn_cli.py'), DATA_CONFIG, TRAIN_CONFIG, BERT_CONFIG, MODEL_CONFIG])
    print('execute cmd ', train_cmd)
    os.system(train_cmd)
