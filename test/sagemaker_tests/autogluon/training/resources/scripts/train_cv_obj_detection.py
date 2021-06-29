import argparse
import os
from pprint import pprint

import yaml
from autogluon.vision import ObjectDetector


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        print(f'WARN: more than one file is found in {path} directory')
    print(f'Using {file}')
    filename = f'{path}/{file}'
    return filename


def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
    return result


if __name__ == '__main__':
    # Disable Autotune
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    # ------------------------------------------------------------ Args parsing
    print('Starting AG')
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=get_env_if_present("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=get_env_if_present("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=str, default=get_env_if_present("SM_NUM_GPUS"))
    parser.add_argument("--ag_config", type=str, default=get_env_if_present("SM_CHANNEL_CONFIG"))

    args, _ = parser.parse_known_args()

    print(f'Args: {args}')

    # See SageMaker-specific environment variables: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    os.makedirs(args.output_data_dir, mode=0o777, exist_ok=True)

    config_file = get_input_path(args.ag_config)
    with open(config_file) as f:
        config = yaml.safe_load(f)  # AutoGluon-specific config

    if args.n_gpus:
        config['num_gpus'] = int(args.n_gpus)

    # ---------------------------------------------------------------- Training

    train_dataset = ObjectDetector.Dataset.from_voc(config['dataset'], splits='trainval')

    ag_predictor_args = config['ag_predictor_args']
    ag_predictor_args['path'] = args.model_dir
    ag_fit_args = config['ag_fit_args']

    print('Running training job with the config:')
    pprint(config)

    predictor = ObjectDetector(**ag_predictor_args).fit(train_dataset, **ag_fit_args)
    predictor.save(f'{args.model_dir}/predictor.pkl')

    # --------------------------------------------------------------- Inference

    test_dataset = ObjectDetector.Dataset.from_voc(config['dataset'], splits='test')
    y_pred = predictor.predict(test_dataset)
    if config.get('output_prediction_format', 'csv') == 'parquet':
        y_pred.to_parquet(f'{args.output_data_dir}/predictions.parquet')
    else:
        y_pred.to_csv(f'{args.output_data_dir}/predictions.csv')

