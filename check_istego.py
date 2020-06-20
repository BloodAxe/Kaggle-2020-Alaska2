import argparse
import os

from alaska2.dataset import get_istego100k_train, INPUT_IMAGE_KEY

parser = argparse.ArgumentParser()
parser.add_argument(
    "-dd2", "--data-dir-istego", type=str, default=os.environ.get("KAGGLE_2020_ISTEGO100K", "d:\datasets\istego100k")
)
args = parser.parse_args()


data_dir_istego = args.data_dir_istego

get_istego100k_train(data_dir_istego, fold=0, features=[INPUT_IMAGE_KEY], output_size="random_crop")
get_istego100k_train(data_dir_istego, fold=1, features=[INPUT_IMAGE_KEY], output_size="random_crop")
get_istego100k_train(data_dir_istego, fold=2, features=[INPUT_IMAGE_KEY], output_size="random_crop")
get_istego100k_train(data_dir_istego, fold=3, features=[INPUT_IMAGE_KEY], output_size="random_crop")
