export KAGGLE_2020_ALASKA2=/data/alaska2

python save_dct.py -f Cover -p 0
python save_dct.py -f JMiPOD -p 0
python save_dct.py -f JUNIWARD -p 0
python save_dct.py -f UERD -p 0

python save_dct.py -f Cover -p 1
python save_dct.py -f JMiPOD -p 1
python save_dct.py -f JUNIWARD -p 1
python save_dct.py -f UERD -p 1

python save_dct.py -f Test
