set -eu

poetry run kaggle competitions submit -f $1/answer.csv -m '$2' $1
