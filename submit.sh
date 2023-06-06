set -eux

COMPETITION=$1
MSG=$2
poetry run kaggle competitions submit -f $COMPETITION/answer.csv -m "$MSG" $COMPETITION
