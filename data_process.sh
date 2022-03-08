# create folders:
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/statement/
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/tokenized/
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/grounded/
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/paths/
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/graph/
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/triples/
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/fairseq/
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/bert/
mkdir /data/hlr/chenzheng/data/MRRG/data/wiqa/roberta/

# graph processing:
python graph_preprocess.py --run wiqa
