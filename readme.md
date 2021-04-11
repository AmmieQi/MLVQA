[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
  
# Multitask Learning for Visual Question Answering (MLVQA)

## The architecture
![MLVQA](https://github.com/dr-majie/MLVQA/blob/master/mlvqa2.png)

## How to run
``python run.py --RUN=train --MODEL=mlvqa --DATASET=vqa --GPU=0,1 --SEED=123``

## Requirements

| Package   | Version   |
| --------- | ---------:|
| nltk      | >= 3.4.5  |
| numpy     | >= 1.19.0 |
| spacy     | >= 2.3.2  |
| torch     | >= 1.1.0  |
| pyyaml    | -         |

License
----
This project is released under the [Apache 2.0 license](https://github.com/MILVLG/openvqa/blob/master/LICENSE).
