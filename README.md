Code of our paper "Few-shot Relation Extraction via Entity Feature Enhancement and Attention-based Prototypical Network", which is based on PyTorch(1.7.1+cu110) and transformers(4.15.0).

### Data
Bridge-FewRel dataset is a domain dataset for bridge inspection, since the Bridge-FewRel dataset contains information about a large number of bridges in China, it is temporarily unable to release the data.

The TinyRel-CM dataset is proposed by [MICK: A Meta-Learning Framework for Few-shot Relation Classification with Small Training Data](https://dl.acm.org/doi/10.1145/3340531.3411858).

The FewRel1.0 dataset is proposed by [FewRel: A Large-Scale Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation](https://aclanthology.org/D18-1514/).


### Model
The pre-trained RoBERTa model can be downloaded in [Google Drive](https://drive.google.com/open?id=1H6f4tYlGXgug1DdhYzQVBuwIGAkAflwB).

### Experiments

Experiment in the Bridge-FewRel dataset(5-way-1-shot):
```bash
python train_demo.py \
    --trainN 5 --N 5 --K 1 --Ke 1 --Q 1 --train_iter 8000 --test_iter 5000 --val_iter 500 --val_step 500\
    --train Bridge-FewRel/bridge-train_have_negative \
    --val Bridge-FewRel/bridge-train_have_negative \
    --test Bridge-FewRel/bridge-train_have_negative\
    --batch_size 4 --max_length 128
```

Experiment in the TinyRel-CM dataset(5-way-15-shot):
```bash
python train_demo.py \
    --trainN 5 --N 5 --K 15 --Ke 15 --Q 1 --train_iter 8000 --test_iter 5000 --val_iter 500 --val_step 500\
    --train TinyRel-CM/yiliao_train_for_DD \
    --val TinyRel-CM/yiliao_test_for_DD \
    --test TinyRel-CM/yiliao_test_for_DD \
    --batch_size 2 --max_length 256\
```

Experiment in the FewRel 1.0 dataset(以5-way-1-shot):
```bash
python train_demo.py\
    --trainN 5 --N 5 --K 1 --Ke 1 --Q 1 --train_iter 30000 --test_iter 20000 --val_iter 1000 --val_step 20000\
    --train FewRel1.0/train_wiki \
    --val FewRel1.0/val_wiki \
    --test FewRel1.0/test_wiki \
    --batch_size 4 --max_length 128
```
Notes:when performing experiments in the FewRel 1.0 dataset, the `item['h'][1],item['t'][1]` from the `getraw` method in `data_loader.py` should be revised as the `item['h'][2][0],item['t'][2][0]`.