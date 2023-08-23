# Deep Learning in 5 lines of code with fastai 


fastai makes deep learning accessible with a supportive community and free learning resources 

---

# Why just 5 lines?

- pretrained models

RA: show pretrained models intro

---

# Why fastai? High Level Abstractions for Deep Learning

- Sensible/Practical default behaviour from years of experience(Kaggle Grandmaster)

---


```python
from fastai.vision.all import *

# Data Prep
path = untar_data(URLs.PETS)/'images'


# ML in 5 lines 
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=42, label_func=is_cat, item_tfms=Resize(224))
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

```
---

path = untar_data(URLs.PETS)/'images'

RA: show path structure and example
---

```python

def is_cat(x): return x[0].isupper()
```

RA: show what x can be

---

```python
dls = ImageDataLoaders.from_name_func(path,                   # where to save files like models
				      get_image_files(path),  # where to get the images
				      valid_pct=0.2,          # 20% of data is taken out for testing the model
				      label_func=is_cat,      # the actual label: is cat or not
				      item_tfms=Resize(224))  # resize to 224x224 pixels before sending to te model
```

---

# The power of pre-trained models

```python

# from fastai.vision.all import *

learn = vision_learner(dls, resnet34, metrics=error_rate)
```
---
# Fine Tuning

```python
learn.fine_tune(1)
```
---

# Utilizing the model

```python
learn.predict()
```

---

## Datablock

```python
dls = ImageDataLoaders.from_name_func(...)
```
---
### Another Example

RA: Show Bears and Image classification example

---

### Datablock

```python
path = .... path to images .. 
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = bears.dataloaders(path)
```
---

## SegmentationDataLoader

--- 

## TextDataLoader

---

## TabularDataLoader

---

## CollabDataLoader

---

# Why fastai? Interactivity

* Designed for notebook based experimentation


---
## E.g.
RA: Demo or show with an example

What I love:
- library designed with UX in mind.
    - dls.show_batch(max_n=9, figsize=(4,4)) . Normally, we would use adhoc code to view the data or even the file explorer!
    - datablock_a.summary()
    - train
    - cleaner
    - confusion matrix

---
# Why fastai? Well designed
- RA: cite fastai paper
- RA: copy any high level diagram in paper
---
# Why fastai? Teaching
* "Make Deep Learning Boring again" RA: check quotation
* Top Down approach
* RA: FastAI course/book/forums link


---
# Caution
- Most code is actually Data Prep and Model Evaluation. RA: show the picture of tech debt that you have on linkedin.

- Accessible does not mean simple. 



---

# Next Steps
- RA: FastAI course/book/forums

---

