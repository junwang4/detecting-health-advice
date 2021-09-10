**NOTE**: Our running environment is 
```
Ubuntu: 16.04
PyTorch: 1.8.2 
CUDA: 10.1
GPU: TITAN Xp
```

### STEP 1. Prerequisite

Install bert-sklearn --
a scikit-learn wrapper to finetune BERT model based on the Huggingface's pytorch transformer.
```
git clone -b master https://github.com/charles9n/bert-sklearn
cd bert-sklearn
pip install .
```

Install package `fire` (for use with command line interfaces)
```
pip install fire
```


### STEP 2. Get the repo, including code and data
```
git clone https://github.com/junwang4/detecting-health-advice
cd detecting-health-advice
```

### STEP 3. Model evaluation
```
cd code
```
**NOTE:** Check `Makefile` for quick access to the following commands

#### 3.1 To evaluate the performance of the model, say, 5-fold cross-validation 

First, generate a prediction file as the result of training and testing each of the 5 folds
```
python run.py advice_classifier --task=train_KFold_model
```
This will take as input the annotated dataset `data/annotations_structured_abstract.csv`,
and assemble the prediction results from each fold into `code/working/pred/[TAG]_train_K5_epochs3.csv`
Also, 5 fine-tuned BERT models will be saved at `code/working/model/[TAG]_K5_epochs3_[fold].bin`

Second, display the evaluation results

```
python run.py advice_classifier --task=evaluate_and_error_analysis
```

Result (after assembling results from the K=5 runs):
```
              precision    recall  f1-score   support

           0      0.963     0.951     0.957      3575
           1      0.908     0.922     0.915      1482
           2      0.917     0.941     0.928       925

    accuracy                          0.942      5982
   macro avg      0.929     0.938     0.933      5982
```

#### 3.2 Apply the above fine-tuned models to the unseen sentences

For the unstructured abstracts:
```
python run.py advice_classifier --task=apply_trained_model_to_discussion_sentences
```
For the discussion/conclusion sections:

```
python run.py advice_classifier --task=apply_trained_model_to_unstructured_abstract_sentences
```

#### 3.3 Postprocess the above results with filtering rules

For the unstructured abstracts:
```
python run.py advice_classifier --task=postprocessing_filter_for_unstructured_abstracts
```
For the discussion/conclusion sections:
```
python run.py advice_classifier --task=postprocessing_filter_for_discussions
```

### STEP 4. A data and feature augmentation approach for predicting discussion sentences

This is an approach of augmenting training data with part of annotated discussion sentences,
and then applying the trained model to the remaining discussion sentences.
First, we split the discussion sentences into K groups (grouped by paper ID).
When running evaluation on group k (k=1,2,...K), 
the training data consists of all structured abstract sentences 
and the discussion sentences from the remaining K-1 groups (except group k).
The newly constructed text will take the following form:
```
section name [SEP] citation mentioned [SEP] past tense [SEP] The original sentence
```

Specifically, for structured abstract sentences:
```
structured abstract [SEP] [SEP] [SEP] The original sentence
```

And for discussion sentences:
```
discussion [SEP] No [SEP] Yes [SEP] The original sentence
```

Run the following commands
```
python run.py augmented_advice_classifier --task=train_KFold_model --feature_section=1 --feature_citation=1 --feature_past_tense=1
python run.py augmented_advice_classifier --task=evaluate_and_error_analysis --feature_section=1 --feature_citation=1 --feature_past_tense=1 
```

Result (after assembling results from the K=5 runs):
```
              precision    recall  f1-score   support

           0      0.991     0.990     0.990      3635
           1      0.827     0.883     0.854       162
           2      0.892     0.859     0.875       135

    accuracy                          0.981      3932
   macro avg      0.903     0.911     0.907      3932
```
