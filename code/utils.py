import torch
import os, random
import pandas as pd
import numpy  as np
from configparser import ConfigParser

DATA_FOLDER = '../data'

LABEL_NAME = {0:'No', 1:'Weak', 2:'Strong'}
NUM_CLASSES = len(LABEL_NAME)

LABEL_COLUMN_NAME = 'label'
TEXT_COLUMN_NAME = 'sentence'

config = ConfigParser()
config.read('settings.ini')

GPU_ID = config.get('common', 'GPU_ID')
print(f'\n- GPU_ID: {GPU_ID}', end='')
if str(GPU_ID) == "0":
    print('   // GPU with index 0 is the fastest; in our experiment, it is TITAN Xp\n')
else:
    print()

BERT_MODEL = config.get('common', 'BERT_MODEL')
print(f'- BERT model: {BERT_MODEL}')

RANDOM_STATE = config.getint('common', 'RANDOM_STATE')
print(f'- RANDOM_STATE: {RANDOM_STATE}')

K_FOLDS = config.getint('common', 'K_FOLDS')
EPOCHS = config.getint('common', 'EPOCHS')
print(f'- K_FOLDS: {K_FOLDS}')
print(f'- EPOCHS: {EPOCHS}')

MAX_SEQ_LENGTH = config.getint('common', 'MAX_SEQ_LENGTH')
TRAIN_BATCH_SIZE = config.getint('common', 'TRAIN_BATCH_SIZE')
LEARNING_RATE = config.getfloat('common', 'LEARNING_RATE')
print(f'- MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}')
print(f'- TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE}')
print(f'- LEARNING_RATE: {LEARNING_RATE}')
print()

SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD = config.getboolean('common', 'SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD')
HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT = config.get('common', "HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT")
ANNOTATED_FILE = config.get('common', "ANNOTATED_FILE")
DATA_FILE_TO_WORK_DISCUSSION = config.get('common', "DATA_FILE_TO_WORK_DISCUSSION")
DATA_FILE_TO_WORK_UNSTRUCTURED_ABSTRACT = config.get('common', "DATA_FILE_TO_WORK_UNSTRUCTURED_ABSTRACT")

TAG = 'v1'


def init_settings(random_state=RANDOM_STATE):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    # for reproducibility
    torch.backends.cudnn.deterministic = True
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)

    pd.options.display.max_colwidth = 80
    pd.options.display.width = 1000
    pd.options.display.precision = 3
    np.set_printoptions(precision=3)


def get_or_create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def display_distribution_of_study_design_and_advice_type():
    fpath_ann = f'{DATA_FOLDER}/{ANNOTATED_FILE}'
    df = pd.read_csv(fpath_ann)
    #pmid,sentence,section,design,label
    #27901241,: States of frailty were highly present in the hospital environment.,str_abstract,Cross-Sectional,0

    ct = pd.crosstab(df.label, df.design, margins=True)
    print(ct.columns)

    # change the display order of columns
    cols = ['RCT', 'Cross-Sectional', 'Case-Control', 'Retrospective', 'Prospective']
    cols.append('All')
    print(f'\n***************************************************************\n* Cross Tabulation of two factors: "label" and "study design" *\n*\n')
    print(ct[cols])
    print()

""" # modified based on Yingya's R code
library(ggplot2)
library(dplyr)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
df <- read.csv("../data/annotations_discussion.csv" , header = T, stringsAsFactors = F)

df$label2[df$label == 0] <- "No Advice"
df$label2[df$label == 1] <- "Weak Advice"
df$label2[df$label == 2] <- "Strong Advice"
summary(df$label2)

df$label2 <- ordered(df$label2)
df$label2 <- ordered(df$label2, levels = c("No Advice", "Weak Advice", "Strong Advice"))

ggplot(df, aes(x=rel_loc, color = label2, fill = label2)) +
  geom_histogram(alpha=0.5, bins = 5, boundary= 0) +
  facet_grid(. ~ label2, ) +
  scale_color_manual(values=c('darkgray', 'darkgray', 'darkgray')) +
  scale_fill_manual(values=c('lightgray', 'lightgray', 'lightgray')) +
  labs(x="Location", y = "Number of sentences") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=10),
        axis.text.y = element_text(size=10),
    axis.text=element_text(size=9),
        axis.title=element_text(size=12)) +
  xlim(c(0, 1)) + ylim(c(0, 1000)) + #theme_classic() +
  theme(legend.position = "none")

figname = '/tmp/a.pdf'
ggsave(figname, plot=last_plot(), width=4, height=3)
embedFonts(figname, outfile = figname)
"""
def plot_advice_position_distribution_of_discussion_sentences():
    print('\nFor better visualization effect, check the R code given in the above comment block\n')

    import seaborn as sns
    from matplotlib import pyplot as plt

    fpath = f'{DATA_FOLDER}/{DATA_FILE_TO_WORK_DISCUSSION}'
    # label,pmid,sentence,rel_loc,citation_mentioned,past_tense
    df = pd.read_csv(fpath, usecols=['label', 'rel_loc'])
    df['label'] = df['label'].apply(lambda x: LABEL_NAME[x])
    df.rename(columns={'rel_loc':'Relative location'}, inplace=True)

    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    sns.displot(data=df, x="Relative location", hue="label", col="label", legend=False, bins=4)
    plt.savefig('/tmp/a.pdf')



def main():
    #display_distribution_of_study_design_and_advice_type()
    plot_advice_position_distribution_of_discussion_sentences()

if __name__ == '__main__':
    main()
