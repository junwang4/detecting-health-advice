import time, sys
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils import *
from advice_classifier import AdviceClassifier

class AugmentedAdviceClassifier(AdviceClassifier):
    def __init__(self, use_section=1, use_citation=1, use_past_tense=1):
        super().__init__('working9')

        self.method = 'augmented'
        self.text_column_name = 'text_SEP'
        self.use_section = use_section
        self.use_citation = use_citation
        self.use_past_tense = use_past_tense
        self.fields_to_use = []
        if use_section:
            self.fields_to_use.append('section')
        if use_citation:
            self.fields_to_use.append('citation_mentioned')
        if use_past_tense:
            self.fields_to_use.append('past_tense')
        self.fields_to_use.append('sentence')
        print('\n- Features to use:', self.fields_to_use, '\n')

    def _gen_text_SEP(self, x):
        convert_map = {'0':'No', '1':'Yes'}
        def convert_01_to_words(s):
            s = str(s)
            return convert_map[s] if s in convert_map else s
        return " [SEP] ".join([convert_01_to_words(x[col]) for col in self.fields_to_use])

    def get_train_test_data(self, fold, df=None):
        missing_value = ''
        section_structured_abstract = 'structured abstract'
        section_discussion = 'discussion'

        df_train_6k = pd.read_csv(f'{DATA_FOLDER}/{ANNOTATED_FILE}')
        df_train_6k['section'] = section_structured_abstract
        df_train_6k['citation_mentioned'] = missing_value
        df_train_6k['past_tense'] = missing_value

        df_discussion = pd.read_csv(f'{DATA_FOLDER}/{DATA_FILE_TO_WORK_DISCUSSION}')
        df_discussion = df_discussion.sample(frac=1, random_state=RANDOM_STATE)
        df_discussion['section'] = section_discussion
        df_discussion.drop('rel_loc', axis=1, inplace=True)

        skf = GroupKFold(n_splits=K_FOLDS) # no parameter for random_state which generates fixed result
        splits = skf.split(df_discussion.sentence, df_discussion.label, df_discussion.pmid)

        for i, (train_index, test_index) in enumerate(splits):
            if i != fold: continue
            df_discussion_train = df_discussion.iloc[train_index]
            df_discussion_test = df_discussion.iloc[test_index]
            df_train = pd.concat((df_train_6k[df_discussion.columns], df_discussion_train))

        train = df_train.copy()
        test = df_discussion_test.copy()
        print(f'- train size: {len(train)}      test size: {len(test)}\n')

        train.fillna('', inplace=True)
        test.fillna('', inplace=True)
        if self.text_column_name == 'text_SEP':
            train[self.text_column_name] = train.apply(self._gen_text_SEP, axis=1)
            test[self.text_column_name] = test.apply(self._gen_text_SEP, axis=1)
        print(train.iloc[0][self.text_column_name][:80], '...')
        print(test.iloc[0][self.text_column_name][:80], '...')
        print('- unique PMID in train data:', len(train.pmid.unique()), ' for test data:', len(test.pmid.unique()), '\n')
        return train, test


def main():
    clf = AugmentedAdviceClassifier(use_section=1, use_citation=1, use_past_tense=1)
    clf.train_KFold_model()
    clf.evaluate_and_error_analysis()


if __name__ == "__main__":
    tic = time.time()
    main()
    print(f'time used: {time.time()-tic:.0f} seconds')
