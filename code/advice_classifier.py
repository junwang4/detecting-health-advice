import os, sys, time, random
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bert_sklearn import BertClassifier, load_model
from utils import *

init_settings()

class AdviceClassifier:
    def __init__(self, working_folder='working'):
        self.method = 'regular'
        self.kfolds = K_FOLDS
        self.bert_model = BERT_MODEL
        self.text_column_name = TEXT_COLUMN_NAME

        self.data_folder = DATA_FOLDER
        self.working_folder = get_or_create_dir(working_folder)
        self.model_folder = get_or_create_dir(f'{self.working_folder}/model')
        self.pred_folder = get_or_create_dir(f'{self.working_folder}/pred')

        self.annotated_file = ANNOTATED_FILE
        self.data_file_to_work_discussion = DATA_FILE_TO_WORK_DISCUSSION
        self.data_file_to_work_unstructured_abstract = DATA_FILE_TO_WORK_UNSTRUCTURED_ABSTRACT

    def get_test_data_for_section(self, sentence_section='discussion'):
        if sentence_section == 'discussion':
            return f'{self.data_folder}/{self.data_file_to_work_discussion}'
        else:
            return f'{self.data_folder}/{self.data_file_to_work_unstructured_abstract}'

    def get_train_data_csv_fpath(self):
        fpath = f'{self.data_folder}/{self.annotated_file}'
        print('- annotated csv file:', fpath)
        if os.path.exists(fpath):
            return fpath
        else:
            print('- error: training csv file not exists:', fpath)
            sys.exit()

    def read_train_data(self):
        return pd.read_csv(self.get_train_data_csv_fpath(), usecols=[self.text_column_name, LABEL_COLUMN_NAME], encoding = 'utf8', keep_default_na=False)

    def get_model_bin_file(self, fold=0):
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            print(f'\ncreate a new folder for storing BERT model: "{self.model_folder}"\n')
        if fold>=0:
            return f'{self.model_folder}/{TAG}_K{self.kfolds}_epochs{EPOCHS}_{fold}.bin.{BERT_MODEL}'
        elif fold==-1:
            return f'{self.model_folder}/{TAG}_full_epochs{EPOCHS}.bin.{BERT_MODEL}'
        else:
            print('- wrong value for fold:', fold)
            sys.exit()

    def get_pred_csv_file(self, mode='train', sentence_section=None):
        if mode == 'train':
            fpath = f'{self.pred_folder}/{TAG}_{mode}_K{self.kfolds}_epochs{EPOCHS}.csv.{BERT_MODEL}'
            if self.method == 'augmented':
                fpath += f'.section{self.use_section}_citation{self.use_citation}_pasttense{self.use_past_tense}'

        elif mode == 'apply':
            #sentence_section = kwargs['sentence_section']
            fpath = f'{self.pred_folder}/{TAG}_{mode}_epochs{EPOCHS}_{sentence_section}.csv.{BERT_MODEL}'
        else:
            print('- wrong mode:', mode, '\n')
            sys.exit()
        print('- get pred csv file:', fpath)
        return fpath

    def get_train_test_data(self, fold=0, df=None):
        df[self.text_column_name] = df[self.text_column_name].apply(lambda x: x.strip())
        kf = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=RANDOM_STATE)
        cv = kf.split(df[self.text_column_name], df[LABEL_COLUMN_NAME])

        for i, (train_index, test_index) in enumerate(cv):
            if i == fold:
                break
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        print(f"\nALL: {len(df)}   TRAIN: {len(train)}   TEST: {len(test)}")
        return train, test


    def train_model(self, df_train, model_file_to_save, val_frac=0.1):
        X_train = df_train[self.text_column_name]
        y_train = df_train[LABEL_COLUMN_NAME]

        model = BertClassifier(bert_model=self.bert_model, random_state=RANDOM_STATE, \
                                max_seq_length=MAX_SEQ_LENGTH, \
                                train_batch_size=TRAIN_BATCH_SIZE, learning_rate=LEARNING_RATE, \
                                epochs=EPOCHS, validation_fraction=val_frac)
        print(model)
        model.fit(X_train, y_train)

        if model_file_to_save:
            model.save(model_file_to_save)
            print(f'\n- model saved to: {model_file_to_save}\n')
        return model


    def train_one_full_model(self):
        df_train = self.read_train_data()

        model_file_to_save = self.get_model_bin_file(fold=-1) # -1: for one full model
        val_frac = 0.0
        self.train_model(df_train, model_file_to_save, val_frac=val_frac)


    def train_KFold_model(self, val_frac=0.0):
        if self.method == 'regular':
            df = self.read_train_data()
            print('- label value counts:')
            print(df[LABEL_COLUMN_NAME].value_counts())
        else:
            df = None

        y_test_all, y_pred_all = [], []
        results = []
        df_out_proba = None
        for fold in range(self.kfolds):
            #if fold != 2: continue
            train_data, test_data = self.get_train_test_data(fold, df)

            if self.method == 'regular' and SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD:
                model_file = self.get_model_bin_file(fold)
            else:
                model_file = ''

            model = self.train_model(train_data, model_file, val_frac=val_frac)

            X_test = test_data[self.text_column_name]
            y_test = test_data[LABEL_COLUMN_NAME]
            y_test_all += y_test.tolist()

            y_proba = model.predict_proba(X_test)
            del model

            tmp = pd.DataFrame(data=y_proba, columns=[f'c{i}' for i in range(NUM_CLASSES)])
            tmp['confidence'] = tmp.max(axis=1)
            tmp['winner'] = tmp.idxmax(axis=1)
            tmp[self.text_column_name] = X_test.tolist()
            tmp[LABEL_COLUMN_NAME] = y_test.tolist()
            df_out_proba = tmp if df_out_proba is None else pd.concat((df_out_proba, tmp))

            y_pred = [int(x[1]) for x in tmp['winner']]
            y_pred_all += y_pred

            acc = accuracy_score(y_pred, y_test)
            res = precision_recall_fscore_support(y_test, y_pred, average='macro')
            print(f'\nAcc: {acc:.3f}      F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

            #item = {'Acc': acc, 'weight': len(test_data)/len(df), 'size': len(test_data)}
            item = {'Acc': acc, 'size': len(test_data)}
            item.update({'P':res[0], 'R':res[1], 'F1':res[2]})
            for cls in np.unique(y_test):
                res = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[cls])
                for i, scoring in enumerate('P R F1'.split()):
                    item['{}_{}'.format(scoring, cls)] = res[i][0]
            results.append(item)

            acc_all = np.mean(np.array(y_pred_all) == np.array(y_test_all))
            res = precision_recall_fscore_support(y_test_all, y_pred_all, average='macro')
            print( f'\nAVG of {fold+1} folds  |  Acc: {acc_all:.3f}    F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

        # show an overview of the performance
        df_2 = pd.DataFrame(list(results)).transpose()
        df_2['avg'] = df_2.mean(axis=1)
        df_2 = df_2.transpose()
        df_2['size'] = df_2['size'].astype(int)
        display(df_2)#.drop('weight', axis=1))

        # put together the results of all k-fold tests and save
        output_pred_csv_file_train = self.get_pred_csv_file(mode='train')
        df_out_proba.to_csv(output_pred_csv_file_train, index=False, float_format="%.3f")
        print(f'\noutput all {self.kfolds}-fold test results to: "{output_pred_csv_file_train}"\n')


    def apply_trained_model_to_discussion_sentences(self):
        self.apply_trained_model_to_new_sentences(sentence_section='discussion')

    def apply_trained_model_to_unstructured_abstract_sentences(self):
        self.apply_trained_model_to_new_sentences(sentence_section='unstructured_abstract')

    def apply_trained_model_to_new_sentences(self, sentence_section='discussion'):
        data_file = self.data_file_to_work_discussion if sentence_section=='discussion' else self.data_file_to_work_unstructured_abstract
        fpath_data = f'{self.data_folder}/{data_file}'

        nrows = None
        cols = 'pmid sentence'.split()
        df = pd.read_csv(fpath_data, nrows=nrows, keep_default_na=False, usecols=cols)

        output_pred_file = self.get_pred_csv_file('apply', sentence_section=sentence_section)
        print('\n- predictions to save to:', output_pred_file)

        y_prob = None
        for fold in range(K_FOLDS):
            #if not fold==0: continue
            model_file = self.get_model_bin_file(fold)
            print(f'\n- use trained model: {model_file}\n')
            model = load_model(model_file)
            model.eval_batch_size = 32
            y_prob_ = model.predict_proba(df.sentence)
            y_prob = y_prob_ if y_prob is None else y_prob + y_prob_
        y_prob /= K_FOLDS

        df_out = pd.DataFrame(data=y_prob, columns=[f'c{i}' for i in range(NUM_CLASSES)])
        df_out['confidence'] = df_out.max(axis=1)
        df_out['winner'] = df_out.idxmax(axis=1)
        df_out['pmid'] = df['pmid']
        df_out['sentence'] = df['sentence']

        df_out.to_csv(output_pred_file, index=False, float_format="%.3f")
        print(f'\n- predictions saved to: {output_pred_file}\n')


    def evaluate_and_error_analysis(self):
        df = pd.read_csv(self.get_pred_csv_file(mode='train')) # -2: a flag indicating putting together the results on all folds
        df['pred'] = df['winner'].apply(lambda x:int(x[1])) # from c0->0, c1->1, c2->2, c3->3

        print('\nConfusion Matrix:\n')
        cm = confusion_matrix(df[LABEL_COLUMN_NAME], df.pred)
        print(cm)

        print('\n\nClassification Report:\n')
        print(classification_report(df[LABEL_COLUMN_NAME], df.pred, digits=3))

        out = ["""
<style>
* {font-family:arial}
body {width:900px;margin:auto}
.wrong {color:red;}
.hi1 {font-weight:bold}
</style>
<div><table cellpadding=10>
    """]

        row = f'<tr><th><th><th colspan=4>Predicted</tr>\n<tr><td><td>'
        label_name = LABEL_NAME
        for i in range(NUM_CLASSES):
            row += f"<th>{label_name[i]}"
        for i in range(NUM_CLASSES):
            row += f'''\n<tr>{'<th rowspan=4>Actual' if i==0 else ''}<th align=right>{label_name[i]}'''
            for j in range(NUM_CLASSES):
                row += f'''<td align=right><a href='#link{i}{j}'>{cm[i][j]}</a></td>'''
        out.append(row + "</table>")

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                row = f"<div id=link{i}{j}><h2>{label_name[i]} => {label_name[j]}</h2><table cellpadding=10>"
                label_names = ' '.join([f'<th>{label_name[k]}</th>' for k in range(len(label_name))])
                row += f'<tr> <th></th> <th>Sentence</th> <th>Label</th> {label_names} <th>mark</th> </tr>'

                out.append(row)

                df_ = df[(df[LABEL_COLUMN_NAME]==i) & (df.pred==j)]
                df_ = df_.sort_values('confidence', ascending=False)

                cnt = 0
                for idx, row in df_.iterrows():
                    sentence, label, pred = row[self.text_column_name], row[LABEL_COLUMN_NAME], row['pred']
                    cnt += 1
                    td_mark = "" if label == pred else "<span class=wrong>oops</span>"

                    td_confidence_list = []
                    c_max = max([row[f'c{k}'] for k in range(NUM_CLASSES)])
                    for k in range(NUM_CLASSES):
                        c = row[f'c{k}']
                        is_max = int(c >= c_max)
                        td_confidence_list.append(f'<td valign=top class=hi{is_max}>{c:.2f}</td>')

                    item = f"""<tr><th valign=top>{cnt}.
                        <td valign=top width=70%>{sentence}
                        <td valign=top>{label_name[label]}
                        {''.join(td_confidence_list)}
                        <td valign=top>{td_mark}</tr>"""
                    out.append(item)

                out.append('</table></div>')

        folder_out = get_or_create_dir(f'{self.working_folder}/{HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT}')
        fpath_out = f'{folder_out}/error_analysis_{TAG}.html'
        with open(fpath_out, 'w') as fout:
            fout.write('\n'.join(out))
            print(f'\n- Error analysis result saved to: "{fpath_out}"\n')


    def postprocessing_filter_general(self, sentence_section, new_features, clf=LinearSVC):
        df_pred = pd.read_csv(self.get_pred_csv_file('apply', sentence_section=sentence_section))
        df_pred['winner'] = df_pred.winner.apply(lambda x:int(x[1]))

        df = pd.read_csv(self.get_test_data_for_section(sentence_section))
        df = pd.concat((df, df_pred[['winner']]), axis=1)
        #print(df.iloc[0])
        print('\n********************************\n* before postprocessing filter *\n*')
        print(classification_report(df.label, df.winner, digits=3))

        basic_features = ['c0', 'c1', 'c2']

        df_tmp = pd.get_dummies(df['winner'])
        df_tmp.columns = basic_features
        df = pd.concat((df, df_tmp), axis=1)
        #print(df.iloc[0])
        columns = basic_features + new_features
        if sentence_section=='unstructured_abstract':
            print('\n--- convert rel_loc to 0 or 1 ---\n')
            df.rel_loc = df.rel_loc.apply(lambda x: int(x>=0.5))

        df['combine'] = df[columns].values.tolist()
        #print(df.head())

        X = df['combine'].values
        y = df['label'].values

        y_pred = cross_val_predict(clf, list(X), y, cv=5)
        print('\n*******************************\n* after postprocessing filter *\n*')
        print(classification_report(y, y_pred,  digits=3))

    def postprocessing_filter_for_unstructured_abstracts(self):
        new_features = ['rel_loc']
        self.postprocessing_filter_general('unstructured_abstract', new_features, DecisionTreeClassifier())

    def postprocessing_filter_for_discussions(self):
        new_features = ['citation_mentioned', 'past_tense']
        self.postprocessing_filter_general('discussion', new_features,  DecisionTreeClassifier())


    def LinearSVC_cross_validation(self, vectorizer=TfidfVectorizer):
        from sklearn.svm import LinearSVC
        df = self.read_train_data()
        vec = vectorizer(ngram_range=(1, 2), min_df=2, token_pattern='[^ ]+')
        df.sentence = df.sentence.apply(lambda x: x.lower())
        X_text, y = df.sentence, df.label
        X_sparse = vec.fit_transform(X_text).astype(float)
        clf = LinearSVC(C=1)
        y_pred = cross_val_predict(clf, X_sparse, y, cv=K_FOLDS)
        print(f'\n********************************************\n* LinearSVC {K_FOLDS}-fold cross-validation report *\n*')
        print(classification_report(df.label, y_pred, digits=3))


def main():
    clf = AdviceClassifier(working_folder='./working5')

    clf.LinearSVC_cross_validation()

    clf.train_KFold_model()
    clf.evaluate_and_error_analysis()

    #clf.apply_trained_model_to_discussion_sentences()
    clf.apply_trained_model_to_unstructured_abstract_sentences()

    clf.postprocessing_filter_for_unstructured_abstracts()


if __name__ == "__main__":
    tic = time.time()
    main()
    print(f'\ntime used: {time.time()-tic:.1f} seconds\n')
