from sklearn.metrics import f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode
from pyvi import ViTokenizer
import pandas as pd


def scoring(valid_df, valid_preds):
    result = {}
    result['valid_f1_macro'] = f1_score(valid_df['class_label'], valid_preds, average='macro')
    print("Valid F1 macro score", result['valid_f1_macro'] )
    result['valid_f1_micro'] = f1_score(valid_df['class_label'], valid_preds, average='micro')
    print("Valid F1 micro score", result['valid_f1_micro'] )
    detail_results = valid_df.groupby(['class_label','class_name']).query_id.nunique().reset_index().sort_values(by='class_label')
    detail_results['valid_f1_score'] = f1_score(valid_df['class_label'], valid_preds, average=None)
    detail_results['precision_score'] = precision_score(valid_df['class_label'], valid_preds, average=None)
    detail_results['recall_score'] = recall_score(valid_df['class_label'], valid_preds, average=None)
    result['detail_results'] = detail_results
    return result

def plot_result(result, title):
    ax1 = sns.set_style(style='darkgrid', rc=None )
    fig, ax1 = plt.subplots(figsize=(12,6))
    sns.barplot(data = result['detail_results'], x='class_name', y='query_id', alpha=0.5, ax=ax1)
    plt.xticks(rotation=70)
    ax2 = ax1.twinx()
    sns.lineplot(data = result['detail_results'][['valid_f1_score','precision_score', 'recall_score']], marker='o', sort = False, ax=ax2)
    plt.title(f'{title}\nValid F1 micro average score: {result["valid_f1_micro"]}\nValid F1 macro average score: {result["valid_f1_macro"]}')
    plt.savefig(f'results/{title}.png')

def load_data(fn='train'):
    df = pd.read_csv(f"dataset/{fn}.csv", sep='|')
    df.columns = [i.lower().replace(" ",'_') for i in df.columns]

    query_df = pd.read_csv(f'dataset/{fn}_serp.csv', sep='|')
    query_df.columns = [i.lower().replace(" ",'_') for i in query_df.columns]
    df = pd.merge(df, query_df, on = 'query_id')
    with open("dataset/category_list") as f:
        label2idx = eval(f.read())
    idx2label = {v:k for k,v in label2idx.items()}
    df['class_name'] = df.class_label.map(idx2label)
    df.fillna("NaN",inplace=True)
    return df



class MetaFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, individual_models):
        self.models = individual_models
        self.column_names = ['query','title_1','title_2','title_3','content_1','content_2','content_3']
    def fit(self, X, y= None):
        return self

    def transform(self, X,y=None):
        meta_columns = []
        for col in self.column_names:
            X_model = self.models[col]['pipeline'].transform(X)
            X[f'{col}_pred'] = self.models[col]['model'].predict(X_model)
            meta_columns.append(f'{col}_pred')
        for col in self.column_names[1:]:
            X[f'{col}_num_bolds'] = X[col].str.count("<b>")
            meta_columns.append(f'{col}_num_bolds')
        return X[meta_columns]


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y = None):
        return X[self.column]

class TagRemoval(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X.str.replace(r'<[^>]+>','')

class ToLowercase(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X.str.lower()

class QuoteRemoval(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X.str.replace("&quot;",'')

class CleanTextField(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X

class ToneRemoval(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.map(unidecode)

class Tokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y = None):
        return X.map(lambda x: ViTokenizer.tokenize(x).replace("_",' '))


class TokenizerWithComplex(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y = None):
        return X.map(lambda x: ViTokenizer.tokenize(x))