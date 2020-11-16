from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from unidecode import unidecode
from utils import *
from argparse import ArgumentParser
import pickle
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, required=True)
    args = parser.parse_args()
    if args.mode == 'train':
        df = load_data("train")
        train_df, valid_df = train_test_split(df, random_state=1,
                                              test_size=0.2,
                                              stratify=df.class_label
                                              )
        input_columns = ['query', 'title_1', 'title_2',
                         'title_3', 'content_1', 'content_2', 'content_3']
        results = defaultdict(list)
        # Arguments data with tone removal
        print("Augmenting data")
        augmented_df = train_df.copy()
        for col in input_columns:
            augmented_df[col] = augmented_df[col].map(unidecode)
        train_df = pd.concat([train_df, augmented_df], ignore_index=True)
        print("Training individual model each field")
        for col in input_columns:
            print(f"Start transforming for {col}....")
            pipeline = Pipeline([
                ('column_selection', ColumnSelector(col)),
                ('tag_removal', TagRemoval()),
                ('quote_removal', QuoteRemoval()),
                ('to_lowercase', ToLowercase()),
                ('tokenize', Tokenizer()),
                ('clean', CleanTextField()),
                ('vectorizer', TfidfVectorizer(ngram_range=(1, 3), min_df=3))
            ])
            X_train = pipeline.fit_transform(train_df)
            y_train = train_df['class_label']
            X_valid = pipeline.transform(valid_df)
            y_valid = valid_df['class_label']
            clf = LinearSVC(C=1)
            print("Training...")
            clf.fit(X_train, y_train)
            valid_preds = clf.predict(X_valid)
            eval_result = scoring(valid_df, valid_preds)
            plot_result(eval_result, f'Linear SVC Model for {col}')
            eval_result['model'] = clf
            eval_result['pipeline'] = pipeline
            results[col] = eval_result
            print("*"*20 + '\n')
        print("Training Meta Model")
        kf = StratifiedKFold(n_splits=5)
        i = 0
        for train_idx, valid_idx in kf.split(valid_df, valid_df.class_label):
            print("Training fold", i)
            meta_train_df = valid_df.iloc[train_idx]
            meta_valid_df = valid_df.iloc[valid_idx]
            pipeline = Pipeline([
                ('feature_generation', MetaFeatureGenerator(results))
            ])    
            X_meta_train = pipeline.fit_transform(meta_train_df)
            y_meta_train = meta_train_df.class_label
            X_meta_valid = pipeline.transform(meta_valid_df)
            y_meta_valid = meta_valid_df.class_label
            model = RandomForestClassifier()
            model.fit(X_meta_train, y_meta_train)
            pred = model.predict(X_meta_valid)
            result = scoring(meta_valid_df, pred)
            plot_result(result,f'META FOLD {i}')
            i+=1
        # Retrain Meta model with full valid df
        model = RandomForestClassifier()
        meta_pipeline = Pipeline([
                ('feature_generation', MetaFeatureGenerator(results))
            ])   
        X_meta = meta_pipeline.fit_transform(valid_df)
        model.fit(X_meta,valid_df.class_label)
        results['meta_model'] = {"model": model,'pipeline':meta_pipeline}
        with open("results.pkl", 'wb') as f:
            pickle.dump(results, f)

    elif args.mode == 'predict':
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)
        test_df = load_data('test')
        pipeline = results['meta_model']['pipeline']
        X_test = pipeline.transform(test_df)
        test_df['prediction'] = results['meta_model']['model'].predict(X_test)
        test_result = scoring(test_df,test_df.prediction)
        plot_result(test_result,'Test Performance')
        result_df = test_df[['query_id','prediction']]
        result_df.columns = ['Query ID','Prediction']
        result_df.to_csv("result.csv",sep='|')
