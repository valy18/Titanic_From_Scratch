import pandas as pd

def cast_cols(df, col_to_cast, type='object'):
    for col in col_to_cast:
        df[col] = df[col].astype(type)
    return df


def fill_missing_values(df):
    for col in df.columns:
        if(df[col].dtype == float) or (df[col].dtype == int):
            df[col] = df[col].fillna(df[col].median())
        if(df[col].dtype == object):
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def parse_model(df, target_col, x_cols):
    if target_col not in df.columns:
        raise ValueError("Target colums should belong to df")
    y = df[target_col]
    X = df[x_cols]
    return X, y


def transform(df):
    # cast variable
    df = cast_cols(df, ['Survived', 'Pclass'])
    
    # missing values
    df = fill_missing_values(df)
    
    # add is child column
    df['isChild'] = df['Age'].apply(lambda x: 1 if x < 10  else 0)

    # add tittle colubn
    df['title'] = df['Name'].apply(lambda x: x.split(' ')[1].split('.')[0].strip())

    # adding surname
    df['surname'] = df['Name'].map(lambda x: '(' in x)

    # parse model
    return parse_model(df, 'Survived', ['SibSp', 'Parch', 'Fare', 'Sex', 'Pclass', 'isChild', 'title', 'surname'])
