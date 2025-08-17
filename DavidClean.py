
import pandas as pd





def clean_bathrooms(df):
    """
    take from bathroom text num of bathroom and +
    when entire aparment always private, when shared always shared except if written explicitly that private
    :param df:
    :return: df with the coloumns: 'id', 'bathrooms', 'bathroom_private' (1 if yes)
    """
    # Make a copy to avoid warnings
    df = df.copy()

    # text to num
    string_col = 'bathrooms_text'  # column with strings containing numbers
    target_col = 'bathrooms'  # column that might be empty

    df['extracted_number'] = df[string_col].str.extract(r'(\d+)')  # Note the r prefix

    # Fill empty target_column values
    df[target_col] = df[target_col].fillna(df['extracted_number'])

    # new column indicating private bathroom or not
    private_col = 'bathroom_private'
    df[private_col] = (
            df['bathrooms_text'].str.contains('private', case=False, na=False) |
            df['property_type'].str.contains('entire', case=False, na=False)
    ).astype(int)

    # Return ALL columns including id and the new ones
    result_cols = ['id', 'bathrooms', 'bathroom_private']
    result = df[result_cols]
    return result


if __name__ == '__main__':
    df =pd.read_csv('listings_ams.csv')
    df=clean_bathrooms(df)
    df.to_csv('bathrooms_cleaned.csv',index=False)