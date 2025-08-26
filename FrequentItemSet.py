
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import clean_data




def find_freq_itemsets(df):

    # df = pd.read_csv("test.csv")
    df = df.dropna()
    df = df.drop(columns=["id"])

    df = filter_columns_frequency(df)
    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True,low_memory=True)


    print(f"Found {len(frequent_itemsets)} frequent itemsets")

    find_top_associations(frequent_itemsets)
    calculate_sparsity(df)

    # with pd.option_context('display.max_colwidth', None):
    #     print(frequent_itemsets.tail())
    # Make sure all columns are fully visible
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    large_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 6)]

    # Get top 10 by support (or any metric you choose)
    top10_large = large_itemsets.sort_values(by='support', ascending=False).head(10)

    print(top10_large)

    # # prints the items by how common they are - useful for seeing if one of the cols is meaningless or wrong
    # # print(frequent_itemsets)
    # # Count of 1s per column
    # feature_counts = df.sum()
    #
    # # Fraction of rows with the feature (support)
    # feature_support = feature_counts / len(df)
    #
    # # Sort ascending to see least common features
    # feature_support = feature_support.sort_values()
    # print(feature_support)

def find_top_associations(frequent_itemsets,num_rules=10):
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    meaningful_rules = rules[
        (rules['support'] >= 0.05) &
        (rules['confidence'] >= 0.7) &
        (rules['lift'] > 1)
        ].copy()
    meaningful_rules = meaningful_rules.sort_values(by='lift', ascending=False).head(num_rules)
    # Make sure all columns are fully visible
    # Print each rule in A -> B format with metrics
    for _, row in meaningful_rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        print(f"({antecedent}) → ({consequent})| "
              f"support={row['support']:.3f}, "
              f"confidence={row['confidence']:.3f}, "
              f"lift={row['lift']:.3f}, "
              f"certainty={row.get('certainty', 0):.3f}, "
              f"kulczynski={row.get('kulczynski', 0):.3f}")

def calculate_sparsity(df):
    """
    from GPT:
    Low sparsity (<50%) → most features appear frequently; mining is easy, but rules might be trivial.

    Moderate sparsity (50–85%) → common in real datasets; enough variability to find meaningful rules.

    High sparsity (>90%) → many features are rare; support for combinations will be low, making frequent
    itemsets harder to find.
    :param df:
    :return:
    """
    # Total number of cells
    total_cells = df.shape[0] * df.shape[1]

    # Total number of 1s (non-zero values)
    total_ones = df.sum().sum()

    # Sparsity = fraction of zeros
    sparsity = 1 - (total_ones / total_cells)
    print(f"Sparsity: {sparsity:.2%}")


def create_merged_databases():
    """
    Merges multiple databases with the same structure while:
    1. Ensuring unique IDs across all databases
    2. Adding a source identifier column
    3. Reporting any lost rows

    Parameters:
        database_list (list of DataFrames): List of databases to merge
        source_names (list of str): Optional names for each source database

    Returns:
        merged_df (DataFrame): Combined database
        report (dict): Merge statistics
    """
    # Initialize report
    report = {
        'total_rows_before': 0,
        'total_rows_after': 0,
        'rows_lost': 0,
        'duplicate_ids': 0
    }
    source_names = ['Amsterdam','Barcelona','Paris']
    database_list= [pd.read_csv('listings_ams.csv'),pd.read_csv('listings_clean_barcelona.csv'),pd.read_csv('listings_clean_paris.csv')]
    # Add source identifiers
    for i, df in enumerate(database_list):
        source_id = source_names[i]
        df['data_source'] = source_id
        report['total_rows_before'] += len(df)

    # Concatenate all databases
    merged_df = pd.concat(database_list, ignore_index=False)

    # Check for duplicate IDs
    duplicate_mask = merged_df['id'].duplicated(keep=False)
    report['duplicate_ids'] = duplicate_mask.sum()

    # Remove duplicates (keep first occurrence)
    merged_df = merged_df.drop_duplicates(subset=['id'], keep='first')

    # Generate report

    merged_df = merged_df.dropna(subset=['review_scores_rating'])

    report['total_rows_after'] = len(merged_df)
    report['rows_lost'] = report['total_rows_before'] - report['total_rows_after']
    merged_df.to_csv('merged_database.csv', index=False)

    return merged_df, report


def filter_columns_frequency(df, min_freq=0.05, max_freq=0.9):
    """
    Remove columns that are too common or too rare
    min_freq: minimum proportion of 1s to keep column
    max_freq: maximum proportion of 1s to keep column
    """
    column_frequency = df.mean()

    # Columns to keep
    keep_mask = (column_frequency >= min_freq) & (column_frequency <= max_freq)
    columns_to_keep = column_frequency[keep_mask].index
    columns_to_remove = column_frequency[~keep_mask].index

    print(f"Removing {len(columns_to_remove)} columns:")
    for col in columns_to_remove:
        freq = column_frequency[col]
        status = "too rare" if freq < min_freq else "too common"
        print(f"  {col}: {freq:.3f} ({status})")

    return df[columns_to_keep]

def merge_dbs():

    df1=pd.read_csv('normalized_freq_db.csv')
    df2=pd.read_csv('image_analysis_norm.csv')
    df3=pd.read_csv('normalized_amenities.csv')
    df4=pd.read_csv('tagged_listings_one_hot.csv')
    df4= df4.drop('none', axis=1)
    print(len(df1)+len(df2)+len(df3)+len(df4))

    merged1 = pd.merge(df1, df2, on="id", how="inner")
    merged2 = pd.merge(df3, merged1, on="id", how="inner")
    merged3 = pd.merge(df4, merged2, on="id", how="inner")

    print(len(merged3))
    merged3.to_csv('freq_item_db.csv', index=False)

if __name__ == '__main__':
    # merge_dbs()
    df =pd.read_csv('freq_item_db.csv')
    find_freq_itemsets(df)

    # merge_dbs()
    # df =pd.read_csv('tagged_listings_one_hot.csv')
    # print(df['none'].sum())
    # df=pd.read_csv('clean_merged_database.csv')
    # column_data = df['review_scores_rating']
    #
    # # Calculate the 10th and 90th percentiles
    # low_10_percentile = column_data.quantile(0.10)
    # high_10_percentile = column_data.quantile(0.85)
    #
    # print(f"10th percentile (lowest 10% cutoff): {low_10_percentile}")
    # print(f"90th percentile (highest 10% cutoff): {high_10_percentile}")

    # merge_dbs()
    # create_merged_databases()
    # clean_data.clean_db(pd.read_csv('merged_database.csv'), 'clean_merged_database')
    # df=pd.read_csv('clean_merged_database.csv')
    # df=normalize_cols(df)
    # df.to_csv('normalized_freq_db.csv',index=False)
    # find_freq_itemsets(df)

