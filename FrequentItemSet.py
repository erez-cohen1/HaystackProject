
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import clean_data

def find_freq_itemsets_by_diff_popularity(df):
    """
    for each take the top 10 percentile and find the frequent itemsets and association rules
    :param df:
    :return:
    """
    df=normalize_db_freq(df)

    # by rating review
    print("Finding frequent itemsets according to rating...")
    print('#'*50)
    threshold_rating = df['review_scores_rating'].quantile(0.9)
    df_rating=df[df['review_scores_rating'] >= threshold_rating]
    df_rating=df_rating.drop(columns=["estimated_occupancy_l365d","review_scores_rating","estimated_revenue_l365d"])
    find_freq_itemsets(df_rating)
    # by estimated occupancy
    print("Finding frequent itemsets according to estimated occupancy...")
    print('#'*50)
    threshold_est_occ = df['estimated_occupancy_l365d'].quantile(0.9)
    df_est_occ = df[df['estimated_occupancy_l365d'] >= threshold_est_occ]
    df_est_occ=df_est_occ.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    find_freq_itemsets(df_est_occ)

    # by estimated revenue
    print("Finding frequent itemsets according to estimated revenue...")
    print('#'*50)
    threshold_est_rev = df['estimated_revenue_l365d'].quantile(0.9)
    df_est_rev = df[df['estimated_revenue_l365d'] >= threshold_est_rev]
    df_est_rev=df_est_rev.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    find_freq_itemsets(df_est_rev)





def normalize_db_freq(df):
    df = df.dropna()
    df = df.drop(columns=["id"])

    df = filter_columns_frequency(df)

    return df
def find_freq_itemsets(df,min_supp=0.25):
    """
    min supp should be changed according to the dataframe
    :param df:
    :param min_supp:
    :return:
    """
    df = df.copy()
    # Identify non-binary columns

    # non_binary_cols = [col for col in df.columns
    #                    if not df[col].dropna().isin([0, 1, True, False]).all()]
    #
    # print("Columns with invalid values:", non_binary_cols)


    #find it
    frequent_itemsets = apriori(df, min_support=min_supp, use_colnames=True,low_memory=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets with min support {min_supp}")

    find_top_associations(frequent_itemsets)
    calculate_sparsity(df)


    # Make sure all columns are fully visible
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    large_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 6)]

    # Get top 10 by support (or any metric you choose)
    top10_large = large_itemsets.sort_values(by='support', ascending=False).head(30)
    print("Frequent itemsets:")
    print(top10_large)


def find_top_associations(frequent_itemsets,num_rules=10):
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    meaningful_rules = rules[
        (rules['support'] >= 0.05) &
        (rules['confidence'] >= 0.7) &
        (rules['lift'] > 1)
        ].copy()
    meaningful_rules = meaningful_rules.sort_values(by='lift', ascending=False).head(num_rules)
    # print_association_rules(meaningful_rules)
    find_specific_associations(meaningful_rules,'has_bedroom_core')

def find_specific_associations(rules,target):
    filtered_rules = rules[rules['antecedents'].apply(lambda x: target in x)]
    if len(filtered_rules) ==0:
        print("No associations found")
    print_association_rules(filtered_rules)
def print_association_rules(rules):
    print("Association rules:")
    for _, row in rules.iterrows():
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

    columns_to_keep = list(set(columns_to_keep) | {'review_scores_rating', 'estimated_occupancy_l365d',
                                                   'estimated_revenue_l365d'})

    print(f"Removing {len(columns_to_remove)} columns:")
    for col in columns_to_remove:
        freq = column_frequency[col]
        status = "too rare" if freq < min_freq else "too common"
        print(f"  {col}: {freq:.3f} ({status})")

    return df[columns_to_keep]



if __name__ == '__main__':
    # merge_dbs()
    df =pd.read_csv('freq_item_db.csv')
    find_freq_itemsets(df)


