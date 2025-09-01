import networkx as nx
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
    print('#'*150)
    threshold_rating = df['review_scores_rating'].quantile(0.9)
    df_rating=df[df['review_scores_rating'] >= threshold_rating]
    df_rating=df_rating.drop(columns=["estimated_occupancy_l365d","review_scores_rating","estimated_revenue_l365d"])
    threshold_brating = df['review_scores_rating'].quantile(0.1)
    df_rating_bad=df[df['review_scores_rating'] >= threshold_brating]
    df_rating_bad = df_rating_bad.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    # good_bad_sets(df_rating,df_rating_bad)
    find_freq_itemsets(df_rating)
    # find_freq_itemsets(df_rating)
    # by estimated occupancy
    print("Finding frequent itemsets according to estimated occupancy...")
    print('#'*50)
    threshold_est_occ = df['estimated_occupancy_l365d'].quantile(0.9)
    df_est_occ = df[df['estimated_occupancy_l365d'] >= threshold_est_occ]
    df_est_occ=df_est_occ.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    threshold_est_occ = df['estimated_occupancy_l365d'].quantile(0.1)
    df_est_occb = df[df['estimated_occupancy_l365d'] <= threshold_est_occ]
    df_est_occb=df_est_occb.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    # good_bad_sets(df_est_occ,df_est_occb)
    # find_freq_itemsets(df_est_occ)
    #
    # # by estimated revenue
    # print("Finding frequent itemsets according to estimated revenue...")
    # print('#'*50)
    threshold_est_rev = df['estimated_revenue_l365d'].quantile(0.9)
    df_est_rev = df[df['estimated_revenue_l365d'] >= threshold_est_rev]
    df_est_rev=df_est_rev.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    threshold_est_rev = df['estimated_revenue_l365d'].quantile(0.1)
    df_est_revb = df[df['estimated_revenue_l365d'] <= threshold_est_rev]
    df_est_revb=df_est_revb.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    # good_bad_sets(df_est_rev,df_est_revb)
    # find_freq_itemsets(df_est_rev)





def normalize_db_freq(df):
    df = df.dropna()
    df = df.drop(columns=["id"])

    df = filter_columns_frequency(df)

    return df
def find_freq_itemsets(df,min_supp=0.2):
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

    print(str(len(df)) +" this is the lengfth of data frame")
    #find it
    frequent_itemsets = apriori(df, min_support=min_supp, use_colnames=True,low_memory=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets with min support {min_supp}")
    large_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 3)]

    find_top_associations(frequent_itemsets)
    calculate_sparsity(df)


    # Make sure all columns are fully visible
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    large_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 3)]

    # Get top 10 by support (or any metric you choose)
    top10_large = large_itemsets.sort_values(by='support', ascending=False).head(10)
    print("Frequent itemsets:")
    print(top10_large)
    # # Find all itemsets that contain a specific item
    # target_item = 'price_256+'  # Replace with your item name
    #
    # # Filter itemsets containing the target item
    # containing_itemsets = frequent_itemsets[
    #     frequent_itemsets['itemsets'].apply(lambda x: target_item in x)
    # ]
    #
    # print(f"Itemsets containing '{target_item}':")
    # print(containing_itemsets)
    #
    # # Calculate average support for itemsets containing this item
    # avg_support = containing_itemsets['support'].mean()
    # print(f"\nAverage support for itemsets containing '{target_item}': {avg_support:.2%}")


def find_top_associations(frequent_itemsets,num_rules=50):
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    meaningful_rules = rules[
        (rules['support'] >= 0.05) &
        (rules['confidence'] >= 0.7) &
        (rules['lift'] > 1)
        ].copy()
    meaningful_rules = meaningful_rules.sort_values(by='lift', ascending=False).head(num_rules)
    # meaningful_rules=representative_rules(meaningful_rules, threshold=0.5, metric="lift")
    # print_association_rules(meaningful_rules)
    # find_specific_associations(meaningful_rules,'superhost')
    plot_graph_rules(meaningful_rules)
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



def jaccard(a, b):
    return len(a & b) / len(a | b)

def representative_rules(rules, threshold=0.5, metric="lift"):
    """
    Deduplicate association rules:
    - groups rules by Jaccard similarity of (antecedent ∪ consequent)
    - keeps only the strongest rule (by chosen metric) from each group
    """
    rules = rules.copy()
    # Combine antecedent + consequent into one set
    rules["items"] = rules.apply(lambda r: set(r["antecedents"]) | set(r["consequents"]), axis=1)

    selected_items, selected_rules = [], []
    for idx, row in rules.sort_values(metric, ascending=False).iterrows():
        items = row["items"]
        # check if this rule is too similar to one we've already kept
        if any(jaccard(items, s) > threshold for s in selected_items):
            continue
        selected_items.append(items)
        selected_rules.append(idx)

    return rules.loc[selected_rules].drop(columns=["items"])

def filter_columns_frequency(df, min_freq=0.05, max_freq=0.8):
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


def good_bad_sets(df_top,df_bottom):
    # Mine top listings
    freq_top = apriori(df_top, min_support=0.1, use_colnames=True)
    # Mine bottom listings
    freq_bottom = apriori(df_bottom, min_support=0.1, use_colnames=True)

    # Merge on the same itemsets (as frozen sets)
    freq_top["itemset_str"] = freq_top["itemsets"].apply(frozenset)
    freq_bottom["itemset_str"] = freq_bottom["itemsets"].apply(frozenset)

    top_sets = set(freq_top["itemset_str"])
    # Get bottom itemsets as a set for fast lookup
    bottom_sets = set(freq_bottom["itemset_str"])

    # Keep only itemsets that appear in top but NOT in bottom
    # contrastive_sets = freq_top[~freq_top["itemset_str"].isin(bottom_sets)]
    contrastive_sets = freq_bottom[~freq_bottom["itemset_str"].isin(top_sets)]
    pd.set_option('display.max_colwidth', None)
    # Sort by support or lift (if you generate rules after)
    contrastive_sets = contrastive_sets.sort_values("support", ascending=False)
    contrastive_sets=contrastive_sets[contrastive_sets['itemsets'].apply(lambda x: len(x) > 3)]
    top10_large = contrastive_sets.sort_values(by='support', ascending=False).head(10)


    print("Frequent itemsets:")
    print(top10_large)

def plot_graph_rules(rules):
    G = nx.DiGraph()
    for _, row in rules.head(50).iterrows():
        for a in row['antecedents']:
            for c in row['consequents']:
                G.add_edge(a, c, weight=row['lift'])

    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u, v in edges]

    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='grey', width=weights)
    plt.show()
if __name__ == '__main__':
    # merge_dbs()
    df =pd.read_csv('freq_item_db.csv')
    find_freq_itemsets(df)


