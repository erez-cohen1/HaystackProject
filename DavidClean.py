
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

def clean_bathrooms(df):
    """
    take from bathroom text num of bathroom and +
    when entire apartment always private, when shared always shared except if written explicitly that private
    :param df:
    :return: df with the coloumns: 'id', 'bathrooms', 'bathroom_private' (1 if yes)
    """
    # Make a copy to avoid warnings
    df = df.copy()

    # print_rows_per_val(df['estimated_occupancy_l365d'])
    # plot_dist(df['estimated_occupancy_l365d'])
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
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')

    # Return ALL columns including id and the new ones
    result_cols = ['id', 'bathrooms', 'bathroom_private']
    # result = df[result_cols]
    result=df
    return result

def convert_bool_to_bin(df,old_col,new_col):
    df[new_col] = df[old_col].str.lower().map({'f': 0, 't': 1})
    return df

def bucketize_column(df, col, bins, labels=None):
    """
    Create binary columns for ranges in a numeric column.
    make sure bins dont overlap.
    Parameters:
        df (pd.DataFrame): The DataFrame.
        col (str): Column to bucketize.
        bins (list of tuples): Each tuple is (min_val, max_val), inclusive. Use None for open-ended.
        labels (list of str, optional): Names for the new columns. If None, auto-generated.

    Returns:
        pd.DataFrame: DataFrame with new binary columns added.
    """
    if labels is None:
        labels = [f"{col}_{i}" for i in range(len(bins))]

    for (min_val, max_val), label in zip(bins, labels):
        if min_val is None:
            df[label] = (df[col] <= max_val).astype(int)
        elif max_val is None:
            df[label] = (df[col] >= min_val).astype(int)
        else:
            df[label] = ((df[col] >= min_val) & (df[col] <= max_val)).astype(int)
    return df

def normalize_cols(df):
    """
    normalize df for finding freq itemsets
    :param df:
    :return:
    """
    #should not be here, keeps the popular ones
    df = df[df['review_scores_rating'] >=4.9].copy()


    #convert the bools to binary
    convert_bool_to_bin(df,'host_is_superhost','superhost')
    convert_bool_to_bin(df,'host_identity_verified','host_verified')
    convert_bool_to_bin(df,'instant_bookable','instant_bookable_bin')
    convert_bool_to_bin(df,'host_has_profile_pic','host_profile_pic')

    #buckets for host_total_listings_count [1-2,3-10,11-200,200+]
    host_total_col='host_total_listings_count'
    host_total_bins = [(1, 2), (3, 10), (11, 200), (201,None)]
    host_total_labels = ['total_host_1-2', 'total_host_3-10', 'total_host_11-200', 'total_host_200+']
    df=bucketize_column(df,host_total_col,host_total_bins,host_total_labels)


    df=norm_neighborhoods(df)
    df=norm_property_type(df)
    # plot_dist(df)

    #buckets for accomodates [1,2,3-5,6+]
    accomodates_col='accommodates'
    accommodates_bins = [(1, 1), (2, 2), (3, 5), (6,None)]
    accommodates_labels=['accommodates_1','accommodates_2','accommodates_3-5','accommodates_6+']
    df=bucketize_column(df,accomodates_col,accommodates_bins,accommodates_labels)

    #buckets for bathrooms [0.5-1,1.5,2-2.5,3+]
    #there are listings with half bathroom, checked a few and it just looks like one.
    # counts = df['bathrooms'].value_counts().sort_index()
    # print(counts)
    bathrooms_col='bathrooms'
    bathrooms_bins = [(0.5, 1), (1.5, 1.5), (2, 2.5), (3,None)]
    bathrooms_labels=['bathrooms_0.5-1','bathrooms_1.5','bathrooms_2-2.5','bathrooms_3+']
    df=bucketize_column(df, bathrooms_col, bathrooms_bins, bathrooms_labels)


    #buckets for beds [0-1,2,3-4,5+]
    #sometimes 0 beds but when checked it appears to be atleast 1
    # counts = df['beds'].value_counts().sort_index()
    # print(counts)
    beds_col='beds'
    beds_bins = [(0, 1), (2, 2), (3, 4), (5,None)]
    beds_labels=['beds_0-1','beds_2','beds_3-4','beds_5+']
    df=bucketize_column(df, beds_col, beds_bins, beds_labels)

    #buckets for price ['$181.41', '$282.40', '$415.30', '$635.14']
    #found them with k-means clustring after cleaning a bit and removing all val above 1000
    df['price_clean'] = (
        df['price']
        .str.replace(r'[^\d.]', '', regex=True)  # Remove all non-numeric chars except .
        .astype(float)
    )
    df = df[df['price_clean'] <= 5000]
    price_col='price_clean'
    price_bins = [(0, 181.4), (181.5, 282.4), (282.5,415.3),(415.4,635),(635,None)]
    price_labels=['price_0-181','price_181-282','price_282-415','price_415-635','price_635+']
    df=bucketize_column(df,price_col,price_bins,price_labels)


    minimum_col='minimum_nights'
    df['min_nights_1'] = (df[minimum_col] == 1).astype(int)
    df['min_nights_2'] = (df[minimum_col] == 2).astype(int)
    df['min_nights_3'] = (df[minimum_col] == 3).astype(int)
    df['min_nights_4+'] = (df[minimum_col] >= 4).astype(int)

    neighborhoods = ['Centrum-West', 'Centrum-Oost', 'De Baarsjes - Oud-West',
                     'Buitenveldert - Zuidas', 'Bos en Lommer', 'IJburg - Zeeburgereiland',
                     'Zuid', 'Oud-Oost', 'De Pijp - Rivierenbuurt', 'Slotervaart', 'Noord-Oost',
                     'Westerpark', 'Oostelijk Havengebied - Indische Buurt', 'Watergraafsmeer',
                     'Oud-Noord', 'Bijlmer-Oost', 'Noord-West', 'Geuzenveld - Slotermeer',
                     'De Aker - Nieuw Sloten', 'Osdorp', 'Bijlmer-Centrum',
                     'Gaasperdam - Driemond']
    new_neigh=[f"{"neigh_"}{s}" for s in neighborhoods]

    #removed neighborhood and boats to reduce sparsity and because boats is rare
    result_cols=(['id','superhost','host_profile_pic','host_verified','instant_bookable_bin',
                 'total_host_1-2', 'total_host_3-10', 'total_host_11-200', 'total_host_200+',
                  'entire_house','shared_room_in_house','hotel/hostel_room',
                  'accommodates_1','accommodates_2','accommodates_3-5','accommodates_6+',
                  'bathrooms_0.5-1','bathrooms_1.5','bathrooms_2-2.5','bathrooms_3+',
                  'beds_0-1', 'beds_2', 'beds_3-4', 'beds_5+',
                  'price_0-181', 'price_181-282', 'price_282-415', 'price_415-635', 'price_635+',
                  'min_nights_1', 'min_nights_2', 'min_nights_3', 'min_nights_4+'
                  ]
                 )
    test_csv(df,result_cols)


def print_rows_per_val(col):
    counts = col.value_counts().sort_index()
    print(counts)
    print("\nMean:", col.mean())

def find_buckets_price(df):
    df['price_clean'] = (
        df['price']
        .str.replace(r'[^\d.]', '', regex=True)  # Remove all non-numeric chars except .
        .astype(float)
    )
    df = df[df['price_clean'] <= 5000]
    # Use CLEANED prices (reshape for K-Means)
    X = df['price_clean'].dropna().values.reshape(-1, 1)

    # Fewer clusters (3-5 max for your range)
    kmeans = KMeans(n_clusters=5).fit(X)  # Try 3 or 4
    centers = sorted(kmeans.cluster_centers_.flatten())

    # Get boundaries between clusters
    boundaries = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

    print("Realistic Bucket Boundaries:")
    print([f"${b:.2f}" for b in boundaries])
    # Define your bucket boundaries

    boundaries = [181.41, 282.40, 415.30, 635.14]
    labels = [
        "Under 181",
        "181 - 282",
        "282 - 415",
        "415 - 635",
        "Over 635"
    ]

    # Assign buckets
    df['price_bucket'] = pd.cut(
        df['price_clean'],
        bins=[0] + boundaries + [float('inf')],
        labels=labels
    )

    # Count rows per bucket
    bucket_counts = df['price_bucket'].value_counts().sort_index()
    print("Rows per bucket:")
    print(bucket_counts)

def norm_neighborhoods(df):
    """
    turn each unique neighbourhood into a binary column (22 cols). could create a sparsity problem that can be imporved
    by grouping them to areas i.e. west,south,center...
    :param df:
    :return:
    """

    # List of all 22 neighborhoods

    neighborhoods = ['Centrum-West', 'Centrum-Oost', 'De Baarsjes - Oud-West',
                     'Buitenveldert - Zuidas', 'Bos en Lommer', 'IJburg - Zeeburgereiland',
                     'Zuid', 'Oud-Oost', 'De Pijp - Rivierenbuurt', 'Slotervaart', 'Noord-Oost',
                     'Westerpark', 'Oostelijk Havengebied - Indische Buurt', 'Watergraafsmeer',
                     'Oud-Noord', 'Bijlmer-Oost', 'Noord-West', 'Geuzenveld - Slotermeer',
                     'De Aker - Nieuw Sloten', 'Osdorp', 'Bijlmer-Centrum',
                     'Gaasperdam - Driemond']
    # Create binary columns
    for neigh in neighborhoods:
        df[f'neigh_{neigh}'] = (df['neighbourhood_cleansed'] == neigh).astype(int)
    return df

def norm_property_type(df):
    """
    divides the property type into 4 categories each has its own binary column
    :param df:
    :return:
    """
    grouping_rules = {
        # Entire House (standalone properties)
        r'Entire home|Entire house|Entire villa|Entire cabin|Entire cottage|Entire townhouse|Entire loft|Entire rental unit|'
        r'Entire guest suite|Entire condo|Entire guesthouse|Tiny home|Entire place|'
        r'Entire serviced apartment|Entire vacation home|Casa particular': 'Entire house',

        # Shared Rooms (in houses)
        r'Shared room in home|Shared room in townhouse|Shared room in houseboat|'
        r'Private room in rental unit|Private room in condo|Private room in home|Private room in loft|'
        r'Private room in guest suite|Private room in villa|Private room in townhouse|Private room in farm stay|'
        r'Private room|Private room in windmill|Private room in cottage|Private room in casa particular|'
        r'Private room in serviced apartment|Private room in cabin|Private room in barn|'
        r'Private room in nature lodge|Private room in earthen home|Private room in vacation home|'
        r'Private room in hut': 'Shared room in house',

        # Hotel/Hostel Rooms
        r'Room in hotel|Room in hostel|Room in boutique hotel|Room in serviced apartment|'
        r'Private room in bed and breakfast|Room in bed and breakfast|Private room in guesthouse|'
        r'Room in aparthotel|Private room in tiny home|Shared room in hotel': 'Hotel/hostel room',

        # Boats private + shared
        r'Boat|Houseboat|Private room in boat|Shared room in boat|Private room in houseboat': 'Boats'
    }
    # Create the grouped column
    df['property_group'] = 'Other'  # Default for unclassified types

    for pattern, group in grouping_rules.items():
        mask = df['property_type'].str.contains(pattern, case=False, regex=True)
        df.loc[mask, 'property_group'] = group

    df = df[df['property_group'] != 'Other'].copy()
    groups = ['Entire house', 'Shared room in house', 'Hotel/hostel room', 'Boats']

    for group in groups:
        df[f'{group.lower().replace(" ", "_")}'] = (df['property_group'] == group).astype(int)

    df = df.drop(columns=['property_group'])
    return df

def plot_dist(col):
    # plt.figure(figsize=(10, 6))
    # plt.hist( df['host_total_listings_count'], bins=1000, color='skyblue', edgecolor='black')
    # plt.title('Distribution of Percentage Values')
    # plt.xlabel('Percentage (%)')
    # plt.ylabel('Frequency')
    # plt.grid(True, alpha=0.3)
    # plt.show()
    plt.figure(figsize=(10, 6))
    new_col = col.astype(str)
    plt.hist(new_col, bins=40)
    plt.title('Distribution')
    plt.xlabel('Values')
    plt.ylabel('Frequency (log scale)')
    plt.grid(True, alpha=0.3)
    # plt.savefig('host_total_distlog.jpg')  # Saves in current directory

    plt.show()


# def normalize_host_response_rate(df):
#     """
#     we didnt use this because the median is 100%
#     if above 90 percent then turn to 1 otherwise 0
#     :param df:
#     :return:
#     """
#     df['percentage_clean'] = df['host_response_rate'].str.replace('%', '').astype(float)
#     mean_val = df['percentage_clean'].mean()
#     median_val = df['percentage_clean'].median()
#
#     print("Mean:", mean_val)
#     print("Median:", median_val)


def test_csv(df,result_cols):
    """
    for testing, saves csv called test.csv with specific columns
    :param df:
    :param result_cols:
    :return:
    """
    result = df[result_cols]
    result.to_csv('test.csv',index=False)

def find_freq_itemsets(df2):

    df = pd.read_csv("test.csv")
    df = df.dropna()
    df = df.drop(columns=["id"])
    # Find frequent itemsets with minimum support threshold
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
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


def merge_databases():
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

if __name__ == '__main__':
    # df =pd.read_csv('listings_clean.csv')
    # df=clean_bathrooms(df)
    # df=normalize_cols(df)
    # find_freq_itemsets(df)
    _,report=merge_databases()
    print(report)