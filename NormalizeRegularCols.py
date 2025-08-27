
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import clean_data








def add_private_bathrooms_col(df):
    """
    when entire apartment always private, when shared always shared except if written explicitly that private
    :param df:
    :return:
    """
    df=df.copy()
    private_col = 'bathroom_private'
    df[private_col] = (
            df['bathrooms_text'].str.contains('private', case=False, na=False) |
            df['property_type'].str.contains('entire', case=False, na=False)
    ).astype(int)
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    return df

def convert_bool_to_bin(df, old_col, new_col):
    """
    convert bool to binary + fill 0 where null
    :param df:
    :param old_col:
    :param new_col:
    :return:
    """
    df[new_col] = df[old_col].str.lower().map({'f': 0, 't': 1})
    df[new_col] = df[new_col].fillna(0)
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

def normalize_regular_cols(df):
    """
    normalize df for finding freq itemsets
    :param df:
    :return:
    """

    df = df.copy()


    # TODO need to add - bucketize cities




    df = add_private_bathrooms_col(df)

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
                  'min_nights_1', 'min_nights_2', 'min_nights_3', 'min_nights_4+',
                  'review_scores_rating','estimated_occupancy_l365d','estimated_revenue_l365d'
                  ]
                 )
    df=df[result_cols]
    df.to_csv('normalized_reg_cols.csv',index=False)
    return df[result_cols]



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
    df = df[df['price_clean'] <= 1000]
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
    only for amsterdam
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
    only for amsterdam
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






if __name__ == '__main__':
    df=pd.read_csv('clean_merged_database.csv')
    df=normalize_regular_cols(df)
    df.to_csv('test_norm.csv', index=False)
