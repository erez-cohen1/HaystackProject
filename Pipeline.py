import pandas as pd

# import NormalizeDescriptions
# import NormalizeImageCol
import NormalizeRegularCols
import clean_data
# import normalize_amenities


def create_final_db(df):

    #create city merged database, saves a new csv and return the new df
    df_merged=create_merged_city_databases()

    #remove rows with missing values and convert bathroom text to bathrooms
    # also creates a new csv and returns the new df
    df_merg_clean=clean_data.clean_db(df_merged)

    #normalize regular cols (no amenities or description or images),keeps rating and estimated
    df_norm_reg = NormalizeRegularCols.normalize_regular_cols(df)

    #normalize amenities, returns df with normalized cols for amenities
    # # sends database after cleaning without normalizing
    # df_norm_amenities=normalize_amenities.normalize_amenities(df_merg_clean)
    #
    # #normalize description, return df with normalized cols for description, saves a new csv
    # df_norm_descriptions=NormalizeDescriptions.normalize_descriptions(df_merg_clean)
    #
    # #normalize image to cols, creates a non normalize csv in the middle and a normalized one, returns the norm
    # df_norm_images=NormalizeImageCol.create_normalize_image_db(df_merg_clean)

    #merge the 4 dbs into one, create the final dbs with all the normalized columns and
    # rating, estimated revenue, estimated occupancy
    merge_final_dbs()



def merge_final_dbs():
    """
    those are the defualt names i put for the df's so change them if needed.

    :return:
    """
    df1 = pd.read_csv('normalized_reg_cols.csv')
    df2 = pd.read_csv('image_analysis_norm.csv')
    df3 = pd.read_csv('cleaned_norm_amenities.csv')
    df4 = pd.read_csv('normalized_description.csv')

    #this drop shouldnt be here but quick fix
    df4 = df4.drop('none', axis=1)

    print(len(df1) + len(df2) + len(df3) + len(df4))

    merged1 = pd.merge(df1, df2, on="id", how="inner")
    merged2 = pd.merge(df3, merged1, on="id", how="inner")
    merged3 = pd.merge(df4, merged2, on="id", how="inner")

    print(len(merged3))
    merged3.to_csv('final_norm_database.csv', index=False)

def create_merged_city_databases():
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
    database_list= [pd.read_csv('listings_amsterdam.csv'),pd.read_csv('listings_clean_barcelona.csv'),pd.read_csv('listings_clean_paris.csv')]
    # Add source identifiers
    for i, df in enumerate(database_list):
        source_id = source_names[i]
        df['city'] = source_id
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

    return merged_df




if __name__ == '__main__':
    # df=create_merged_city_databases()
    # df=clean_data.clean_db(df,'cleaned_merged_db')
    # df=pd.read_csv('cleaned_merged_db.csv')
    # NormalizeRegularCols.normalize_regular_cols(df)
    merge_final_dbs()
