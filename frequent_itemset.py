import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import clean_data
import seaborn as sns
from collections import Counter
from scipy.stats import chi2_contingency, fisher_exact
from itertools import combinations

def find_freq_itemsets_by_diff_popularity(df):
    """
    for each take the top 10 percentile and find the frequent itemsets and association rules
    :param df:
    :return:
    """
    # plt.close('all')
    df=normalize_db_freq(df)

    # # by rating review
    # print("Finding frequent itemsets according to rating...")
    # print('#'*150)
    # threshold_rating = df['review_scores_rating'].quantile(0.9)
    # df_rating=df[df['review_scores_rating'] >= threshold_rating]
    # df_rating=df_rating.drop(columns=["estimated_occupancy_l365d","review_scores_rating","estimated_revenue_l365d"])
    # threshold_brating = df['review_scores_rating'].quantile(0.1)
    # df_rating_bad=df[df['review_scores_rating'] <= threshold_brating]
    # df_rating_bad = df_rating_bad.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    # good_bad_sets(df_rating,df_rating_bad)
    # # find_freq_itemsets(df_rating)
    #
    # # by estimated occupancy
    # print("Finding frequent itemsets according to estimated occupancy...")
    # print('#'*50)
    # threshold_est_occ = df['estimated_occupancy_l365d'].quantile(0.9)
    # df_est_occ = df[df['estimated_occupancy_l365d'] >= threshold_est_occ]
    # df_est_occ=df_est_occ.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    # threshold_est_occ = df['estimated_occupancy_l365d'].quantile(0.1)
    # df_est_occb = df[df['estimated_occupancy_l365d'] <= threshold_est_occ]
    # df_est_occb=df_est_occb.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    # good_bad_sets(df_est_occ,df_est_occb)
    # # find_freq_itemsets(df_est_occ)
    # #
    # # # by estimated revenue
    # # print("Finding frequent itemsets according to estimated revenue...")
    # # print('#'*50)
    # threshold_est_rev = df['estimated_revenue_l365d'].quantile(0.9)
    # df_est_rev = df[df['estimated_revenue_l365d'] >= threshold_est_rev]
    # df_est_rev=df_est_rev.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    # threshold_est_rev = df['estimated_revenue_l365d'].quantile(0.1)
    # df_est_revb = df[df['estimated_revenue_l365d'] <= threshold_est_rev]
    # df_est_revb=df_est_revb.drop(columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    # good_bad_sets(df_est_rev,df_est_revb)
    # # find_freq_itemsets(df_est_rev)
    print(df["review_scores_rating"].quantile(0.8))
    print(df["review_scores_rating"].quantile(0.1))
    # df=df.drop(['superhost'], axis=1)
    df_top = df[
        (df["review_scores_rating"] >= df["review_scores_rating"].quantile(0.8)) &
        (df["estimated_occupancy_l365d"] >= df["estimated_occupancy_l365d"].quantile(0.8)) &
        (df["estimated_revenue_l365d"] >= df["estimated_revenue_l365d"].quantile(0.8))
        ]
    df_top = df_top.drop(
        columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])

    df_bottom = df[
        (df["review_scores_rating"] <= df["review_scores_rating"].quantile(0.1)) &
        (df["estimated_occupancy_l365d"] <= df["estimated_occupancy_l365d"].quantile(0.1)) &
        (df["estimated_revenue_l365d"] <= df["estimated_revenue_l365d"].quantile(0.1))
        ]
    df_bottom = df_bottom.drop(
        columns=["estimated_occupancy_l365d", "review_scores_rating", "estimated_revenue_l365d"])
    print(f" the len of df_top : {len(df_top)}")
    print(f" the len of df_bottom : {len(df_bottom)}")
    # good_bad_sets(df_top,df_bottom)

    # result = compare_itemsets(df_top, df_bottom, min_support=0.5, use_fpgrowth=True)
    # result = add_significance(result, len(df_top), len(df_bottom), use_fisher=True)
    # result.to_csv(f"frequent_itemsets_supportdiff_p_value.csv",index=False)
    # print(result[['itemset','support_top','support_bot','support_diff','p_value']].head(10))
    top_items = top_items_analysis(df_top, df_bottom, top_n=40)
    # top_item_list = top_items['item'].tolist()
    columns_list = df_bottom.columns.tolist()
    # df_pairs=analyze_pairs_added_value(df_top, df_bottom, columns_list)
    # df_pairs.to_csv('freq_desc_pair_item.csv')
    # pd.set_option('display.max_colwidth', None)
    # analyze_pairs_added_value(df_top,df_bottom,top_item_list)
    plot_top_vs_bottom_vertical(top_items)
    # df1=top_item_combinations_meaningful(df_top,df_bottom,3,top_item_list)
    # rank_by_added_value(df1)
    # print(top_item_pairs_analysis_meaningful(df_top,df_bottom,columns_list))
    # print(top_items)

def plot_pie_chart():
    df= pd.read_csv('freq_general_results_one_item.csv')
    # dictionary of your prefix-to-category mapping
    prefix_map = {
        "has": "Amenities",
        "img": "Image",
        "Desc": "Description",
    }

    # function to assign category
    def get_category(item):
        for prefix, category in prefix_map.items():
            if item.startswith(prefix + "_"):
                return category
        return "Other"  # fallback category

    # apply
    df["category"] = df["item"].apply(get_category)
    # --- take top 10 per category ---
    top10_per_cat = (
        df.assign(abs_diff=df["support_diff"].abs())  # add abs column
        .sort_values(["category", "abs_diff"], ascending=[True, False])
        .groupby("category")
        .head(12)  # keep top 10 per category
    )

    # --- compute category importance ---
    category_importance = (
        top10_per_cat.groupby("category")["abs_diff"].sum()
        .sort_values(ascending=False)
    )

    # normalize to %
    category_importance = category_importance / category_importance.sum() * 100
    print(category_importance)
    # find biggest category
    explode = [0.1 if i == category_importance.argmax() else 0
               for i in range(len(category_importance))]

    plt.figure(figsize=(6, 6))
    plt.pie(
        category_importance,
        labels=category_importance.index,
        autopct="%.1f%%",
        startangle=90,
        pctdistance=0.8,
        explode=explode
    )
    plt.title("Category Importance (Top 10 Items Each)")
    plt.show()
def analyze_pairs_added_value(
        df_top,
        df_bot,
        top_items=None,
        max_candidates=1000,
        top_n=150,
        min_joint_support=0.01,
        min_support_diff=0.02,
        min_added_value=0.02,
        require_conditional=True,
        conditional_threshold=0.03,
        apply_fdr=False
):
    """
    Analyze pairs and rank by added_value = support_diff(pair) - max(support_diff(item1), support_diff(item2)).

    Parameters
    ----------
    df_top, df_bot : pd.DataFrame
        One-hot encoded DataFrames with the same columns.
    top_items : list or None
        If provided, only consider pairs among these items. Otherwise auto-select
        up to `max_candidates` items ranked by absolute single-item support diff.
    max_candidates : int
        Max single items to consider when top_items is None.
    top_n : int
        Number of top pairs to return.
    min_joint_support : float
        Minimum joint support in at least one group to consider pair.
    min_support_diff : float
        Minimum support difference (top - bot) for pair to be considered.
    min_added_value : float
        Minimum added value required: support_diff(pair) - max(single_item_diffs) >= this.
    require_conditional : bool
        If True, require conditional gain in at least one direction:
        conditional_diff = P(B|A)_top - P(B|A)_bot >= conditional_threshold
        OR the same with A|B.
    conditional_threshold : float
        Threshold for conditional gain if require_conditional=True.
    apply_fdr : bool
        If True, apply Benjamini-Hochberg correction to p-values (requires statsmodels).

    Returns
    -------
    pd.DataFrame
        Columns: item1, item2, support_top_pair, support_bot_pair, support_diff_pair,
                 added_value, count_top_pair, count_bot_pair,
                 support_diff_item1, support_diff_item2,
                 cond_top_B_given_A, cond_bot_B_given_A, cond_diff_B_given_A,
                 cond_top_A_given_B, cond_bot_A_given_B, cond_diff_A_given_B,
                 p_value, p_value_fdr (if apply_fdr)
    """
    if set(df_top.columns) != set(df_bot.columns):
        raise ValueError("df_top and df_bot must have the same columns.")
    items_all = list(df_top.columns)
    n_top = len(df_top)
    n_bot = len(df_bot)

    # single-item supports and diffs
    s_top = df_top.mean()
    s_bot = df_bot.mean()
    s_diff = s_top - s_bot

    # choose candidate items
    if top_items is None:
        ranked = s_diff.abs().sort_values(ascending=False).index.tolist()
        top_items = ranked[:max_candidates]
    else:
        missing = set(top_items) - set(items_all)
        if missing:
            raise ValueError(f"Provided top_items not in columns: {missing}")

    results = []
    pvals = []

    for a, b in combinations(top_items, 2):
        # pair joint supports
        top_joint = (df_top[a] & df_top[b]).mean()
        bot_joint = (df_bot[a] & df_bot[b]).mean()
        support_diff_pair = top_joint - bot_joint

        # filter by joint support and pair support diff
        if (top_joint < min_joint_support) and (bot_joint < min_joint_support):
            continue
        if support_diff_pair < min_support_diff:
            continue

        # compute added_value vs best single item
        sdiff_a = s_diff[a]
        sdiff_b = s_diff[b]
        best_single = max(sdiff_a, sdiff_b)
        added_value = support_diff_pair - best_single
        if added_value < min_added_value:
            continue

        # conditional checks both ways
        # P(b|a) in top/bot
        denom_top_a = s_top[a]
        denom_bot_a = s_bot[a]
        if denom_top_a > 0:
            cond_top_b_given_a = top_joint / denom_top_a
        else:
            cond_top_b_given_a = np.nan
        if denom_bot_a > 0:
            cond_bot_b_given_a = bot_joint / denom_bot_a
        else:
            cond_bot_b_given_a = np.nan
        cond_diff_b_given_a = np.nan
        if not np.isnan(cond_top_b_given_a) and not np.isnan(cond_bot_b_given_a):
            cond_diff_b_given_a = cond_top_b_given_a - cond_bot_b_given_a

        # P(a|b)
        denom_top_b = s_top[b]
        denom_bot_b = s_bot[b]
        if denom_top_b > 0:
            cond_top_a_given_b = top_joint / denom_top_b
        else:
            cond_top_a_given_b = np.nan
        if denom_bot_b > 0:
            cond_bot_a_given_b = bot_joint / denom_bot_b
        else:
            cond_bot_a_given_b = np.nan
        cond_diff_a_given_b = np.nan
        if not np.isnan(cond_top_a_given_b) and not np.isnan(cond_bot_a_given_b):
            cond_diff_a_given_b = cond_top_a_given_b - cond_bot_a_given_b

        # require conditional in at least one direction if asked
        if require_conditional:
            cond_ok = False
            if (not np.isnan(cond_diff_b_given_a)) and (cond_diff_b_given_a >= conditional_threshold):
                cond_ok = True
            if (not np.isnan(cond_diff_a_given_b)) and (cond_diff_a_given_b >= conditional_threshold):
                cond_ok = True
            if not cond_ok:
                continue

        # counts and Fisher
        count_top = int((df_top[a] & df_top[b]).sum())
        count_bot = int((df_bot[a] & df_bot[b]).sum())
        try:
            _, p = fisher_exact([[count_top, count_bot], [n_top - count_top, n_bot - count_bot]])
        except Exception:
            p = np.nan

        results.append({
            'item1': a,
            'item2': b,
            'support_top_pair': top_joint,
            'support_bot_pair': bot_joint,
            'support_diff_pair': support_diff_pair,
            'added_value': added_value,
            'count_top_pair': count_top,
            'count_bot_pair': count_bot,
            'support_diff_item1': sdiff_a,
            'support_diff_item2': sdiff_b,
            'cond_top_B_given_A': cond_top_b_given_a,
            'cond_bot_B_given_A': cond_bot_b_given_a,
            'cond_diff_B_given_A': cond_diff_b_given_a,
            'cond_top_A_given_B': cond_top_a_given_b,
            'cond_bot_A_given_B': cond_bot_a_given_b,
            'cond_diff_A_given_B': cond_diff_a_given_b,
            'p_value': p
        })
        pvals.append(p)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res

    # optional FDR correction
    if apply_fdr:
        try:
            import statsmodels.stats.multitest as smm
            pvals_arr = df_res['p_value'].fillna(1.0).values
            rejected, pvals_corrected, _, _ = smm.multipletests(pvals_arr, alpha=0.05, method='fdr_bh')
            df_res['p_value_fdr'] = pvals_corrected
            df_res['significant_fdr'] = rejected
        except Exception:
            # if statsmodels not available, skip
            df_res['p_value_fdr'] = np.nan
            df_res['significant_fdr'] = False

    df_res = df_res.sort_values(by='added_value', ascending=False).reset_index(drop=True)

    return df_res.head(top_n)

def rank_by_added_value(df_res, top_n=30, min_count_top=5, min_added=0.02, require_significant=False, alpha=0.05):
    """
    Rank combos by added_value = support_diff - max_subset_diff and apply basic filters.
    """
    if df_res.empty:
        return df_res

    df = df_res.copy()
    df['added_value'] = df['support_diff'] - df['max_subset_diff']
    # relative added (optional)
    df['added_rel'] = df['added_value'] / df['max_subset_diff'].replace(0, np.nan)

    # basic filters: minimum added, and minimum count in top
    df = df[df['added_value'] >= min_added]
    df = df[df['count_top'] >= min_count_top]

    # optional: require statistical significance (after you do p-value correction upstream)
    if require_significant:
        df = df[df['p_value'] <= alpha]

    df = df.sort_values(by='added_value', ascending=False).reset_index(drop=True)
    df.to_csv(f"frequent_itemsets_tripels_biggest.csv", index=False)
    return df.head(top_n)

def top_item_combinations_meaningful(
    df_top,
    df_bot,
    k=3,
    top_items=None,
    top_n=50,
    min_joint_support=0.01,
    min_support_diff=0.05,
    conditional_threshold=0.05,
    require_conditional=True,
    max_candidates=1000
):
    """
    Find meaningful item combinations of size k (k=2 pairs, k=3 triples, etc.)
    that are more common in df_top than df_bot and add information beyond subsets.

    Parameters
    ----------
    df_top, df_bot : pd.DataFrame
        One-hot encoded dataframes (same columns).
    k : int
        Size of combinations to test (2 for pairs, 3 for triples).
    top_items : list or None
        List of items (columns) to consider. If None, will use the top `max_candidates`
        columns ranked by absolute single-item support difference.
    top_n : int
        Number of top combinations to return.
    min_joint_support : float
        Minimum joint support (fraction) required in at least one group to consider the combo.
    min_support_diff : float
        Minimum difference (support_top_combo - support_bot_combo) required.
    conditional_threshold : float
        Minimum conditional-support increase required:
          P(new_item | subset)_top - P(new_item | subset)_bot >= conditional_threshold
    require_conditional : bool
        If True, require the conditional check to pass for the combination.
    max_candidates : int
        If top_items is None, select this many top single items to form combinations
        (limits combo explosion).

    Returns
    -------
    pd.DataFrame
        For each combination returns:
         - items (tuple), support_top, support_bot, support_diff
         - counts in top/bot
         - max_single_diff, max_subset_diff (max across all subsets of size < k)
         - conditional_diff (computed w.r.t. the most-supported subset in top)
         - p_value (Fisher)
    """
    # basic validations
    if not set(df_top.columns) == set(df_bot.columns):
        raise ValueError("df_top and df_bot must have the same columns (items).")

    items_all = list(df_top.columns)
    n_top = len(df_top)
    n_bot = len(df_bot)

    # single-item supports and diffs
    s_top = df_top.mean()
    s_bot = df_bot.mean()
    s_diff = s_top - s_bot

    # choose candidate items
    if top_items is None:
        ranked = s_diff.abs().sort_values(ascending=False).index.tolist()
        top_items = ranked[:max_candidates]
    else:
        missing = set(top_items) - set(items_all)
        if missing:
            raise ValueError(f"Some items in top_items are not columns in df: {missing}")

    # precompute subset supports/diffs for subsets of size < k (to compare against)
    # We'll store supports for singles and pairs (if k>=3)
    subset_support_top = {}
    subset_support_bot = {}
    subset_support_diff = {}

    # singles
    for it in top_items:
        subset_support_top[frozenset([it])] = s_top[it]
        subset_support_bot[frozenset([it])] = s_bot[it]
        subset_support_diff[frozenset([it])] = s_diff[it]

    # pairs (if needed)
    if k >= 3:
        for a, b in combinations(top_items, 2):
            key = frozenset([a, b])
            top_joint = (df_top[a] & df_top[b]).mean()
            bot_joint = (df_bot[a] & df_bot[b]).mean()
            subset_support_top[key] = top_joint
            subset_support_bot[key] = bot_joint
            subset_support_diff[key] = top_joint - bot_joint

    results = []
    # iterate over combos of size k
    for combo in combinations(top_items, k):
        combo_key = frozenset(combo)

        # compute joint support in top and bot
        mask_top = np.ones(n_top, dtype=bool)
        mask_bot = np.ones(n_bot, dtype=bool)
        for it in combo:
            mask_top &= df_top[it].values.astype(bool)
            mask_bot &= df_bot[it].values.astype(bool)

        top_joint = mask_top.mean()
        bot_joint = mask_bot.mean()
        support_diff_combo = top_joint - bot_joint

        # skip if joint support too small in both groups
        if (top_joint < min_joint_support) and (bot_joint < min_joint_support):
            continue

        # skip if diff too small
        if support_diff_combo < min_support_diff:
            continue

        # compute max single diff among members
        singles = [frozenset([it]) for it in combo]
        max_single_diff = max(subset_support_diff[s] for s in singles)

        # compute max subset diff for subsets of size 1..k-1 (we already have singles and pairs)
        # generate all proper subsets and check precomputed diffs
        subset_diffs = []
        # for efficiency: only check subsets that we precomputed (singles and pairs)
        for r in range(1, k):
            for sub in combinations(combo, r):
                sub_key = frozenset(sub)
                if sub_key in subset_support_diff:
                    subset_diffs.append(subset_support_diff[sub_key])
        max_subset_diff = max(subset_diffs) if subset_diffs else 0.0

        # require combo diff > max subset diff (adds signal beyond any subset)
        if support_diff_combo <= max_subset_diff:
            continue

        # conditional check:
        # choose the subset of size k-1 with the highest support in top as conditioning set
        best_subset = None
        best_subset_support_top = -1.0
        for sub in combinations(combo, k - 1):
            sub_key = frozenset(sub)
            # compute sub top support (if precomputed use it; otherwise compute)
            if sub_key in subset_support_top:
                sup_top_sub = subset_support_top[sub_key]
                sup_bot_sub = subset_support_bot[sub_key]
            else:
                # compute on the fly
                mask_sub_top = np.ones(n_top, dtype=bool)
                mask_sub_bot = np.ones(n_bot, dtype=bool)
                for it in sub:
                    mask_sub_top &= df_top[it].values.astype(bool)
                    mask_sub_bot &= df_bot[it].values.astype(bool)
                sup_top_sub = mask_sub_top.mean()
                sup_bot_sub = mask_sub_bot.mean()
                subset_support_top[sub_key] = sup_top_sub
                subset_support_bot[sub_key] = sup_bot_sub
                subset_support_diff[sub_key] = sup_top_sub - sup_bot_sub

            if sup_top_sub > best_subset_support_top:
                best_subset_support_top = sup_top_sub
                best_subset = sub_key
                best_subset_support_bot = sup_bot_sub

        # compute conditional P(new_item | best_subset)
        # new item is the element in combo not in best_subset
        new_item = tuple(combo_key - best_subset)[0]
        denom_top = best_subset_support_top
        denom_bot = best_subset_support_bot

        if denom_top > 0:
            cond_top = top_joint / denom_top
        else:
            cond_top = np.nan

        if denom_bot > 0:
            cond_bot = bot_joint / denom_bot
        else:
            cond_bot = np.nan

        conditional_diff = np.nan
        if not np.isnan(cond_top) and not np.isnan(cond_bot):
            conditional_diff = cond_top - cond_bot

        if require_conditional:
            if np.isnan(conditional_diff) or conditional_diff < conditional_threshold:
                continue

        # counts for Fisher
        a = int(mask_top.sum())
        b = int(mask_bot.sum())
        c = n_top - a
        d = n_bot - b
        try:
            _, p = fisher_exact([[a, b], [c, d]])
        except Exception:
            p = np.nan

        results.append({
            'items': tuple(combo),
            'support_top': top_joint,
            'support_bot': bot_joint,
            'support_diff': support_diff_combo,
            'count_top': a,
            'count_bot': b,
            'max_single_diff': max_single_diff,
            'max_subset_diff': max_subset_diff,
            'best_subset': tuple(best_subset) if best_subset is not None else None,
            'conditional_diff': conditional_diff,
            'p_value': p
        })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res
    df_res = df_res.sort_values(by='support_diff', ascending=False).reset_index(drop=True)
    df_res.to_csv('frequent_itemsets_triples.csv', index=False)
    return df_res.head(top_n)
def add_significance(merged, total_top, total_bot, use_fisher=False):
    p_values = []
    for _, row in merged.iterrows():
        a = int(row['support_top'] * total_top)
        b = int(row['support_bot'] * total_bot)
        c = total_top - a
        d = total_bot - b
        table = [[a, b],
                 [c, d]]
        if use_fisher:
            _, p = fisher_exact(table)
        else:
            _, p, _, _ = chi2_contingency(table)
        p_values.append(p)
    merged['p_value'] = p_values
    return merged

def normalize_db_freq(df):
    df = df.dropna()
    df = df.drop(columns=["id"])

    # df = filter_columns_frequency(df)

    return df

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


def compare_itemsets(df_top, df_bot, min_support=0.5, use_fpgrowth=False):
    """
    Compare frequent itemsets between two dataframes.

    Parameters
    ----------
    df_top : pd.DataFrame
        One-hot encoded transactions (columns = items, rows = baskets).
    df_bot : pd.DataFrame
        Same format as df_top.
    min_support : float
        Minimum support threshold for frequent itemsets.
    use_fpgrowth : bool
        If True, use fpgrowth instead of apriori.

    Returns
    -------
    pd.DataFrame
        DataFrame with itemsets, support_top, support_bot, support_diff.
    """
    if use_fpgrowth:
        from mlxtend.frequent_patterns import fpgrowth
        freq_func = fpgrowth
    else:
        freq_func = apriori
    print("looking for frequent itemsets...")
    # Mine frequent itemsets for both groups
    top_sets = freq_func(df_top, min_support=min_support, use_colnames=True)
    bot_sets = freq_func(df_bot, min_support=min_support, use_colnames=True)

    print("normalzing frequent itemsets...")
    # Normalize itemsets to frozensets so we can join
    top_sets['itemset'] = top_sets['itemsets'].apply(frozenset)
    bot_sets['itemset'] = bot_sets['itemsets'].apply(frozenset)

    # Filter by minimum length
    top_sets = top_sets[top_sets['itemset'].apply(len) >= 1]
    bot_sets = bot_sets[bot_sets['itemset'].apply(len) >= 1]

    print("merging frequent itemsets...")
    merged = top_sets[['itemset', 'support']].merge(
        bot_sets[['itemset', 'support']],
        on='itemset',
        suffixes=('_top', '_bot')
    )

    print("calculating diffs...")
    merged['support_diff'] = merged['support_top'] - merged['support_bot']
    merged['abs_diff'] = merged['support_diff'].abs()

    # Sort by biggest absolute difference
    merged = merged.sort_values(by='abs_diff', ascending=False).reset_index(drop=True)

    merged.to_csv(f"frequent_itemsets_supportdiff.csv",index=False)

    return merged

def top_items_analysis(df_top, df_bot, top_n=20, plot=True):
    """
    Identify the most important items differentiating top vs bottom groups.

    Parameters
    ----------
    df_top : pd.DataFrame
        One-hot encoded transactions for the top group.
    df_bot : pd.DataFrame
        One-hot encoded transactions for the bottom group.
    top_n : int
        Number of top items to return/plot.
    plot : bool
        If True, plot the top items.

    Returns
    -------
    pd.DataFrame
        DataFrame with item, support_top, support_bot, support_diff, p_value.
    """
    total_top = len(df_top)
    total_bot = len(df_bot)

    items = df_top.columns
    support_top = df_top.mean()
    support_bot = df_bot.mean()
    support_diff = support_top - support_bot
    p_values = []

    # Calculate Fisher's Exact Test for each item
    for col in items:
        a = df_top[col].sum()
        b = df_bot[col].sum()
        c = total_top - a
        d = total_bot - b
        table = [[a, b], [c, d]]
        _, p = fisher_exact(table)
        p_values.append(p)

    result = pd.DataFrame({
        'item': items,
        'support_top': support_top.values,
        'support_bot': support_bot.values,
        'support_diff': support_diff.values,
        'p_value': p_values
    })

    # Sort by absolute support difference and optionally significance
    result['abs_diff'] = result['support_diff'].abs()
    result = result.sort_values(by='abs_diff', ascending=False).reset_index(drop=True)
    result.to_csv('freq_general_results_one_item.csv',index=False)
    return result.head(top_n)

def top_item_pairs_analysis_meaningful(
    df_top,
    df_bot,
    top_items=None,
    top_n=50,
    min_joint_support=0.01,
    min_support_diff=0.05,
    conditional_threshold=0.05,
    require_conditional=True,
    max_candidates=200
):
    """
    Find meaningful item pairs that are more common in df_top than df_bot and not driven
    purely by one dominant item.

    Parameters
    ----------
    df_top, df_bot : pd.DataFrame
        One-hot encoded dataframes (same columns).
    top_items : list or None
        List of items (columns) to consider. If None, will use the top `max_candidates`
        columns ranked by absolute single-item support difference.
    top_n : int
        Number of top pairs to return.
    min_joint_support : float
        Minimum joint support (fraction) required in at least one group to consider the pair.
    min_support_diff : float
        Minimum difference (support_top_pair - support_bot_pair) required.
    conditional_threshold : float
        Minimum conditional-support increase required:
        P(item2 | item1)_top - P(item2 | item1)_bot >= conditional_threshold
    require_conditional : bool
        If True, only keep pairs that pass the conditional check above.
    max_candidates : int
        If top_items is None, select this many top single items to form pairs (limits combo explosion).

    Returns
    -------
    pd.DataFrame
        Columns:
        ['item1','item2',
         'support_top_pair','support_bot_pair','support_diff_pair',
         'count_top_pair','count_bot_pair',
         'support_diff_item1','support_diff_item2',
         'cond_top_item2_given_item1','cond_bot_item2_given_item1','conditional_diff_2_given_1',
         'p_value']
    """
    # Basic checks
    if not set(df_top.columns) == set(df_bot.columns):
        raise ValueError("df_top and df_bot must have the same columns (items).")

    items = list(df_top.columns)
    n_top = len(df_top)
    n_bot = len(df_bot)

    # Single-item supports and diffs (per column)
    s_top = df_top.mean()
    s_bot = df_bot.mean()
    s_diff = s_top - s_bot

    # If top_items not provided, pick top single items by abs diff (limit by max_candidates)
    if top_items is None:
        ranked = s_diff.abs().sort_values(ascending=False).index.tolist()
        top_items = ranked[:max_candidates]
    else:
        # ensure provided items exist in df
        missing = set(top_items) - set(items)
        if missing:
            raise ValueError(f"Some items in top_items are not columns in df: {missing}")

    results = []

    for item1, item2 in combinations(top_items, 2):
        # joint support (fraction) in each group
        top_joint = (df_top[item1] & df_top[item2]).mean()
        bot_joint = (df_bot[item1] & df_bot[item2]).mean()
        support_diff_pair = top_joint - bot_joint

        # skip if joint support too small in both groups
        if (top_joint < min_joint_support) and (bot_joint < min_joint_support):
            continue

        # skip if pair support diff is too small
        if support_diff_pair < min_support_diff:
            continue

        # check pair provides more effect than either single item alone
        sdiff1 = s_diff[item1]
        sdiff2 = s_diff[item2]
        if support_diff_pair <= max(sdiff1, sdiff2):
            # Not meaningfully larger than the best single-item diff -> skip
            continue

        # conditional support P(item2 | item1)
        # Use means to compute safely; handle zero denom
        denom_top_item1 = s_top[item1]
        denom_bot_item1 = s_bot[item1]
        if denom_top_item1 > 0:
            cond_top_2_given_1 = top_joint / denom_top_item1
        else:
            cond_top_2_given_1 = np.nan

        if denom_bot_item1 > 0:
            cond_bot_2_given_1 = bot_joint / denom_bot_item1
        else:
            cond_bot_2_given_1 = np.nan

        conditional_diff_2_given_1 = np.nan
        if not np.isnan(cond_top_2_given_1) and not np.isnan(cond_bot_2_given_1):
            conditional_diff_2_given_1 = cond_top_2_given_1 - cond_bot_2_given_1

        # optionally require conditional gain
        if require_conditional:
            # if we cannot compute conditional diff (NaN) -> skip
            if np.isnan(conditional_diff_2_given_1) or conditional_diff_2_given_1 < conditional_threshold:
                continue

        # counts for Fisher
        a = int((df_top[item1] & df_top[item2]).sum())
        b = int((df_bot[item1] & df_bot[item2]).sum())
        c = n_top - a
        d = n_bot - b

        # Fisher's Exact Test (robust to small counts)
        # wrap in try to avoid edge-case errors
        try:
            _, p = fisher_exact([[a, b], [c, d]])
        except Exception:
            p = np.nan

        results.append({
            'item1': item1,
            'item2': item2,
            'support_top_pair': top_joint,
            'support_bot_pair': bot_joint,
            'support_diff_pair': support_diff_pair,
            'count_top_pair': a,
            'count_bot_pair': b,
            'support_diff_item1': sdiff1,
            'support_diff_item2': sdiff2,
            'cond_top_item2_given_item1': cond_top_2_given_1,
            'cond_bot_item2_given_item1': cond_bot_2_given_1,
            'conditional_diff_2_given_1': conditional_diff_2_given_1,
            'p_value': p
        })

    df_pairs = pd.DataFrame(results)
    if df_pairs.empty:
        return df_pairs  # empty DataFrame

    df_pairs = df_pairs.sort_values(by='support_diff_pair', ascending=False).reset_index(drop=True)
    df_pairs.to_csv('frequent_itemsets_pairs_general.csv', index=False)
    return df_pairs.head(top_n)

def good_bad_sets(df_top,df_bottom,min_supp=0.2):
    # Mine top listings
    freq_top = apriori(df_top, min_support=min_supp, use_colnames=True)
    # Mine bottom listings
    freq_bottom = apriori(df_bottom, min_support=min_supp, use_colnames=True)

    pd.set_option('display.max_colwidth', None)

    #find sets
    top_no_bottom_freq_sets(freq_top,freq_bottom)
    bottom_no_top_freq_sets(freq_top,freq_bottom)


    # Generate rules for good and bad separately
    rules_good = association_rules(freq_top, metric="lift", min_threshold=1.0)
    rules_bad = association_rules(freq_bottom, metric="lift", min_threshold=1.0)

    #get all items
    items_ante = rules_good['antecedents'].explode().unique()
    items_cons = rules_good['consequents'].explode().unique()

    # Combine and get unique
    items = list(set(items_ante) | set(items_cons))
    # print(f"amount of different items: {len(items)}")
    # Convert antecedents/consequents to frozensets for comparison
    rules_good["rule"] = rules_good.apply(lambda row: (frozenset(row["antecedents"]),
                                                       frozenset(row["consequents"])), axis=1)
    rules_bad["rule"] = rules_bad.apply(lambda row: (frozenset(row["antecedents"]),
                                                     frozenset(row["consequents"])), axis=1)

    # Keep only rules in good but not in bad
    # contrastive_rules = rules_good[~rules_good["rule"].isin(set(rules_bad["rule"]))]

    # Keep only rules in bad but not in good
    contrastive_rules = rules_bad[~rules_bad["rule"].isin(set(rules_good["rule"]))]

    # contrastive_rules=contrastive_rules.head(500)
    min_lift = 1.2
    min_confidence = 0.6
    min_support = 0.05

    contrastive_rules = contrastive_rules[
        (contrastive_rules['lift'] >= min_lift) &
        (contrastive_rules['confidence'] >= min_confidence) &
        (contrastive_rules['support'] >= min_support)
        ]
    find_significant_item(contrastive_rules,items)
    # contrastive_rules=contrastive_rules.head(50)
    contrastive_rules = contrastive_rules.sort_values("lift", ascending=False)
    # plot_graph_rules(contrastive_rules)
    # plot_heatmap(contrastive_rules)
    # plot_graph_rules_weighted_size(contrastive_rules)
    # plot_scatter(contrastive_rules)
    # sub_rules = contrastive_rules[
    #     contrastive_rules['antecedents'].apply(lambda x: 'price_0-100' in x) |
    #     contrastive_rules['consequents'].apply(lambda x: 'price_0-100' in x)
    #     ]
    # find_significant_item(contrastive_rules,items)
    # plot_graph_rules_weighted_size(sub_rules)


def plot_top_vs_bottom_vertical(result, top_n=15):
    """
    Vertical bar plot of items where support_top > support_bottom.
    Each item has two bars: top = support_top, bottom = support_bottom.

    Parameters
    ----------
    result : pd.DataFrame
        Must contain 'item', 'support_top', 'support_bot', 'support_diff'
    top_n : int
        Number of top items to display
    """
    # Filter for positive support differences
    pos_diff = result[result['support_diff'] > 0].head(top_n).copy()
    x_pos = np.arange(len(pos_diff))
    width = 0.35  # width of each bar

    plt.figure(figsize=(max(8, len(pos_diff)*0.4), 6))

    # Top support bars
    plt.bar(x_pos - width/2, pos_diff['support_top'], width=width, color='lightblue', label='Top')

    # Bottom support bars
    plt.bar(x_pos + width/2, pos_diff['support_bot'], width=width, color='orange', label='Bottom')

    # Annotate item names below each pair of bars
    plt.xticks(x_pos, pos_diff['item'], rotation=45, ha='right', fontsize=9)
    plt.ylabel('Support')
    plt.title('Item Support: Top vs Bottom (Top > Bottom)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def top_no_bottom_freq_sets(freq_top,freq_bottom,min_size=3):
    freq_top["itemset_str"] = freq_top["itemsets"].apply(frozenset)
    freq_bottom["itemset_str"] = freq_bottom["itemsets"].apply(frozenset)

    top_sets = set(freq_top["itemset_str"])
    # Get bottom itemsets as a set for fast lookup
    bottom_sets = set(freq_bottom["itemset_str"])

    # Keep only itemsets that appear in top but NOT in bottom

    # contrastive_sets = top_sets.union(bottom_sets)
    top_no_bottom_sets = freq_top[~freq_top["itemset_str"].isin(bottom_sets)]

    if len(top_no_bottom_sets)==0:
        print("no top no bottom sets found")
        return
    contrastive_sets = top_no_bottom_sets.sort_values("support", ascending=False)
    contrastive_sets=contrastive_sets[contrastive_sets['itemsets'].apply(lambda x: len(x) > min_size)]
    top10_large = contrastive_sets.sort_values(by='support', ascending=False).head(10)
    print("Frequent itemsets top no bottom sets:")
    print(top10_large)

def bottom_no_top_freq_sets(freq_top,freq_bottom,min_size=3):
    freq_top["itemset_str"] = freq_top["itemsets"].apply(frozenset)
    freq_bottom["itemset_str"] = freq_bottom["itemsets"].apply(frozenset)

    top_sets = set(freq_top["itemset_str"])
    # Get bottom itemsets as a set for fast lookup
    bottom_sets = set(freq_bottom["itemset_str"])

    bottom_no_top_sets = freq_bottom[~freq_bottom["itemset_str"].isin(top_sets)]
    if len(bottom_no_top_sets)==0:
        print("no bottom no top sets found")
        return
    contrastive_sets = bottom_no_top_sets.sort_values("support", ascending=False)
    contrastive_sets = contrastive_sets[contrastive_sets['itemsets'].apply(lambda x: len(x) > min_size)]
    top10_large = contrastive_sets.sort_values(by='support', ascending=False).head(10)
    print("Frequent itemsets bottom no top sets:")
    print(top10_large)

    #find rules
    rules = association_rules(freq_bottom, metric="lift", min_threshold=1.0)
    # contrastive_rules=contrastive_rules.head(500)
    min_lift = 1.2
    min_confidence = 0.6
    min_support = 0.05

    rules = rules[
        (rules['lift'] >= min_lift) &
        (rules['confidence'] >= min_confidence) &
        (rules['support'] >= min_support)
        ]
    rules=rules.sort_values("lift",ascending=False).head(50)
    # print_association_rules(rules)
    # items=['total_host_21+']
    # find_significant_item(rules,items)
    top_only,bottom_only,common=compare_rules(freq_top,freq_bottom)
    print("Rules only in top listings:")
    print(top_only.sort_values("lift", ascending=False).head(10))

    print("\nRules only in bottom listings:")
    print(bottom_only.sort_values("lift", ascending=False).head(10))

    print("\nRules common to both but stronger in top:")
    print(common.sort_values("lift_diff", ascending=False).head(10))
    plot_rule_comparison(common)


def compare_rules(freq_top, freq_bottom, min_lift=1.0):
    # 1. Generate rules for each group
    rules_top = association_rules(freq_top, metric="lift", min_threshold=min_lift)
    rules_bottom = association_rules(freq_bottom, metric="lift", min_threshold=min_lift)

    # 2. Build a stable rule identifier
    def make_rule_id(row):
        return tuple(sorted(list(row["antecedents"] | row["consequents"])))

    rules_top["rule"] = rules_top.apply(make_rule_id, axis=1)
    rules_bottom["rule"] = rules_bottom.apply(make_rule_id, axis=1)

    # 3. Rules unique to each
    top_only = rules_top[~rules_top["rule"].isin(rules_bottom["rule"])]
    bottom_only = rules_bottom[~rules_bottom["rule"].isin(rules_top["rule"])]

    # 4. Common rules → compare metrics
    common = rules_top.merge(
        rules_bottom[["rule", "lift", "confidence", "support"]],
        on="rule",
        suffixes=("_top", "_bottom")
    )
    common["lift_diff"] = common["lift_top"] - common["lift_bottom"]
    common["conf_diff"] = common["confidence_top"] - common["confidence_bottom"]

    return top_only, bottom_only, common


def plot_rule_comparison(common):
    plt.figure(figsize=(8, 8))

    # Scatter: lift in bottom vs lift in top
    plt.scatter(common["lift_bottom"], common["lift_top"], alpha=0.6)

    # Add diagonal line = equal lift in both
    max_val = max(common["lift_top"].max(), common["lift_bottom"].max())
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--")

    plt.xlabel("Lift (Bottom listings)")
    plt.ylabel("Lift (Top listings)")
    plt.title("Association Rules: Top vs Bottom comparison")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def find_significant_item(rules,items):
    i=0

    for item in items:
        sub_rules = rules[
            rules['antecedents'].apply(lambda x: item in x) |
            rules['consequents'].apply(lambda x: item in x)
            ]

        print(f"for item {item} the index is {i}")
        if len(sub_rules) > 0:
            plot_graph_rules_weighted_size(sub_rules)
            i+=1


def plot_graph_rules_weighted_size(rules, top_n=50):
    G = nx.DiGraph()
    rules = rules.sort_values("lift", ascending=False)
    # Add edges with weight = lift
    for _, row in rules.head(top_n).iterrows():
        for a in row['antecedents']:
            for c in row['consequents']:
                G.add_edge(a, c, weight=row['lift']*1)  # Edge thickness also reflects lift

    # Count how often each item appears in rules (antecedents + consequents)
    items = []
    for _, row in rules.head(top_n).iterrows():
        items.extend(row['antecedents'])
        items.extend(row['consequents'])
    freq = Counter(items)

    # Node size proportional to frequency
    node_sizes = [freq[node]*100 for node in G.nodes()]

    # Spring layout using edge weights
    pos = nx.spring_layout(G, k=1, iterations=100, weight='weight')

    # Edge widths proportional to lift
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Draw graph
    nx.draw(
        G, pos, with_labels=True, node_color='skyblue', edge_color='grey',
        width=weights, node_size=node_sizes
    )
    plt.title("Association Rules Graph\n(Node size = frequency, distance = lift)")
    plt.savefig("paris_graph.png")
    plt.show()
def plot_scatter(rules):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        rules['support'],
        rules['confidence'],
        c=rules['lift'],  # color by lift
        s=rules['lift'] * 20,  # size scaled by lift
        cmap='viridis',
        alpha=0.7,
        edgecolors="k"
    )

    plt.colorbar(scatter, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules (Support vs Confidence)")
    plt.show()
def plot_heatmap(rules):
    # Pivot into matrix form (rows=antecedents, cols=consequents, values=lift)
    heatmap_data = rules.pivot_table(
        index='antecedents',
        columns='consequents',
        values='lift',
        aggfunc='max',  # in case there are multiple rules
        fill_value=0
    )

    plt.figure(figsize=(10, 8))

    # Choose a diverging colormap with more contrast
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",  # try "viridis", "magma", or "YlGnBu" too
        annot=False,  # show numbers on top of colors
        fmt=".2f",  # number format
        linewidths=.5,  # grid lines between cells
        cbar_kws={'label': 'Lift'}  # label for colorbar
    )

    plt.title("Association Rules Heatmap (Lift)")
    plt.ylabel("Antecedent")
    plt.xlabel("Consequent")
    plt.savefig("Heatmap.png")
    plt.show()

if __name__ == '__main__':
    # merge_dbs()
    # plt.close('all')
    df =pd.read_csv('normalized_description.csv')
    df.drop('id', axis=1, inplace=True)
    calculate_sparsity(df)


