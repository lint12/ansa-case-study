import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_histogram(target_feature_data, non_target_feature_data, target_bin_size, non_target_bin_size, feature):
    plt.figure(figsize=(8, 4))
    # density = true, normalizes the histogram around to area = 1 (so we can compare shape and not absolute counts)
    plt.hist(target_feature_data, bins=target_bin_size, alpha=0.8, label='Target', density=True)
    plt.hist(non_target_feature_data, bins=non_target_bin_size, alpha=0.6, label='Non-Target', density=True)
    plt.xlabel(f"{feature}")
    plt.ylabel('Density')
    plt.title(f"Distribution of {feature}: Target vs Non-Target")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_kde(data, feature, label='label'):
    plt.figure(figsize=(4,3))
     # because we have a huge class imbalance,
     # common_norm=False will not normalize the KDEs to the same area
     # to show us the shape of the distributions and not the absolute counts
    sns.kdeplot(data=data, x=feature, hue=label, common_norm=False)
    plt.title(f'Distribution of {feature} by Target vs Non-target')
    plt.grid(True)
    plt.show()

def log_transform_deltas(df, delta_cols):
    """
    Safely log-transform delta features that may have negative values by shifting them.
    """
    # Find the overall minimum value among all delta columns (consistent scale)
    min_val = df[delta_cols].min().min()

    # Determine the shift needed
    shift = 0
    if min_val <= -1:
        shift = abs(min_val) + 1.01  # Make sure everything > 0 for log1p

    # Apply the shift and log1p
    for col in delta_cols:
        df[f'log_{col}'] = np.log1p(df[col] + shift)

def plot_boxplot(
    data, label_col, feature_col, hue_col=None, log_scale=False, figsize=(8, 4), rotate_xticks=False
):
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=label_col, y=feature_col, hue=hue_col)

    if log_scale:
        plt.yscale('log')

    plt.xlabel(label_col)
    plt.ylabel(feature_col)
    
    if log_scale:
        plt.title(f"Boxplot of {feature_col} by {label_col} (Log Scale)")
    else:
        plt.title(f"Boxplot of {feature_col} by {label_col}")
    
    if rotate_xticks:
        plt.xticks(rotation=45)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_kde_and_boxplot(data, feature, label_col, log_scale=False):
    """
    Plot KDE and Boxplot side-by-side for a given numeric feature split by label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # KDE Plot
    sns.kdeplot(
        data=data, x=feature, hue=label_col,
        common_norm=False, fill=True, alpha=0.4, ax=axes[0]
    )
    axes[0].set_title(f'{feature} - KDE Plot {log_scale and "(Log Scale)"}')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    if log_scale:
        axes[0].set_xscale('log')
    
    # Box Plot
    sns.boxplot(x=label_col, y=feature, data=data, ax=axes[1])
    axes[1].set_title(f'{feature} - Box Plot {log_scale and "(Log Scale)"}')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    if log_scale:
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_missing_comparison(missing_comparison, top_n=50):
    # Only show top N features with biggest missing % difference
    top_features = missing_comparison.head(top_n)

    # Plot
    x = np.arange(len(top_features))  # the label locations
    width = 0.45  # width of the bars

    fig, ax = plt.subplots(figsize=(36, 14))
    ax.bar(x - width/2, top_features['missing_pct_target'], width, label='Target Companies')
    ax.bar(x + width/2, top_features['missing_pct_non_target'], width, label='Non-Target Companies')

    ax.set_ylabel('Missing %')
    ax.set_title('Missing Percentage Comparison by Feature')
    ax.set_xticks(x)
    ax.set_xticklabels(top_features.index, rotation=90)
    ax.legend()

    for i, diff in enumerate(top_features['missing_pct_diff']):
        ax.text(i, max(top_features['missing_pct_target'].iloc[i], top_features['missing_pct_non_target'].iloc[i]) + 1,
                f"Δ{diff:.1f}%", ha='center', va='bottom', fontsize=8, color='red')

    plt.show()

def plot_growth(metric, df, num_months=17, time_direction='backward'):
    month_cols = []
    
    if 'growth' in metric and 'trailing' not in metric:
        month_cols = [f'{metric}_{m}m' for m in [1, 3, 6, 12]]
    else:
        month_cols = [f'{metric}_{m}m' for m in range(num_months + 1)]

    # Plot setup
    plt.figure(figsize=(8, 4))
    for i, row in df.iterrows():
        y = row[month_cols].apply(lambda x: np.nan if pd.isna(x) else x).values
        if 'growth' in metric and 'trailing' not in metric:
            plt.plot([1, 3, 6, 12], y, alpha=0.4)
        else:
            plt.plot(range(num_months + 1), y, alpha=0.4)  # label only a few for clarity

    plt.xlabel('Months (0m = reference point)')
    plt.ylabel(f"{metric}")
    if time_direction == 'backward':
        plt.title(f'{metric} up to {num_months} Months Ago (n={len(df)})')
    else:
        plt.title(f'{metric} {num_months} Months in (n={len(df)})')
    plt.grid(True)
    plt.tight_layout()

def plot_categorical_feature(data, feature, label_col, normalize=True, fig_size=(12,6)):
    """
    Plot bar chart for categorical feature distribution by label.
    """
    # Prepare data
    counts = pd.crosstab(data[feature], data[label_col])
    
    if normalize:
        counts = counts.div(counts.sum(axis=0), axis=1)  # column-wise normalization
    
    counts = counts.reset_index().melt(id_vars=feature, var_name='Label', value_name='Proportion' if normalize else 'Count')

    # Plot
    plt.figure(figsize=fig_size)
    sns.barplot(x=feature, y='Proportion' if normalize else 'Count', hue='Label', data=counts)
    plt.title(f'Distribution of {feature} by {label_col}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
