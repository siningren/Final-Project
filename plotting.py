
import matplotlib.pyplot as plt
import seaborn as sns
 
# This function creates a histogram of the target variable (Adj Close)
def hist_plot(data, target):
    adj_close_data = data[target]
    plt.rcParams['axes.unicode_minus'] = False
    # Plot the histogram
    plt.figure(figsize=(12, 4), dpi=300)
    plt.hist(adj_close_data, bins=30, edgecolor='black', color='skyblue')
    # Add title and axis labels
    plt.title(f'Histogram of the Target Variable {target}')
    plt.xlabel(f'{target} Values')
    plt.ylabel('Frequency')
    # Display the figure
    plt.show()

# This function creates a time series plot of the target variable (Adj Close)
def time_series_plot(data, target):
    data[target].plot(figsize=(12, 4), color='blue', linewidth=1.5, marker='o')
    # Add title and axis labels
    plt.title(f'Time Trend of the Target Variable {target}')
    plt.xlabel('Time')
    plt.ylabel(f'{target} Values')
    # Set the rotation angle of the x-axis tick labels to make the date display clearer
    plt.xticks(rotation=45)
    # Add grid lines to make the chart more regular
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Display the figure
    plt.show()

# This function creates a correlation heatmap of the variables in the dataset
def corr_heatmap(corr):
    fig, axes = plt.subplots(figsize=(18, 6), dpi=300)  # This method creates a figure and a set of subplots
    heatmap = sns.heatmap(data=corr, annot=False, linewidths=.5, cmap="coolwarm", ax=axes)  # Figure out heatmap
    for text in heatmap.texts:
        text.set_fontsize(12)
    plt.show()  # Shows only plot and remove other informations

# This function creates a bar chart of the correlation coefficients between features and the target variable (Adj Close)
def sort_corr(corr):
    # Sorted correlation coefficients between features and Adj Close
    sort_corr = corr['Adj Close'].sort_values()
    plt.figure(figsize=(18, 6), dpi=300)
    # Plot the bar chart with variable names on the x-axis and correlation coefficients on the y-axis
    bars = plt.bar(sort_corr.index, sort_corr.values, color="skyblue")
    # Add title and axis labels
    plt.title("Sorting of Correlation Coefficients between Variables and Adj Close")
    plt.xlabel("Variables")
    plt.ylabel("Correlation Coefficients")
    # Set the rotation angle of the x-axis tick labels to avoid text overlap
    plt.xticks(rotation=90)
    # Add grid lines to make the chart more regular
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.show()
    return sort_corr
