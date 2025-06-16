import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def load_data(file_path):
    # I made it load the CSV file as a pandas DataFrame and return it.
    return pd.read_csv(file_path)

# Created a function to calculate basic descriptive statistics (mean, median, variance, standard deviation, standard error).
def calculate_basic_statistics(data):
    # I made it calculate the mean, median, variance, standard deviation, and standard error of the data.
    mean = data.mean()  # I made it calculate the mean value.
    median = data.median()  # I made it calculate the median value.
    variance = data.var()  # I made it calculate the variance.
    std_dev = data.std()  # I made it calculate the standard deviation.
    std_error = std_dev / np.sqrt(len(data))  # I made it calculate the standard error.

    # I made it print the calculated statistics.
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median}")
    print(f"Variance: {variance:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Standard Error: {std_error:.2f}")

# Created a function to visualize the data using a histogram and boxplot.
def visualize_data(data):
    # I made it create a histogram and boxplot to visualize the distribution of the data.
    plt.figure(figsize=(8, 6))  # I made it set the size for the histogram.
    plt.hist(data, bins=10, color='lightgreen', edgecolor='black')  # I created the histogram.
    plt.title("Event Participation Histogram")  # I added the title.
    plt.xlabel("Participation Count")  # I created the x-axis.
    plt.ylabel("Number of People")  # I created the y-axis.
    plt.grid(True, linestyle='--', alpha=0.7)  # I added gridlines.
    plt.tight_layout()  # I optimized the layout.
    plt.show()  # I made it show the plot.

    # I created the boxplot.
    plt.figure(figsize=(6, 5))  # I made it set the size for the boxplot.
    plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral'))  # I created the boxplot.
    plt.title("Event Participation Box Plot")  # I added the title.
    plt.xlabel("Participation Count")  # I added the x-axis label.
    plt.grid(True, linestyle=':', alpha=0.7)  # I added gridlines.
    plt.tight_layout()  # I optimized the layout.
    plt.show()  # I made it show the plot.

# Created a function using the IQR (Interquartile Range) method to detect outliers in the data.
def detect_outliers(data):
    # I made it use the IQR method to detect outliers.
    print("\n--- Outlier Detection ---")
    Q1 = data.quantile(0.25)  # I wrote the first quartile (25th percentile) value.
    Q3 = data.quantile(0.75)  # I wrote the third quartile (75th percentile) value.
    IQR = Q3 - Q1  # I calculated the IQR (Q3 - Q1).
    lower_bound = Q1 - 1.5 * IQR  # I defined the lower bound (1.5 times the IQR below Q1).
    upper_bound = Q3 + 1.5 * IQR  # I defined the upper bound (1.5 times the IQR above Q3).
    outliers = data[(data < lower_bound) | (data > upper_bound)]  # I selected the outliers.

    # I made it print the number of outliers and their values if any.
    print(f"Number of Outliers: {len(outliers)}")
    if not outliers.empty:
        print("Detected Outliers:")
        print(outliers.values)  # I printed the outlier values.
    else:
        print("No outliers found.")  # If no outliers were found, I printed a message.

# Created a function to calculate the confidence interval for the mean of the data.
def calculate_confidence_interval(data, confidence_level=0.95):
    # I made it calculate the confidence interval for the mean of the data with the specified confidence level.
    print(f"\n--- {confidence_level * 100}% Confidence Interval Calculation ---")

    mean = data.mean()  # I made it calculate the mean of the data.
    std_error = data.std() / np.sqrt(len(data))  # I made it calculate the standard error.

    # I made it calculate the confidence interval using the normal distribution.
    lower, upper = stats.norm.interval(confidence_level, loc=mean, scale=std_error)  # I made it calculate the confidence interval.
    print(f"Confidence Interval: ({lower:.2f}, {upper:.2f})")  # I made it print the calculated confidence interval.

def calculate_variance_confidence_interval(data, confidence_level=0.95):
    # Calculate the sample variance
    sample_variance = data.var()
    n = len(data)

    # Chi-Square critical values for the confidence level
    chi2_lower = stats.chi2.ppf((1 - confidence_level) / 2, n - 1)
    chi2_upper = stats.chi2.ppf(1 - (1 - confidence_level) / 2, n - 1)

    # Calculate the confidence interval for variance
    lower_bound = (n - 1) * sample_variance / chi2_upper
    upper_bound = (n - 1) * sample_variance / chi2_lower

    print(f"\n--- {confidence_level * 100}% Confidence Interval for Variance ---")
    print(f"Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")


# Created a function to perform a t-test to compare the sample mean with the expected population mean.
def perform_hypothesis_test(data, expected_mean):
    # I performed a t-test to compare the sample mean with the expected population mean.
    print("\n--- Hypothesis Test ---")

    print(f"Null Hypothesis (H₀): The population mean is {expected_mean:.2f}")  # I wrote the null hypothesis.
    print(f"Alternative Hypothesis (H₁): The population mean is not equal to {expected_mean:.2f}")  # I wrote the alternative hypothesis.

    # I performed the t-test
    t_stat, p_value = stats.ttest_1samp(data, expected_mean)  # I performed a one-sample t-test.

    print(f"T Statistic: {t_stat:.2f}")  # I printed the calculated t-statistic.
    print(f"P Value: {p_value:.4f}")  # I printed the p-value.

    # I defined the significance level (alpha).
    alpha = 0.05
    if p_value < alpha:  # If the p-value is smaller than alpha, I rejected the null hypothesis.
        print(f"Result: The P-value ({p_value:.4f}) is less than alpha ({alpha}).")
        print("We reject the null hypothesis, indicating a statistically significant difference.")
    else:  # If the p-value is equal to or greater than alpha, I failed to reject the null hypothesis.
        print(f"Result: The P-value ({p_value:.4f}) is greater than or equal to alpha ({alpha}).")
        print("We fail to reject the null hypothesis, suggesting no significant difference.")

# Created a function to calculate the required sample size based on margin of error and confidence level.
def calculate_sample_size(data, margin_of_error, confidence_level=0.95):
    # I calculated the required sample size based on the margin of error and confidence level.
    print("\n--- Sample Size Calculation ---")

    std_dev = data.std()  # I calculated the standard deviation of the data.
    z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)  # I calculated the Z-value for the confidence level.

    # I calculated the required sample size for the given margin of error.
    sample_size = (z_value * std_dev / margin_of_error) ** 2  # I calculated the sample size.

    print(f"Desired Margin of Error (E): {margin_of_error:.2f}")  # I printed the margin of error.
    print(f"Confidence Level: {confidence_level * 100}%")  # I printed the confidence level.
    print(f"Calculated Z-Value: {z_value:.2f}")  # I printed the calculated Z-value.
    print(f"Current Standard Deviation: {std_dev:.2f}")  # I printed the standard deviation.
    print(f"Minimum Required Sample Size: {int(np.ceil(sample_size))} (rounded up)")  # I printed the minimum required sample size.

# Created the main function to run all the analysis steps.
def main():
    file_path = "data.csv"  # I specified the file path.
    data_frame = load_data(file_path)  # I loaded the data.

    if data_frame is None:  # If there was an error loading the data, I terminated the process.
        print("Analysis terminated due to data loading error.")
        return

    column_to_analyze = 'Social_event_attendance'  # I specified the column to analyze.
    if column_to_analyze not in data_frame.columns:  # If the specified column doesn't exist in the dataset:
        print(f"Error: '{column_to_analyze}' column not found in the dataset.")
        print(f"Available columns: {data_frame.columns.tolist()}")
        print("Analysis terminated.")
        return

    # I dropped missing values from the data.
    participation_data = data_frame[column_to_analyze].dropna()

    if participation_data.empty:  # If no valid data is left after dropping missing values, I terminated the process.
        print(f"No valid data found in '{column_to_analyze}' after dropping missing values. Analysis terminated.")
        return

    print(f"\n--- Proceeding with analysis on {len(participation_data)} valid data points ---")

    # I performed all analysis steps.
    calculate_basic_statistics(participation_data)
    calculate_variance_confidence_interval(participation_data)
    visualize_data(participation_data)
    detect_outliers(participation_data)
    calculate_confidence_interval(participation_data, confidence_level=0.95)
    expected_mean = participation_data.mean()
    perform_hypothesis_test(participation_data, expected_mean)
    calculate_sample_size(participation_data, margin_of_error=1.0, confidence_level=0.95)

    print("\n--- Analysis Complete ---")

# I made it run the program.
if __name__ == "__main__":
    main()
