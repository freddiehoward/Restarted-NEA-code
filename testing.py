from data_fetcher import fetch_stock_data

from data_preprocessing import scaling_and_reshaping_raw_data

test_dataframe = fetch_stock_data("AAPL")

print(scaling_and_reshaping_raw_data(test_dataframe))

#should output something with mean 0 variance 1

