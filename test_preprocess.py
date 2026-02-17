from preprocess_data import clean_data

df = clean_data("clean_data.csv")  # use your actual file name

print(df.head())
print("Data cleaned successfully!")
