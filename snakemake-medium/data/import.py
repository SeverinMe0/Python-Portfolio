import statsmodels.datasets
data = statsmodels.datasets.heart.load_pandas().data
data.to_csv('./snakemake-medium/data/snakemake_test_data.csv', sep=',', encoding='utf-8')