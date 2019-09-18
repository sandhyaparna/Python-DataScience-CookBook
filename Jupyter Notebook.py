# “4 Awesome Tips for Enhancing Jupyter Notebooks” by George Seif https://link.medium.com/OY7PpANt5Z

# Import data
Df = pd.read_csv('path/file.csv', encoding='latin-1')

# Description
def return_desc(df): 
    return print (df.dtypes),print (df.head(3)) ,print(df.apply(lambda x: [x.unique()])), print(df.apply(lambda x: [len(x.unique())])),print (df.shape)




