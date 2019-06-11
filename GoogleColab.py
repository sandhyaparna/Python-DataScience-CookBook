### Loading data ###
from google.colab import files
uploaded = files.upload()

# After the file is uploaded
import io

# CSV files
Df = pd.read_csv(io.BytesIO(uploaded['Df.csv']))

# Pickle files
Df = pickle.load(io.BytesIO(uploaded['Df.pkl']))





