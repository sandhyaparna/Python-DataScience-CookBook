### Loading data ###

# Uploading large files
https://www.freecodecamp.org/news/how-to-transfer-large-files-to-google-colab-and-remote-jupyter-notebooks-26ca252892fa/

from google.colab import files
uploaded = files.upload()

# After the file is uploaded
import io

# CSV files
Df = pd.read_csv(io.BytesIO(uploaded['Df.csv']))

# Pickle files
Df = pickle.load(io.BytesIO(uploaded['Df.pkl']))


### Exporting csv file to local machine ###
from google.colab import files
df.to_csv('filename.csv') 
files.download('filename.csv')

### Install packages that 
git clone https://github.com/text-machine-lab/CliNER.git
  

