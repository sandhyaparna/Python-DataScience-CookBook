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

### Install packages that consists of packages to be installed from CliNER
!git clone https://github.com/text-machine-lab/CliNER.git
 
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
# Upload Requirements file here

!pip install -r requirements.txt

# Install pandas_profiling
# run the command: pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip 
# then restart the kernal
# run the command: from pandas_profiling import ProfileReport





