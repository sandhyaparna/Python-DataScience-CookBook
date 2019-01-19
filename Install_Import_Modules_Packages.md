#### Install Python on Windows
https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10 <br/>
https://www.youtube.com/watch?v=A7E18apPQJs <br/>
https://www.reddit.com/r/Python/comments/2crput/how_to_install_with_pip_directly_from_github/ <br/>

#### Install git 
* In git - Type bash, git commands to see if they are working <br/>

#### Install C++ compiler for python
Download Visual studio installer - https://visualstudio.microsoft.com/downloads/ <br/>
How to install visual studio installer - https://stackoverflow.com/questions/29846087/microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat

#### Install Anaconda 64 bit, git
http://docs.anaconda.com/anaconda/user-guide/getting-started/ <br/>
* After installation, check if anaconda prompt and anaconda navigator are installed <br/>
* In Anaconda prompt, type python, conda (or) conda list , jupyter notebook to see if they are installed or not  <br/>
* Run Anaconda navigator as Administrator if error related to user appears <br/>
 <br/>
If python/conda are not recoginzed - https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444 <br/>
https://stackoverflow.com/questions/31935714/how-to-install-rodeo-ide-in-anaconda-python-distribution <br/>
https://medium.com/@GalarnykMichael/setting-up-pycharm-with-anaconda-plus-installing-packages-windows-mac-db2b158bd8c <br/>
https://stackoverflow.com/questions/11438727/how-to-use-subprocess-to-run-a-virtualenv-package-install <br/>
https://www.youtube.com/watch?v=z0qhKP2liHs <br/>
 <br/>
Update Anaconda - "conda update conda" , "conda update --all" <br/>
Install packages - conda install package-name #In Anaconda Prompt <br/>
Update packages - conda update package-name #In Anaconda Prompt <br/>
 <br/>
Upgrade pip using command - "python -m pip install --upgrade pip" #In Anaconda Prompt <br/>
conda update pip #In Anaconda Prompt <br/>
check pip version - pip --version #In Anaconda Prompt <br/>
 <br/>
Install python packages from github <br/>
In Anaconda prompt - pip install -e git+https://github.com/package-name/package-name.git#egg=package-name <br/>
and then again type command - pip install package-name <br/>

#### Install Specific Packages - Anaconda website, whl, github
* Install using Anaconda website, https://conda.anaconda.org/conda-forge ,look for package and see the syntax associated with the specific package u r looking for <br/>
conda install -c https://conda.anaconda.org/conda-forge wordcloud #wordcloud package <br/>
conda install -c anaconda tensorflow #tensorflow package <br/>
https://stackoverflow.com/questions/41409570/cant-install-wordcloud-in-python-anaconda <br/>
* Install packages from whl: wordcloud, rpy2 Save file from below link in the cd directory of Anaconda Prompt  <br/>
pip install rpy2-2.9.5-cp37-cp37m-win_amd64.whl <br/>
https://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2  <br/>
https://stackoverflow.com/questions/30083067/encountering-error-when-installing-rpy2-tried-to-guess-rs-home-but-no-r-comman <br/>
* Install from github <br/>
git clone https://github.com/cbellei/word2veclite.git <br/> <br/>
cd word2veclite <br/>
pip install .  <br/>
* Install torch, use command in this website - https://pytorch.org/ <br/>
* Install fastai - https://forums.fast.ai/t/howto-installation-on-windows/10439 <br/>
pip install --upgrade setuptools <br/>
* Install Tensorflow:  <br/>
python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl #Whole line is a command <br/>
https://github.com/bhavsarpratik/install_Tensorflow_Windows #Requires gpu <br/>

 
conda install scipy
pip install --upgrade sklearn
pip install --upgrade pandas
pip install --upgrade pandas-datareader
pip install --upgrade matplotlib
pip install --upgrade pillow
pip install --upgrade requests
pip install --upgrade h5py
pip install tensorflow==1.8.0
pip install keras==2.2.0


# To Open jupyter notebook
type - jupyter notebook in Anaconda prompt


when installing python check mark - add python to Path var
# To install python packages using cmd
Setup python in Environmental variables - https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/
If adding to environment vars doesnt work - set path=C:\Python24 in cmd, then type 'python' to check if it z working
set - cd PATHoftheModule
Type - python setup.py install]

pip.main(['install','-e','C:/Users/spashikanti/Downloads/scikit-plot-master'])


### New Version of pip
import pip
import subprocess
subprocess.check_call(["python", '-m', 'pip', 'install', 'scipy']) # install pkg - pkg name should all be small letter NO CAPITALS
subprocess.check_call(["python", '-m', 'pip', 'install',"--upgrade", 'scipy']) # upgrade pkg - pkg name should all be small letter NO CAPITALS
from scipy import *
import scipy as sy

### List of installed packages
import subprocess
import sys
reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
print(installed_packages)

# Version of pip installed
pip.__version__

### Packages for Data Science
# General
import numpy as np
import pandas as pd
import os

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# To install packages that has '-' in their package name
Use either _ instead of -
Use no space/join words instead of -
Use shortform direct in import command. for eg- scikit-optimize is used in pip install command, but in import command directly use skopt


################## OLD VERSION pip ##################
### Upgrade pip
pip.main(['install', '--upgrade', 'pip'])

### Install and Upgrade Packages in Rodeo
import pip
pip.main(['install','SciPy']) # install pkg
pip.main(['install', '--upgrade', 'statsmodels']) # upgrade pkg
from scipy import *
import scipy as sy

import pip
pip.main(['install','scikit-learn'])
pip.main(['install', '--upgrade', 'statsmodels'])
from sklearn import *
import sklearn as sklearn


### Import from Git
# Type the command GitBash
pip install git+https://github.com/cbellei/word2veclite.git
# Type the below in cmd
pip install word2veclite
# Packages are stored in this location - 
C:\Users\User\AppData\Local\Programs\Python\Python37\Lib\site-packages


C:\Program Files (x86)\Common Files\Oracle\Java\javapath;C:\Program Files (x86)\Intel\iCLS Client\;C:\Program Files\Intel\iCLS Client\;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files\Hewlett-Packard\SimplePass\;C:\Program Files\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files\Intel\WiFi\bin\;C:\Program Files\Common Files\Intel\WirelessCommon\;C:\Program Files\Git\cmd;C:\Users\User\AppData\Local\Programs\Python\Python37\Scripts\;C:\Users\User\AppData\Local\Programs\Python\Python37\;C:\Users\User\Anaconda3;C:\Users\User\Anaconda3\Library\mingw-w64\bin;C:\Users\User\Anaconda3\Library\usr\bin;C:\Users\User\Anaconda3\Library\bin;C:\Users\User\Anaconda3\Scripts;C:\Users\User\AppData\Local\rodeo\app-2.5.2\bin;C:\Users\User\AppData\Local\rodeo\app-2.5.2\resources\conda


### Packages unable to install
# word2vec
# word2veclite
# fastai

