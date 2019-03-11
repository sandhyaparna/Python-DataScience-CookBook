#### Install Python on Windows
when installing python check mark - add python to Path var
https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10 <br/>
https://www.youtube.com/watch?v=A7E18apPQJs <br/>
https://www.reddit.com/r/Python/comments/2crput/how_to_install_with_pip_directly_from_github/ <br/>
To install python packages using cmd <br/>
Setup python in Environmental variables - https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/ <br/>
If adding to environment vars doesnt work - set path=C:\Python24 in cmd, then type 'python' to check if it z working <br/>
set - cd PATHoftheModule <br/>
Type - python setup.py install] <br/>
pip.main(['install','-e','C:/Users/spashikanti/Downloads/scikit-plot-master']) <br/>

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
Upgrade Packages using pip - "python -m pip install --upgrade pip" #In Anaconda Prompt <br/>
conda update pip #In Anaconda Prompt <br/>
check pip version - pip --version #In Anaconda Prompt <br/>
 <br/>
To Open jupyter notebook
type - jupyter notebook in Anaconda prompt

#### Jupyter Notebook Basics
https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb
* Install packages directly in Jupyter notebook <br/>
import sys <br/>
!{sys.executable} -m pip install gensim <br/>
* Google colab: Install Keras with pip <br/>
!pip install -q keras <br/>
import keras <br/>

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
  * Below code works <br/> 
git clone https://github.com/cbellei/word2veclite.git <br/> 
cd word2veclite <br/>
pip install .  <br/>
  * If setup.py is not present the above code doesn't work. Just an unfinished, broken package, it isn't supposed to be installed by Python tools. Install it manually — clone the repo and copy files to site-packages.
  * Not sure if the below one works <br/> 
pip install -e git+https://github.com/package-name/package-name.git#egg=package-name <br/>
and then again type command - pip install package-name <br/>
  * Another  <br/> 
Type the command GitBash 'pip install git+https://github.com/cbellei/word2veclite.git' <br/> 
Type in cmd 'pip install word2veclite' <br/> 
Packages are stored in this location - C:\Users\User\AppData\Local\Programs\Python\Python37\Lib\site-packages <br/> 
* Install torch, use command in this website - https://pytorch.org/ <br/>
* Install fastai - conda install -c pytorch -c fastai fastai  <br/>
https://forums.fast.ai/t/howto-installation-on-windows/10439 <br/>
Install using new env using conda https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652 <br/>
https://github.com/fastai/fastai/blob/master/README.md#installation <br/>
Set up AWS EC2 instance for fastai https://course.fast.ai/start_aws.html <br/>
pip install --upgrade setuptools <br/>
* Install Tensorflow:  <br/>
Step by step instructions: https://media.readthedocs.org/pdf/tensorflow-object-detection-api-tutorial/latest/tensorflow-object-detection-api-tutorial.pdf <br/>
Install java, check bios update, install latest drivers from dell <br/>
Tensorflow is not supported in 3.7, install 3.6 version from https://www.python.org/downloads/windows/ <br/>
python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl #Whole line is a command <br/>
https://github.com/bhavsarpratik/install_Tensorflow_Windows #Requires gpu <br/>
https://medium.com/datadriveninvestor/install-python-3-6-and-tensorflow-92eeff0ad4f5 <br/>
* A few ways to solve errors
https://www.tensorflow.org/install/errors  <br/>
https://github.com/aymericdamien/TensorFlow-Examples <br/>
#### Create Environments
https://media.readthedocs.org/pdf/tensorflow-object-detection-api-tutorial/latest/tensorflow-object-detection-api-tutorial.pdf  <br/>
To identify environments installed: conda info --envs   <br/>
Environments are loaded at: C:\ProgramData\Anaconda3\envs  <br/>
Ideally you should install each variant(tensorflow & tensorflow-gpu) under a different (virtual) environment.  <br/>
* tensorflow_cpu - Environment for 
* tensorflow - Environment to do chapter2,3 of Tensorflow setup documentation pdf

<br/>
conda install scipy <br/>
pip install --upgrade sklearn <br/>
pip install --upgrade pandas <br/>
pip install --upgrade pandas-datareader <br/>
pip install --upgrade matplotlib <br/>
pip install --upgrade pillow <br/>
pip install --upgrade requests <br/>
pip install --upgrade h5py <br/>
pip install tensorflow==1.8.0 <br/>
pip install keras==2.2.0 <br/>
 <br/>
 
#### To install packages that has '-' in their package name
Use either _ instead of -  <br/>
Use no space/join words instead of - <br/>
Use shortform direct in import command. for eg- scikit-optimize is used in pip install command, but in import command directly use skopt
 <br/>
 
#### Install in Rodeo - New Version of pip ###
* Install/Upgrade Packages
import pip <br/>
import subprocess <br/>
subprocess.check_call(["python", '-m', 'pip', 'install', 'scipy']) # install pkg - pkg name should all be small letter NO CAPITALS <br/>
subprocess.check_call(["python", '-m', 'pip', 'install',"--upgrade", 'scipy']) # upgrade pkg - pkg name should all be small letter NO CAPITALS <br/>
from scipy import * <br/>
import scipy as sy <br/>
* Check Version of pip installed <br/>
pip.__version__ <br/>
* List of all installed packages <br/>
import subprocess <br/>
import sys <br/>
reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']) <br/>
installed_packages = [r.decode().split('==')[0] for r in reqs.split()] <br/>
print(installed_packages) <br/>

#### Install in Rodeo - old Version of pip ###
* Upgrade pip <br/>
pip.main(['install', '--upgrade', 'pip']) <br/>
* Install and Upgrade Packages in Rodeo <br/>
import pip <br/>
pip.main(['install','SciPy']) # install pkg <br/>
pip.main(['install', '--upgrade', 'statsmodels']) # upgrade pkg <br/>
from scipy import * <br/>
import scipy as sy <br/>
  * Packages that have -
import pip <br/>
pip.main(['install','scikit-learn']) <br/>
pip.main(['install', '--upgrade', 'statsmodels']) <br/>
from sklearn import * <br/>
import sklearn as sklearn <br/>

#### Packages for Data Science
* General
import numpy as np <br/>
import pandas as pd <br/>
import os <br/>
* Visualization
import seaborn as sns <br/>
import matplotlib.pyplot as plt <br/>
%matplotlib inline <br/>

#### Packages unable to install


#### Errors
* Could not find a version that satisfies the requirement opencv (from versions: )  No matching distribution found for opencv
  * Either the python version required for the package is not present 
  * Module/Package name is wrong/incorrect. Eg codecs is actually part of 'openapi-codec'
* Building wheel error: pip install --no-cache-dir MODULENAME

###### Path Var
C:\Program Files (x86)\Common Files\Oracle\Java\javapath;C:\Program Files (x86)\Intel\iCLS Client\;C:\Program Files\Intel\iCLS Client\;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files\Hewlett-Packard\SimplePass\;C:\Program Files\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files\Intel\WiFi\bin\;C:\Program Files\Common Files\Intel\WirelessCommon\;C:\Program Files\Git\cmd;C:\Users\User\AppData\Local\Programs\Python\Python37\Scripts\;C:\Users\User\AppData\Local\Programs\Python\Python37\;C:\Users\User\Anaconda3;C:\Users\User\Anaconda3\Library\mingw-w64\bin;C:\Users\User\Anaconda3\Library\usr\bin;C:\Users\User\Anaconda3\Library\bin;C:\Users\User\Anaconda3\Scripts;C:\Users\User\AppData\Local\rodeo\app-2.5.2\bin;C:\Users\User\AppData\Local\rodeo\app-2.5.2\resources\conda

