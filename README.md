# Coastal-Forecast
Web-based application to host a trained coastal forecast machine learning model.

## Setup and Starting of Virtual Environment
### TensorFlow
Anaconda makes it easy to install TensorFlow, enabling data science, machine learning, and artificial intelligence workflows.
TensorFlow with conda is supported on 64-bit Windows 7 or later, 64-bit Ubuntu Linux 14.04 or later, 64-bit CentOS Linux 6 or 
later, and macOS 10.10 or later.

### Installing TensorFlow
1) Download and install [Anaconda](https://www.anaconda.com/products/individual) 
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2) On Windows open the Start menu and open an Anaconda Command Prompt. On macOS or 
Linux open a terminal window. Use the default bash shell on macOS or Linux.
3) Choose a name for your TensorFlow environment, such as "tf".
4) To install the current release of CPU-only TensorFlow, recommended for beginners:
> conda create -n tf tensorflow
> 
> conda activate tf
* Or, to install the current release of GPU TensorFlow on Linux or Windows:
> conda create -n tf-gpu tensorflow-gpu
> 
> conda activate tf-gpu
* TensorFlow is now installed and ready to use.

The environment will be activated, and you should see the environment variable in place of "base", i.e.:
  * (tf) C:\Users\\...>

For using TensorFlow with a GPU, refer to the [TensorFlow documentation](https://www.tensorflow.org/guide/gpu) 
on the topic, specifically the section on [device placement](https://www.tensorflow.org/guide/gpu#manual_device_placement).

Creating conda environment with YAML file:
> conda env create -f environment.yml
> 
> conda activate "environment name"

Creating pure Python virtual environment with requirements.txt file (non conda environment):
> pip install -r requirements.txt
> 
> (Windows): env\scripts\activate.bat
> 
> (Linux or MacOS): source env/bin/activate

For MacOSX M1, refer to this video by [Jeff Heaton](https://youtu.be/_CO-ND1FTOU).

## Starting/Running the Server
* Running <b>run.py</b> will activate the Flask server
* Follow the link provided by IDE, terminal/console, or copy and paste the line below into your browser.
<br/>&ensp;127.0.0.1:5000
