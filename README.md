## 1. Random Forest for Earth Observation

<table>
  <tr>
    <th width = 200>created @ the NOC</th>
    <th><img src="/docs/images/NOC_logo.png" width="100"></th>
  </tr>
</table>
<table>
  <tr>
    <th width = 200>for the ReSOW project</th>
    <th><img src="/docs/images/ReSOW_logo.png" width="300"></th>
  </tr>
</table>
<table>
  <tr>
    <th width = 200>funded by UKRI</th>
    <th><img src="/docs/images/SMMR_logo.png" width="150"</th>
  </tr>
</table>

The RF4EO package applies random forest classsification to optical satellite data and uses a multi-classifier system to enhance results. 

<img src="/docs/images/seagrass.jpg" width="750">

## 1. Installation

### 1.1 Download or clone the **RF4EO** repository

**download option** - navigate to the latest release **(v1.0)** on the RHS of the root RF4EO repo page and download and unzip the source code.

**clone option** - from your local machine clone the repository using the OcTyPy repo address https://github.com/NOC-EO/RF4EO.


### 1.2 Create an environment with Anaconda

To run the code in the project you need to install the required Python packages in an environment. This can be done using **Anaconda**, which can be downloaded [here](https://mamba.readthedocs.io/en/latest/index.html).

Open the Anaconda Prompt windowand use the `cd` command (change directory) to navigate to the directory where you installed the **RF4EO** repository.

Create a new anaconda environment named **rf4eo** with all the required packages and activate this environment by entering the following commands:

```
conda env create --file env\environment.yml
conda activate rf4eo
```

If you have successfully created and activated the **rf4eo** anaconda environment your terminal command line prompt should now start with `(rf4eo)`.


### 1.3 Create an environment with Mamba

To run the code in the project you need to install the required Python packages in an environment. This can be done using **Mamba**, which can be downloaded [here](https://www.anaconda.com/download/).

Open the Mamba Prompt window and create and activate this environment by entering the following commands.

```
mamba create -n rf4eo gdal pandas prettytable proj scikit-image scikit-learn -c conda-forge
mamba activate rf4eo
```

Use the `cd` command (change directory) to navigate to the directory where you installed the **RF4EO** repository to run the code.


## 3. Documentation

to follow



## References

 http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
