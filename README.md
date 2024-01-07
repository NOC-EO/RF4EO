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

**download option** - download the latest release, e.g. `(v1.0)`, on the RHS of the top level project page then unzip the source code locally.

or

**clone option** - from your local machine clone the repository from using GitHub address:

    https://github.com/NOC-EO/RF4EO

### 1.2 Create an environment with Anaconda
To run the code in the project you need to install the python packages that are required by RF4EO. To do this we will use Anaconda, which can be downloaded here.

Open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the cd command (change directory) to the directory where you have installed the SAR-TWL repository.

Create a new anaconda environment named sartwl with all the required packages and activate this environment by entering the following commands:

conda env create --file env\environment.yml
conda activate rf4eo
If you have successfully created and activated the sartwl anaconda environment your terminal command line prompt will now start with (rf4eo).

Finally run this command to add the project direction to the system path:

conda develop <project directory>
Where <project directory> is the directory where you installed the RFR4EO repository e.g. for Windows it might be c:\code\RF4EO-1.0


## 3. Documentation

to follow



## References

 http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
