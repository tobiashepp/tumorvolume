# Tumorvolume

![TumorManual](https://github.com/lab-midas/tumorvolume/blob/master/images/sample_label.gif "Manually labeled tumor lesions")

<div style="text-align:center"><img src="https://github.com/lab-midas/tumorvolume/blob/master/images/sample_label.gif" /></div>
## Install 

Create a virtual environment with **python 3.7.4**

    conda create -n tumorvolume python=3.7.4

Install requirements for the *tumorvolume* projects as well as the *torch-mednet* sub-project.

    conda activate tumorvolume
    pip install -r requirements.txt

Go to the project root directory and run 

    pip install -e ./

Do the same for the sub-directory *torch-mednet*.
