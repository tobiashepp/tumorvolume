# Tumorvolume

![TumorAutomatic](https://github.com/lab-midas/tumorvolume/blob/master/images/sample_output.gif "Automatically labeled tumor lesions")

![TumorManual](https://github.com/lab-midas/tumorvolume/blob/master/images/sample_label.gif "Manually labeled tumor lesions")

![Organs](https://github.com/lab-midas/tumorvolume/blob/master/images/sample_organs.png "Metabolic activation in spleen, liver and spine bone marrow")

## Install 

Create a virtual environment:

    conda create -n tumorvolume python=3.7.4

Install requirements for the *tumorvolume* projects as well as the *torch-mednet* sub-project.

    conda activate tumorvolume
    pip install -r requirements.txt

Go to the project root directory and run:

    pip install -e ./

Do the same for the sub-directory *torch-mednet*.
