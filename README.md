# hypothalamus_seg

This page hosts the code related to the following publication: \
\
**[Automated segmentation of the Hypothalamus and associated subunits in 
brain MRI](https://www.sciencedirect.com/science/article/pii/S1053811920307734)** \
B. Billot, M. Bocchetta, E. Todd, A. V. Dalca, J. D. Rohrer, J. E. Iglesias\
NeuroImage (in press)

\
**IMPORTANT**: any volume analysis made with this code downloaded before the **17/11/2020** will have to be rerun, as we 
noticed a problem in the volume computation. This issue has now been fixed, and the compile code 
(distributed in zip files) has been updated. This does not concern the segmentations, which are left unchanged.

\
This tool is now available in the development version of **[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)** ! \
See how to use the FreeSurfer version [here](https://surfer.nmr.mgh.harvard.edu/fswiki/HypothalamicSubunits).

\
This repository enables automated segmentation of the hypothalamus and its associated subunits in **T1-weighted scans of 
approximatively 1mm isotropic resolution**. \
The presented tool is based on a convolutional neural network, which outputs segmentations in a very short 
processing time (around 10 seconds with a CPU, less than a second with a GPU). \
The network was trained by applying aggressive data augmentation, which makes it robust against variability 
in acquisition parameters (sequence, platform, head positioning), and in anatomy (e.g. atrophy patterns linked to 
ageing or different pathologies).

More specifically, this code produces segmentation maps with 11 labels:

| Label number | Associated subunit      | Colour in figure below |
| :---         | :---                    | :---                   |
| 0            | background              | N/A                    |
| 1            | left anterior-inferior  | yellow                 |
| 2            | left anterior-superior  | blue                   |
| 3            | left posterior          | orange                 |
| 4            | left tubular inferior   | pink                   |
| 5            | left tubular superior   | green                  |
| 6            | right anterior-inferior | yellow                 |
| 7            | right anterior-superior | blue                   |
| 8            | right posterior         | orange                 |
| 9            | right tubular inferior  | pink                   |
| 10           | right tubular superior  | green                  |

\
where the aforementioned subunits are defined by the following hypothalamic nuclei: \
\
**Anterior-inferior**: suprachiasmatic nucleus; supraoptic nucleus (SON) \
**Anterior-superior**: preoptic area; paraventricular nucleus (PVN) \
**Posterior**: mamillary body (including medial and lateral mamillary nuclei); lateral hypothalamus; 
tuberomamillary nucleus (TMN) \
**Tubular inferior**: infundibular (or arcuate) nucleus; ventromedial nucleus; SON; lateral tubular nucleus; TMN \
**Tubular superior**: dorsomedial nucleus; PVN; lateral hypothalamus

\
The following figure shows two segmentation examples (in coronal slices) obtained by this tool for a control 
subject and a subject diagnosed with Alzheimer's disease.

![Segmentation examples](data/hypo.png)



----------------

### Dependencies

**No requirement is needed to run the CPU version of this code, apart from downloading the tool itself.** \
The dependencies listed below are already provided with the tool, and are simply given here for information.

This code was implemented in python 3.6. The neural network was built in Keras with a Tensorflow 2.0 backend. An 
exhaustive list of the python dependencies can be found under `requirements.txt`. \
Moreover, this repository relies on several external python packages (already included for convenience):
- **[lab2im](https://github.com/BBillot/lab2im)**: contains functions for data augmentation, and image processing tools 
[1],
- **[neuron](https://github.com/adalca/neurite)**: contains functions for data augmentation, and to build the network 
[2,3],
- **[pystrum](https://github.com/adalca/pystrum)**: library required by the neuron package.

----------------

### Installation

##### 1- installing hypothalamus_seg
In order to download the segmentation tool (but not the code), click on one of the following links (depending on your 
operating system):
- Linux: [hypo_seg_linux.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/rmappmb_ucl_ac_uk/EXofMBCj_5ROmCFjX4Tj1QIBFRTzGSPsvbcshbTgrVORZQ?e=jY6gcg)
- Mac: [hypo_seg_mac.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/rmappmb_ucl_ac_uk/EY608KRqCtlFj8mAgHGhp4wBsV0MgLolsgbZ2MUplPQ8Ww?e=kg6reO)
- Windows: [hypo_seg_windows.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/rmappmb_ucl_ac_uk/EYx8DX_LNzRFgGlAs2RhatoBHNntJpqFncq10osxSG6Y_A?e=npaGEb)

This will take you to a OneDrive page where you can download a zip file by directly clicking on `Download` (top left). \
Once the file is downloaded, move it to the desired location on your computer, and unzip it. 

That's it, you can now use the CPU version of hypothalamus_seg !

##### 2- optional: running hypothalamus_seg on a GPU
The current set-up will allow you to obtain segmentations in approximatively 10s per scan (depending on your CPU).
If you have a GPU on your machine, this can be decreased below 1s per scan by installing the required libraries (GPU
driver, Cuda, CuDNN). \
These libraries are generally required for a GPU, and are not specific to hypothalamus_seg. In fact, you may already 
have installed them. In this case you can directly use this tool without taking any further actions, as the code will 
automatically run on your GPU. \
We show here how to install a GPU for an Nvidia graphic card. We warn the user that the following steps are more 
difficult than previously, as they may require to change some hardware settings.

- First we need to install cuda-10.0, and the Nvidia driver. While these two can be installed at the same time, we
recommend to install them separately, starting with the driver. Guidelines on how to do this can be found
[here](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux). We recommend the 
manual installation.

- Cuda-10.0 can be downloaded from [here](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal).
Again, we recommend to use the runfile installation. Guidelines on how to install Cuda-10.0 are explained in details in
[this document](https://developer.download.nvidia.com/compute/cuda/10.0/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf).
You can check if the installation was successful by typing `nvidia-smi` in a command line.

- Finally you will need to download the cudnn-7.0 library. The step-by-step installation is available
[here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) (knowing that you can skip the step 2.1.1, 
since you already installed the graphic driver).

You can verify if this installation was successful by running this tool and looking for the name of your GPU in the 
printed output (in the terminal window).

##### 3- optional: developer installation

If you wish to train your own model, or to have a better look at the code, you can directly download this repository 
with:
```
git clone https://github.com/BBillot/hypothalamus_seg.git
```
This code includes functions to predict and evaluate the trained models. We also provide scripts to call the training
and predicting functions from a terminal window (see [scripts](scripts)). \
See the Dependencies section for further details on the requirements.

----------------

### How to use this code

**IMPORTANT**: The path to the `hypothalamus_seg` folder (e.g. `/home/user1/hypothalamus_seg`) will now be referred to 
as `<path to hypothalamus_seg>`. Thus, do not forget to replace it in the following instructions.

To use this code, you will first need to open a terminal window:
- in Linux: Ctrl+Alt+T
- in Mac: Go to Applications>Utilities and open Terminal
- in Windows: Search for PowerShell, right click on it, and select `Run as administrator`:

Once the terminal window is open, navigate to the folder `hypothalamus_seg`.
This can be done with the `cd` command:
```
cd <path to hypothalamus_seg>
```
Depending on your operatin system, you can then simply segment images by calling:
```
# In Linux or in Mac:
./hypo_seg --i <image> --o <segmentation> --post <post> --resample <resample> --vol <vol>

# In Windows:
./hypo_seg.exe --i <image> --o <segmentation> --post <post> --resample <resample> --vol <vol> 
```
where (in all cases):
- `<image>` is the path to an image to segment. \
This can also be a folder, in which case all the images inside that folder will be segmented.
- `<segmentation>` is the path where the output segmentation(s) will be saved. \
This must be a folder if `<image>` designates a folder.
- `<post>` (optional) is the path where the posteriors (given as soft probability maps) will be saved. \
This must be a folder if `<image>` designates a folder.
- `<resample>` (optional) hypothalamus_seg was trained on images at approximately 1mm isotropic resolution, 
so it would be underperforming if presented with images of higher/lower resolutions. Therefore, any image with a 
resolution outside the [0.95, 1.15] range will be internally resampled to 1mm resolution. Use this optional `--post`
flag to save the resampled images. It must be the path to a single image, or a folder if `<image>` designates a folder.
Also, note that if the image is resampled, the segmentation will be given at 1mm resolution as well, regardless of 
whether or not the `--post` flag is used.
- `<vol>` (optional) is the path to an output csv file where the volumes of all subunits
will be saved for all segmented scans (one csv file for all subjects; e.g. /path/to/volumes.csv)

\
Additional optional flags are also available:
- `--crop`: to run the code on a smaller patch of the provided shape (example: `--crop 160` to run on 160<sup>3</sup> 
patches). The patches are extracted around the centre of the inputs. This value defaults to 184, but it can be decreased
for faster analysis or to fit in your GPU.
- `--threads`: to indicate the number of cores to be used if running on a CPU (example: `--threads 3` to run on 3 cores).
 This value defaults to 1, but we recommand increasing it for faster analysis.
- `--cpu`: to enforce the code to run on the CPU, even if a GPU is available.

\
You can have access to these explanations directly by typing once in `<path to hypothalamus_seg>`:
```
./hypo_seg -h        # In Linux or in Mac
./hypo_seg.exe -h    # In Windows
```

**IMPORTANT 1**: Because hypothalamus_seg may resample the images at 1mm isotropic resolution, some viewers will not 
display them correctly when overlaying the segmentations on the original images. If that’s the case, you can use 
the `--resample` flag to obtain a resampled image that lives in the same space as the segmentation, such that any viewer
can be used to visualize them together. We highlight that the resampling is performed internally to avoid the dependence
 on any external tool.

**IMPORTANT 2**: If you wish to run this tool on several images, we recommend that you put them in a single folder and
run hypothalamus_seg on this folder, rather than calling it individually on each image. That way you can save time by
avoiding to setup all the required libraries for each image, which typically takes 60% of the runtime for the CPU 
version, and more than 90% for the GPU version.

----------------

### Contacts

If you have any question regarding the usage of this code, or any suggestions to improve it, you can contact us at: \
benjamin.billot.18@ucl.ac.uk

----------------

### References

[1] **[A Learning Strategy for Contrast-agnostic MRI Segmentation](https://arxiv.org/abs/2003.01995)** \
Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias, Adrian V. Dalca \
MIDL 2020 

[2] **[Anatomical Priors in Convolutional Networks for Unsupervised Biomedical 
Segmentation](http://www.mit.edu/~adalca/files/papers/cvpr2018_priors.pdf)** \
Adrian V. Dalca, John Guttag, Mert R. Sabuncu \
CVPR 2018 

[3] **[Unsupervised Data Imputation via Variational Inference of Deep Subspaces](https://arxiv.org/abs/1903.03503)** \
Adrian V. Dalca, John Guttag, Mert R. Sabuncu \
Arxiv preprint 2019 