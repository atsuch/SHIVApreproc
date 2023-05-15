# SHIVAtools_preproc
Preprocessing pipelines to use SHIVA tools
**`sample`**
## Description
This repository contains simple preprocessing pipelines to process raw Nifti images to be ready for SHIVA segmentation tools:

- SHIVA-WMH: White matter hyperintensity segmentation based on T1w and FLAIR (https://github.com/pboutinaud/SHIVA_WMH/tree/main/WMH)
- SHIVA-PVS: Parivascular space segmentation based on T1w (https://github.com/pboutinaud/SHIVA_PVS)

The pipelines performs:

1) Reorient image to standard (RAS/LAS)
2) Coregister images to a reference modality (usually T1w)
3) Resample images to 1x1x1 mm (if needed) 
4) Brain mask generation (if not provided using SPM)
5) Image cropping to 160 × 214 × 176 
6) Intensity normalization inside the brain mask  (min-max normalization with the max set to the 99th percentile of the brain voxel values to avoid "hot spots")

## Dependency

- FSL v.6.0.4 
- Freesurfer v7.3.2 (for brain mask generation based on SynsthSeg)
- Nipype v.1.7.0 (python package) 
- SMP12 (we are using with Matlab v.R2021a)

## Usage

1. To preproces the T1w and FLAIR raw images please clone this GitHub repository.
2. In SHIVApreproc folder run following command in Terminal to install shiva_preproc module:

python setup.py install

3. Specify the information in config_process_subject.json and subjects_list.txt, specificlly:

- config_process_subject.json
<pre>
{
    "svn_dir": "/path/to/shiva_preproc/folder", 
    "wd": "/path/where/preprocessed/images/will/be/store",
    "data_dir": "/path/to/raw/images",
    "in_dat_tmp": "%s_%s.%s",
    "in_dat_tmp_arg": {
        "T1": [["subject_id", "T1", "nii.gz"]],              **`/name of T1w images (subject_id_T1.nii.gz)`**
        "FLAIR": [["subject_id", "FLAIR", "nii.gz"]],      **`/name of FLAIR images (subject_id_FLAIR.nii.gz)`**
        "brainmask": [["subject_id", "braimask", "nii.gz"]] `**` /if brainmask were provided (subject_id_brainmask.nii.gz) else delete this line)`**` 
        },
    "resampling_to_111": "False",             <b> /if raw T1w images needs to be resample to 1 by 1 by 1 mm change to 'True'<b> 
    "path_to_spm": "/path/to/spm",                                     <b>  / exapmle:    "/srv/shares/softs/spm12-full",<b> 
    "spm_standalone": "/path/to/spm/standalone",                       <b>  / exapmle:    "/srv/shares/softs/spm12/run_spm12.sh",<b> 
    "mcr": "/path/to/mcr",                                             <b>  /example:     "/srv/shares/softs/MCR/v713"<b> 
    "plugin": "MultiProc",
    "plugin_args": {"n_procs": 25}
}
</pre>
- subjects_list.txt
Please provide the subjects IDs of T1w, FLAIR and/or brainmasks which you want to preproces.

4. Run the preprocessing_main.py to start the preprocessing of raw images:

python  preprocessing_main.py


