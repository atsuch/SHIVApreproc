# SHIVAtools_preproc
Preprocessing pipelines to use SHIVA tools

## Description
This repository contains simple preprocessing pipelines to process raw Nifti images to be ready for SHIVA segmentation tools:

- SHIVA-WMH: White matter hyperintensity segmentation based on T1w and FLAIR (https://github.com/pboutinaud/SHIVA_WMH/tree/main/WMH)
- SHIVA-PVS: Parivascular space segmentation based on T1w (https://github.com/pboutinaud/SHIVA_PVS)

The pipelines performs:

1) Reorient image to standard (RAS/LAS)
2) Coregister images to a reference modality (usually T1w)
3) Resample images to 1x1x1 mm
4) Brain mask generation (if not provided)
5) Image cropping to 160 × 214 × 176 
6) Intensity normalization inside the brain mask  (min-max normalization with the max set to the 99th percentile of the brain voxel values to avoid "hot spots")

## Dependency

- FSL
- Freesurfer v7.3.2 (for brain mask generation based on SynsthSeg)
- Nipype
- ...

