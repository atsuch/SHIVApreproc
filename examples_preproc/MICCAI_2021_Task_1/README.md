## Preprocessing MICAI 2021 Task 1


In this example we are providing pre-processing of 3 subjects.<br />
<br />
Modalities which were used: T1w, FLAIR<br />

We assume that we have the brain masks created with T1w images.<br />
The T1w images do not need to be resampled to 1 x 1 x 1 mm.<br />

- The config_process_subject.json in this case:
<br />
 Please note what we do provide the path to brainmask and we don't needs to resample T1w images so the json file will be looking like following: 
<br />
 <pre>
{
    "svn_dir":             "/homes_unix/SHIVApreproc/shiva_preproc/",
    "wd":                  "/homes_unix/SHIVApreproc/examples_preproc/MICCAI_2016/preproc_images",
    "data_dir":            "/homes_unix/SHIVApreproc/examples_preproc/MICCAI_2016/raw_images",
    "in_dat_tmp":          "%s_%s.%s",
    <b>"resampling_to_111":   "False"</b>,    
    "spm_standalone":      "/srv/shares/softs/spm12/run_spm12.sh",
    "path_to_spm":         "/srv/shares/softs/spm12-full",
    "mcr":                 "/srv/shares/softs/MCR/v713",
    "in_dat_tmp_arg": {
       "T1": [["subject_id", "T1", "nii.gz"]],
       "FLAIR": [["subject_id", "FLAIR", "nii.gz"]],
       "brainmasks": [["subject_id", "brainmask", "nii.gz"]]
       },
    "plugin":              "MultiProc",
    "plugin_args":         {"n_procs": 25}   
}
</pre>

 - subjects_list.txt in this case:
<pre>
101
102
103 </pre>
