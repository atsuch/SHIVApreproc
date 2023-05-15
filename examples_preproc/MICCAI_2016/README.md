## Preprocessing MICAI 2016


In this example we provide the preprocessing of 3 subjects.<br />
<br />
Modalities which were used: T1w, FLAIR<br />
We don't provide the brainmask so it's will be generated automaticly in preprocessing process using SPM.<br />
The T1w images needs to be resampling to 1 by 1 by 1 mm.<br />

- The config_process_subject.json in this case:
<br />
 Please note what we don't provide the path to brainmask since we want to generate it during preprocessing and 
 we needs to resample T1w images we put  "resampling_to_111": <b>"True"</b>.
<br />
 <pre>
{
    "svn_dir": "/homes_unix/iastafeva/dev/dev_preproc_github/SHIVApreproc/shiva_preproc/",
    "wd": "/homes_unix/iastafeva/dev/dev_preproc_github/SHIVApreproc/examples_preproc/MICCAI_2016/preproc_images",
    "data_dir": "/homes_unix/iastafeva/dev/dev_preproc_github/SHIVApreproc/examples_preproc/MICCAI_2016/raw_images",
    "in_dat_tmp": "%s_%s.%s",
    "resampling_to_111": <b>"True"</b>,    
    "spm_standalone": "/srv/shares/softs/spm12/run_spm12.sh",
    "path_to_spm": "/srv/shares/softs/spm12-full",
    "mcr": "/srv/shares/softs/MCR/v713",
    "in_dat_tmp_arg": {
       "T1": [["subject_id", "T1", "nii.gz"]],
       "FLAIR": [["subject_id", "FLAIR", "nii.gz"]]
       },
    "plugin": "MultiProc",
    "plugin_args": {"n_procs": 25}   
}
</pre>

 - subjects_list.txt in this case:
<pre>
10304
10804
20804 </pre>
