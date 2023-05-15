'''
A script for processing raw 3D images.

Perform the following steps:
1. Reorient the input images, and optionally resample to 1mm isotropic, and also optionally coregister all the modalities to ref-space (usually T1w).
2. If no brainmask in reference space is provided, use SPM to generate it (requires spm_standalone and mcr path in the configuration).
3. Apply the brain mask to the T1w and FLAIR images and crop it (with _CROP_DIMS) + intensity normalization.

@date: 11-04-2023
@author: atsuchida, iastafeva
'''
# %%
import os
import os.path as op
import json
from nipype import config, logging
from shiva_preproc.workflows import *
import shutil
from pathlib import Path
# %%

SUBJECTFILE = 'subjects_list.txt'  
CONFIGFILE = 'config_process_subject.json'

_CROP_DIMS = [160, 214, 176] 


def main(sublist_txt=SUBJECTFILE, config_json=CONFIGFILE, qc=False, plugin='MultiProc'):
    
    # read various dir setting etc
    with open(config_json) as json_file:
        prep_dict = json.load(json_file)

    # read sublist
    with open(sublist_txt) as f:
        subjects = f.read().splitlines()

    nmax = 50
    batches = [subjects[nmax*k:nmax*(k+1)] for k in range(int(len(subjects)/nmax)+1)]  
    try:
        batches.remove([])
    except ValueError:
        pass
    
    # save running python script to working directory
    try:
        script_name = Path(os.path.abspath(__file__))
        shutil.copy2(script_name, op.join(prep_dict['wd'], script_name.name))
    except NameError:
        print("WARNING: Script file is not copied to the logir. Started from an existing interactive session ?")

    logdir = op.join(prep_dict['wd'], 'log_dir') 

    for (i, batch_subs) in enumerate(batches):
        print ('Processing batch {}'.format(i))
        print ('Subjects in this batch:')
        print (batch_subs)
        # Create batch dir inside log_dir to keep log for each batch
        batch_logdir = op.join(logdir, 'batch{}'.format(i))
        os.makedirs(batch_logdir, exist_ok=True)
        batch_sub_txt = op.join(batch_logdir, 'batch_subjects.txt')
        with open(batch_sub_txt, 'w') as f:
            f.write('\n'.join(batch_subs))
        
        # 1) Reorientation, Resampling (if needed), Coregistration (FLAIR in T1w space) of raw images
        prep_wf_name = 'prepIm'
        print ('Preparing input images in the {} workflow...'.format(prep_wf_name))
        os.makedirs(op.join(batch_logdir, prep_wf_name), exist_ok=True)
        
        in_imnames = list(prep_dict['in_dat_tmp_arg'].keys())
        if 'brainmask' in in_imnames:
            ibrainmask = True
            in_imnames.remove('brainmask') 
        prep_wf = prepImages(
            name=prep_wf_name,
            base_dir=prep_dict['wd'],
            data_dir=prep_dict['data_dir'],
            subjects=batch_subs,
            in_dat_tmp=prep_dict['in_dat_tmp'],
            in_dat_tmp_arg=prep_dict['in_dat_tmp_arg'],
            im_names=in_imnames,
            reorient=True,  
            resample=prep_dict['resampling_to_111'],   
            ibrainmask=ibrainmask, 
            coreg_ref_space='T1')  
        
        prep_wf.config['execution']['crashfile_format'] = 'txt'
        
        config.update_config({
        'logging': {
            'log_directory': op.join(batch_logdir, prep_wf_name),
            'log_to_file': True
            },
        'execution': {'stop_on_first_rerun': False}
        })
        
        logging.update_logging(config)  
           
        prep_wf.run(plugin=prep_dict['plugin'], plugin_args=prep_dict['plugin_args'])

        # 2) Get SPM-based brainmask using T1w images if brainmask is not provided
        
        if not ibrainmask:
   
            brainmask_wf_name = 'getBrainmask'
            print ('Creating SPM-based brainmask in the {} workflow {} qc...'.format(brainmask_wf_name, 'with' if qc else 'without'))
            os.makedirs(op.join(batch_logdir, brainmask_wf_name), exist_ok=True)

            
            if prep_dict['resampling_to_111'] == 'True':
                in_dat_tmp_arg_prep ={'ref_main': [['{}Sink'.format(prep_wf_name), 'resampled_images/_subject_id_', 'subject_id', '/_resampImages0/*']]}
            else:
                in_dat_tmp_arg_prep ={'ref_main': [['{}Sink'.format(prep_wf_name), 'reoriented_images/_subject_id_', 'subject_id', '/_reorImages0/*']]}    
        
            brainmask_wf = getBrainmask(
                name=brainmask_wf_name,
                base_dir=prep_dict['wd'],
                data_dir = prep_dict['wd'],
                subjects=batch_subs,
                in_dat_tmp='%s/%s%s%s',
                in_dat_tmp_arg=in_dat_tmp_arg_prep,
                method='spm',
                method_options={
                    'spm_standalone': prep_dict['spm_standalone'],
                    'mcr': prep_dict['mcr'],
                    'paths': [prep_dict['path_to_spm']]
                    },
                qc=qc)
            brainmask_wf.config['execution']['crashfile_format'] = 'txt'
            config.update_config({
                'logging': {
                    'log_directory': op.join(batch_logdir, brainmask_wf_name),
                    'log_to_file': True
                }
                })
            logging.update_logging(config)
            brainmask_wf.run(plugin=prep_dict['plugin'], plugin_args=prep_dict['plugin_args'])
            
            bm = [['{}Sink'.format(brainmask_wf_name), 'spm_brainmask/_subject_id_', 'subject_id', '/brainmask.nii.gz']]
    
   
        # 3) Preproc images (cropped and intensity normaliztion of preprocessed images using _CROP_DIMS)      
        
        preproc_wf_name = 'Preproc'
        print ('Performing intensity normalizations of images in the {} workflow...'.format(preproc_wf_name))
        os.makedirs(op.join(batch_logdir, preproc_wf_name), exist_ok=True)
        
        if prep_dict['resampling_to_111'] == 'True':
            T1_prep = [['{}Sink'.format(prep_wf_name), 'resampled_images/_subject_id_', 'subject_id', '/_resampImages0/*']]
            bm = [['{}Sink'.format(prep_wf_name), 'resampled_brainmask/_subject_id_', 'subject_id', '/*']]
        else:
            T1_prep =[['{}Sink'.format(prep_wf_name), 'reoriented_images/_subject_id_', 'subject_id', '/_reorImages0/*']]   
           
        preproc_in_dat_tmp_arg = {
            'T1': T1_prep,
            'FLAIR': [['{}Sink'.format(prep_wf_name), 'coregistered_images/_subject_id_', 'subject_id', '/_coreg0/*']],
            'brainmask': bm
        }
        
        preproc_wf = PreprocImages(
            name=preproc_wf_name,
            SVN_dir=prep_dict['svn_dir'],
            path_to_spm = prep_dict['path_to_spm'],
            base_dir=prep_dict['wd'],
            data_dir=prep_dict['wd'],
            subjects=batch_subs,
            in_dat_tmp='%s/%s%s%s',
            in_dat_tmp_arg=preproc_in_dat_tmp_arg,
            im_names= ['T1', 'FLAIR'],
            crop_dims=_CROP_DIMS)
        preproc_wf.config['execution']['crashfile_format'] = 'txt'
        config.update_config({
            'logging': {
                'log_directory': op.join(batch_logdir, preproc_wf_name),
                'log_to_file': True
                 },
        'execution': {'stop_on_first_rerun': False}
            })
        logging.update_logging(config)
        preproc_wf.run(plugin=prep_dict['plugin'], plugin_args=prep_dict['plugin_args'])
        
        print ('Finished processing batch {}.'.format(i))
    print ('Finished processing all the batches!')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Runs processing workflows')
               
    parser.add_argument('--qc',                                 
                        dest='qc',
                        action='store_true',
                        help='produce qc plots and scores')
    parser.add_argument('--no-qc', 
                        dest='qc',
                        action='store_false',
                        help='skip qc plots and scores')

    args= vars(parser.parse_args())

    main(**args)