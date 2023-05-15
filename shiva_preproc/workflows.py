
from shiva_preproc.wf_utils import *
from shiva_preproc.subworkflow_int_norm import *
import os.path as op

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.algorithms.metrics import FuzzyOverlap

from nipype.interfaces.fsl.utils import  ConvertXFM, Reorient2Std, Split, ImageStats, ExtractROI, ConvertXFM
from nipype.interfaces.fsl.preprocess import FLIRT, ApplyXFM
from nipype.interfaces.fsl.maths import Threshold, MultiImageMaths, ApplyMask, BinaryMaths

from nipype.interfaces.spm.preprocess import NewSegment


def coregImages(name="coregImages",
                ref_imname="T1",
                coreg_imnames=["FLAIR", "T2"],
                cost_func="corratio",
                interp="trilinear",
                supply_mat=False):
    '''
    A WF to coregister a single image or a list of iamges to the ref_image, with the 
    supplied cost_func and interpolation.

    If the coreg_imnames is a single string object, coregistration is a simple node. 
    If it is a list of images, coregistration is a MapNode.

    When supply_mat is True, FLIRT is applied with the supplied matrix file.
    '''

    coregWF = Workflow(name)

    # inputNode
    in_fields = ['ref_image', 'coreg_images']
    if supply_mat:
        in_fields.append('mat_file')
    inputNode = Node(IdentityInterface(fields=in_fields), name='inputNode')

    if supply_mat:
        # Simply apply provided matrix files
        if isinstance(coreg_imnames, str):
            # make a simple node
            coreg = Node(ApplyXFM(), name="coreg")
            coreg.inputs.out_file = "{}_to_{}.nii.gz".format(coreg_imnames, ref_imname)
            
        else:
            # make MapNode
            coreg = MapNode(ApplyXFM(),
                            name="coreg",
                            iterfield=["in_file", "in_matrix_file", "out_file"])
            coreg.inputs.out_file = ["{}_to_{}.nii.gz".format(im_name, ref_imname) for im_name in coreg_imnames]
            
        coreg.inputs.apply_xfm = True
        coreg.inputs.cost = cost_func
        coreg.inputs.interp = interp
        coregWF.connect(inputNode, "ref_image", coreg, "reference")
        coregWF.connect(inputNode, "coreg_images", coreg, "in_file")
        coregWF.connect(inputNode, "mat_file", coreg, "in_matrix_file")

    else:
        # Compute coreg mat and also apply
        if isinstance(coreg_imnames, str):
            # make a simple node
            coreg = Node(FLIRT(), name="coreg")
            coreg.inputs.out_file = "{}_to_{}.nii.gz".format(coreg_imnames, ref_imname)
            coreg.inputs.out_matrix_file = "{}_to_{}.mat".format(coreg_imnames, ref_imname) 
        else:
            # Mapnode
            coreg = MapNode(FLIRT(),
                            name="coreg",
                            iterfield=["in_file", "out_file", "out_matrix_file"])
            coreg.inputs.out_file = ["{}_to_{}.nii.gz".format(im_name, ref_imname) for im_name in coreg_imnames]
            coreg.inputs.out_matrix_file = ["{}_to_{}.mat".format(im_name, ref_imname) for im_name in coreg_imnames]
            
        coreg.inputs.cost = cost_func
        coreg.inputs.interp = interp
        coreg.inputs.dof = 6
        coreg.inputs.no_resample_blur = True
        coregWF.connect(inputNode, "ref_image", coreg, "reference")
        coregWF.connect(inputNode, "coreg_images", coreg, "in_file")

        # Also compute th inverse transform
        if isinstance(coreg_imnames, str):
            # make a simple node
            getInvMat = Node(ConvertXFM(), name="getInvMat")
            getInvMat.inputs.out_file = "{}_to_{}.mat".format(ref_imname, coreg_imnames)
        else:
            getInvMat = MapNode(ConvertXFM(),
                                name="getInvMat",
                                iterfield=["in_file", "out_file"])
            getInvMat.inputs.out_file = ["{}_to_{}.mat".format(ref_imname, im_name) for im_name in coreg_imnames]
        getInvMat.inputs.invert_xfm = True
        coregWF.connect(coreg, "out_matrix_file", getInvMat, "in_file")
        
    # Node: coregWM.outputNode
    out_fields = ["coregistered_images"]
    if not supply_mat:
        out_fields.extend(["coreg_mat", "inv_coreg_mat"])
    
    outputNode = Node(IdentityInterface(fields=out_fields),
                                        name="outputNode")
    coregWF.connect(coreg, "out_file", outputNode, "coregistered_images")
    
    if not supply_mat:
        coregWF.connect(coreg, "out_matrix_file", outputNode, "coreg_mat")
        coregWF.connect(getInvMat, "out_file", outputNode, "inv_coreg_mat")
    

        
    return coregWF

def intNormImages(name='intNormImages',
                  base_dir='',
                  SVN_dir='',
                  path_to_spm ='',
                  im_names=['T1', 'FLAIR'],
                  unzip=True,
                  single_sub=False):
    '''
    A WF to intensity normalize a single image or a list of iamges inside the brain, 
    using the brainmask provided.

    If the im_names is a single string object, intensity normalization is a simple node. 
    If it is a list of images, it is a MapNode.

    If unzip is True, all the input images are unzipped prior to intensity normalization that uses
    the Matlab code that requires everything to be unzipped.
    '''

    intNormWF = Workflow(name)

    # inputNode
    inputNode = Node(IdentityInterface(fields=['subject_id', 'apply_to_images', 'brainmask']),
                     name='inputNode')

    # If images are not unzipped, unzip
    if unzip:
        if isinstance(im_names, str):
            unzipIm = Node(Gunzip(), name="unzipIm")
        else:
            unzipIm = MapNode(Gunzip(), name="unzipIm", iterfield=["in_file"])

        intNormWF.connect(inputNode, 'apply_to_images', unzipIm, 'in_file')

        unzipBrainmask = Node(Gunzip(), name="unzipBrainmask")
        intNormWF.connect(inputNode, 'brainmask', unzipBrainmask, 'in_file')

    # Intensity normalize and gzip output
    if isinstance(im_names, str):
        mapnode = False
        len_imnames = 1
        intNorm = Node(CustomIntensityNormalization(),
                       name="intNorm")
        intNorm.inputs.out_file = '{}_intensity_normed.nii'.format(im_names)

        gzipNormed = Node(Function(input_names=['in_file'],
                                   output_names=['out_file'],
                                   function=Gzip),
                          name='gzipNormed')
    else:
        mapnode = True
        len_imnames = len(im_names)
        intNorm = MapNode(CustomIntensityNormalization(),
                          name="intNorm",
                          iterfield=["in_file", "output_dir", "out_file"])
        intNorm.inputs.out_file = ['{}_intensity_normed.nii'.format(im) for im in im_names]

        gzipNormed = MapNode(Function(input_names=['in_file'],
                                      output_names=['out_file'],
                                      function=Gzip),
                             iterfield=['in_file'],
                             name='gzipNormed')
         
    intNorm.inputs.svn_dir = SVN_dir
    intNorm.inputs.path_to_spm = path_to_spm
    if unzip:
        intNormWF.connect(unzipIm, "out_file", intNorm, "in_file")
        intNormWF.connect(unzipBrainmask, "out_file", intNorm, "brain_mask")
    else:
        intNormWF.connect(inputNode, "apply_to_images", intNorm, "in_file")
        intNormWF.connect(inputNode, "brainmask", intNorm, "brain_mask")
    intNormWF.connect(inputNode,("subject_id", createOutputDir, base_dir, intNormWF.name, intNorm.name, single_sub, mapnode, len_imnames),
                      intNorm,"output_dir")
    intNormWF.connect(intNorm, "out_file", gzipNormed, "in_file")

    # OutputNode
    outputNode = Node(IdentityInterface(fields=['intensity_normed_images']),
                                        name="outputNode")
    
    intNormWF.connect(gzipNormed, "out_file", outputNode, "intensity_normed_images")

    return intNormWF

def prepSpmBrainmask(name="prepSpmBrainmask",
                     spm_standalone="/srv/shares/softs/spm12/run_spm12.sh",
                     mcr="/srv/shares/softs/MCR/v713",
                     spm_tpm_file="/srv/shares/softs/spm12-full/tpm/TPM.nii",
                     spm_seg_options={},
                     qc=True):
  
    # SPM seg options to use
    spm_seg_opts  = dict(
        channel_info = (0.001, 60, (False, False)),
        affine_regularization = 'mni',
        sampling_distance = 3,
        use_mcr = False,
        paths = ['/srv/shares/softs/spm12-full/'],
        warping_regularization = [0, 0.001, 0.5, 0.5, 0.2]
        )
    
    # Setup for SPM standalone
    if spm_standalone and mcr:
        from nipype.interfaces import spm
        matlab_cmd = ' '.join([spm_standalone, mcr, 'batch'])
        
        spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
 
        spm_seg_opts['use_mcr'] = True
        spm_seg_opts['mfile'] = False
        spm_seg_opts.pop('paths')
   
    # If any options are provided, change the default values
    if spm_seg_options:
        for k, v in spm_seg_options:
            spm_seg_opts[k] = v

    wf=Workflow(name)
    #inputNode
    inputNode = Node(IdentityInterface(fields=["ref_main"], mandatory_inputs=True),
                     name='inputNode')
    
    # Unzip for spm
    gunzip = Node(Gunzip(), name='gunzip')
    wf.connect(inputNode, 'ref_main', gunzip, 'in_file')

    # SPM NewSegment
    spmSeg_tissues = [
        ((spm_tpm_file, 1), 1, (True, False), (True, False)),
        ((spm_tpm_file, 2), 1, (True, False), (True, False)),
        ((spm_tpm_file, 3), 2, (True, False), (True, False)),
        ((spm_tpm_file, 4), 3, (True, False), (False, False)),
        ((spm_tpm_file, 5), 4, (True, False), (False, False)),
        ((spm_tpm_file, 6), 2, (False, False), (False, False))
        ]
    spmSeg = Node(NewSegment(**spm_seg_opts), name='spmSeg')
    spmSeg.inputs.tissues = spmSeg_tissues
    spmSeg.inputs.write_deformation_fields = [True, True]
    wf.connect(gunzip, 'out_file', spmSeg, 'channel_files')

    # Gzip tissue maps
    gzipNativeTissue = MapNode(Function(input_names=['in_file'],
                                        output_names=['out_file'],
                                        function=Gzip),
                               iterfield=['in_file'],
                               name='gzipNativeTissue')
    wf.connect(spmSeg, ('native_class_images', getElementFromListofList, 0), gzipNativeTissue, 'in_file')

    gzipNormNativeTissue = MapNode(Function(input_names=['in_file'],
                                        output_names=['out_file'],
                                        function=Gzip),
                                   iterfield=['in_file'],
                                   name='gzipNormNativeTissue')
    wf.connect(spmSeg, ('normalized_class_images', getElementFromListofList, 0), gzipNormNativeTissue, 'in_file')

    # Gzip deformation field maps
    gzipInvField = Node(Function(input_names=['in_file'],
                                 output_names=['out_file'],
                                 function=Gzip),
                        name='gzipInvField')
    wf.connect(spmSeg, 'inverse_deformation_field', gzipInvField, 'in_file')

    gzipForField = Node(Function(input_names=['in_file'],
                                 output_names=['out_file'],
                                 function=Gzip),
                        name='gzipForField')
    wf.connect(spmSeg, 'forward_deformation_field', gzipForField, 'in_file')

    # Combine GM/WM/CSF, threshold at 0.5 to vreate a brain mask
    opFileList = Node(Function(input_names=['item1', 'item2'],
                               output_names=['out_list'],
                               function=createListofItems),
                      name='opFileList')
    wf.connect(gzipNativeTissue, ('out_file', getElementFromList, 1), opFileList, 'item1')
    wf.connect(gzipNativeTissue, ('out_file', getElementFromList, 2), opFileList, 'item2')
    
    spmBrainmask = Node(MultiImageMaths(), name='spmBrainmask') 
    spmBrainmask.inputs.op_string = "-add %s -add %s -thr 0.5 -bin -fillh"
    spmBrainmask.inputs.out_file = "brainmask.nii.gz"
    wf.connect(gzipNativeTissue, ('out_file', getElementFromList, 0), spmBrainmask, 'in_file')
    wf.connect(opFileList, 'out_list', spmBrainmask, 'operand_files')

    # Use mask to get brain
    applyMask = Node(ApplyMask(), name="applyMask")
    applyMask.inputs.output_type = 'NIFTI_GZ'
    wf.connect(inputNode, "ref_main", applyMask, "in_file")
    wf.connect(spmBrainmask, "out_file", applyMask, "mask_file")

    if qc:
        # QC plot
        '''
        plotBrainMask = Node(MaskOverlayQCplot(), name="plotBrainMask")
        plotBrainMask.inputs.transparency = 1
        plotBrainMask.inputs.out_file = 'brainmask_plot.png'
        wf.connect(spmBrainmask, "out_file", plotBrainMask, "mask_file")
        wf.connect(inputNode, "ref_main", plotBrainMask, "bg_im_file")
        '''


        # Fuzzy overlap of TPM vs normalized tissue class
        splitTPM = Node(Split(), name='splitTPM')
        splitTPM.inputs.dimension = 't'
        splitTPM.inputs.in_file = spm_tpm_file

        fuzzyScore = Node(FuzzyOverlap(), name='fuzzyScore')
        wf.connect(splitTPM, ('out_files', getElementFromList, 0, 3), fuzzyScore, 'in_ref')
        wf.connect(gzipNormNativeTissue, 'out_file', fuzzyScore, 'in_tst')

        saveScore = Node(Function(input_names=['out_file', 'global_dice',
                                           'class_dice0', 'class_dice1', 'class_dice2'],
                                  output_names=['out_file'],
                                  function=saveToCsv),
                        name='saveScore')
        saveScore.inputs.out_file = 'fuzzy_overlap_scores.csv'
        wf.connect(fuzzyScore, ('dice', toList), saveScore, 'global_dice')
        wf.connect(fuzzyScore, ('class_fdi', getElementFromList, 0, 1), saveScore, 'class_dice0')
        wf.connect(fuzzyScore, ('class_fdi', getElementFromList, 1, 2), saveScore, 'class_dice1')
        wf.connect(fuzzyScore, ('class_fdi', getElementFromList, 2, 3), saveScore, 'class_dice2')
                
    # Node: wf.outputNode
    out_fields = ['SPMbrain', 'SPMbrainmask', 'SPM_tissue_maps', 'SPM_normalized_tissue_maps',
                  'SPM_inverse_deformation_field', 'SPM_forward_deformation_field']
    if qc:
        #out_fields.extend(['SPMbrainmask_plot', 'SPM_tissue_fuzzy_score',])
        out_fields.append('SPM_tissue_fuzzy_score')
    outputNode = Node(IdentityInterface(fields=out_fields),
                                        name="outputNode")
    wf.connect(applyMask, "out_file", outputNode, "SPMbrain")                                   
    wf.connect(spmBrainmask, "out_file", outputNode, "SPMbrainmask")
    wf.connect(gzipNativeTissue, "out_file", outputNode, "SPM_tissue_maps")
    wf.connect(gzipNormNativeTissue, "out_file", outputNode, "SPM_normalized_tissue_maps")
    wf.connect(gzipInvField, "out_file", outputNode, "SPM_inverse_deformation_field")
    wf.connect(gzipForField, "out_file", outputNode, "SPM_forward_deformation_field")
    if qc:
        #wf.connect(plotBrainMask, "out_file", outputNode, "SPMbrainmask_plot")
        wf.connect(saveScore, "out_file", outputNode, "SPM_tissue_fuzzy_score")
    
    return wf
             
def prepImages(name='prepImages',
               base_dir='',
               data_dir='',
               subjects=None,
               in_dat_tmp={'T1' : '%s/*%s.nii.gz'},
               in_dat_tmp_arg={'T1': [['subject_id', 'T1']]},
               im_names=['T1'],
               reorient=True,
               resample=False,
               ibrainmask=False,
               coreg_ref_space='T1'):
    '''
    WF to prepare images to apply PreprocImages WF

    Input arguments:
    - name: (str) name of the WF (default: 'prepImages')
    - base_dir: (str) working directory for the WF
    - data_dir: (str) directory containing input data with patterns specified by 
    in_dat_tmp and in_dat_tmp_arg
    - subjects: (list) list of subject ID
    - in_dat_tmp: (str or dict) nipype DataGrabber input field template, which is a dict.
    If string is provided, assumes that all keys in the in_dat_tmp_arg has the same template.
    - in_dat_tmp_arg: (dict) nipype DataGrabber input template arguments
        <Required item in the dict>
        - All the images specified in im_names
        If ibrainmask = True;
        - 'brainmask': input template for brain mask that will be processed together
                    to be ready for use in PreprocImages WF
        If coreg_ref_space is not None;
        - '{coreg_ref_space}': input template for image specified as coregistration reference space 
    - im_names: (list of string) names of images to be processed
    - reorient: (bool) whether to reorient the input images (default: True)
    - resample: (bool) whether to resample the input images to 1x1x1 mm. If False, assumes the
    images are already roughly in that resolution (default: False)
    - ibrainmask: (bool) whether to compute crop coordinates using a brain mask (default: False)
    - coreg_ref_space: (str or None) if provided images need to be coregistered to one reference space,
    name of the reference image
    
    '''

    wf = Workflow(name)
    wf.base_dir = base_dir

    # Make sure im_names match keys in in_dat_tmp_arg
    for im in im_names:
        if im not in in_dat_tmp_arg.keys():
            raise ValueError('Please provide input template for {}'.format(im))

    # If ibrainmask is True, make sure brainmask is provided
    if ibrainmask:
        if 'brainmask' not in in_dat_tmp_arg.keys():
            raise ValueError('Please provide input template for brainmask if processing it')
    # If coregistering to a ref space, make sure that reference space image is provided
    if coreg_ref_space is not None:
        if coreg_ref_space not in in_dat_tmp_arg.keys():
            raise ValueError('Please provide input template for reference space image')
        if im_names.index(coreg_ref_space) != 0:
            # reorder so that ref space is the first on the list
            im_names = [coreg_ref_space] + [im for im in im_names if im != coreg_ref_space] 
    all_im_names = im_names + ['brainmask'] if ibrainmask else im_names.copy()

    # Iterate over the list of subject names
    infosource = Node(IdentityInterface(fields=['subject_id'], mandatory_inputs=True),
                      name="infosource")
    infosource.iterables = [('subject_id', subjects)]
    
    # Grab input data
    # If in_dat_tmp is string, replace it with dictionary
    if isinstance(in_dat_tmp, str):
        field_tmp = {k: in_dat_tmp for k in in_dat_tmp_arg.keys()}
        in_dat_tmp = field_tmp

    scanList = Node(DataGrabber(infields=['subject_id'],
                                outfields=[*in_dat_tmp_arg]),
                    name="scanList")
    scanList.inputs.base_directory = data_dir
    scanList.inputs.raise_on_empty = True
    scanList.inputs.sort_filelist = True
    scanList.inputs.template = '*'
    scanList.inputs.field_template = in_dat_tmp
    scanList.inputs.template_args = in_dat_tmp_arg
    wf.connect(infosource, "subject_id", scanList, "subject_id")

    imList = Node(Function(input_names=['item{}'.format(i) for i in range(len(all_im_names))],
                           output_names=['out_list'],
                           function=createListofItems),
                  name="imList")
    for i, im_name in enumerate(all_im_names):
        wf.connect(scanList, im_name, imList, "item{}".format(i))

    # 1) Put the brain in the RAS/LAS convention.
    if reorient:
        # FSL reorient does not output nii.gz images if input is nii, so add a dummy node just to 
        # make sure the input to reorient is nii.gz (will not change input if already nii.gz format)
        GZinImages = MapNode(BinaryMaths(), name="GZinImages", iterfield=["in_file"])
        GZinImages.inputs.operand_value = 0
        GZinImages.inputs.operation = 'add'
        GZinImages.inputs.output_type = 'NIFTI_GZ'
        wf.connect(imList, "out_list", GZinImages, "in_file")

        reor_imnames = ["{}_reoriented.nii.gz".format(im) for im in all_im_names]
        reorImages = MapNode(Reorient2Std(),
                             name="reorImages",
                             iterfield=["in_file", "out_file"])
        reorImages.inputs.out_file = reor_imnames
        wf.connect(GZinImages, "out_file", reorImages, "in_file")

    # 2) Resample image to 1x1x1 isotropic if needed. Note that as long as ref space image
    # is resampled, other images will be resampled when coregistering. Resample all image
    # only if no coregistration is performed (coreg_ref_space=None).
    if resample:
        # Resample all images, but only ref image and brainmask will be used if coregistering;
        # Other images will be resampled when coregistering to the resampled ref. (i.e. non-resampled
        # images will be used when coregistering)
        resampImages = MapNode(FLIRT(),
                               name="resampImages",
                               iterfield=["reference", "in_file", "out_file", "out_matrix_file"])
        resampImages.inputs.apply_isoxfm = 1.0
        resampImages.inputs.no_search = True
        resampImages.inputs.no_resample_blur = True
        resampImages.inputs.output_type = "NIFTI_GZ"
        resampImages.inputs.out_file = ["{}_111.nii.gz".format(im) for im in all_im_names]
        resampImages.inputs.out_matrix_file =  ["{}_111.mat".format(im) for im in all_im_names]
        if reorient:
            wf.connect(reorImages, "out_file", resampImages, "in_file")
            wf.connect(reorImages, "out_file", resampImages, "reference")
        else:
            wf.connect(imList, "out_file", resampImages, "in_file")
            wf.connect(imList, "out_file", resampImages, "reference")

        # If brainmask is resampled, binarize it 
        if ibrainmask:
            binBrainmask = Node(Threshold(), name="bin")
            binBrainmask.inputs.thresh = 0.5
            binBrainmask.inputs.args = '-bin'
            binBrainmask.out_file = "brainmask_111_bin.nii.gz"
            wf.connect(resampImages, ("out_file", getElementFromList, -1), binBrainmask, "in_file")

    # 3) Coregister images to ref space (default is T1) unless coreg_ref_space is None.
    if coreg_ref_space is not None:
        coreg = coregImages(name="coregImages",
                            ref_imname=coreg_ref_space,
                            coreg_imnames=im_names[1:],
                            supply_mat=False)
        if resample:
            wf.connect(resampImages, ("out_file", getElementFromList, 0),
                       coreg, "inputNode.ref_image")
        elif reorient:
            wf.connect(reorImages, ("out_file", getElementFromList, 0),
                       coreg, "inputNode.ref_image")
        else:
            wf.connect(scanList, coreg_ref_space, coreg, "inputNode.ref_image")
        slc = -1 if ibrainmask else 0
        if reorient:
            wf.connect(reorImages, ("out_file", getElementFromList, 1, slc),
                       coreg, "inputNode.coreg_images")
        else:
            wf.connect(imList, ("out_list", getElementFromList, 1, slc),
                       coreg, "inputNode.coreg_images")

    # We put the output of interest in a Sink folder.
    sink = Node(DataSink(), name='Sink')
    sink.inputs.base_directory = base_dir
    sink.inputs.container = '{}Sink'.format(name)
    sink.inputs.parameterization = True
    if coreg_ref_space is not None:
        wf.connect(coreg, 'outputNode.coregistered_images', sink, 'coregistered_images')
        wf.connect(coreg, 'outputNode.coreg_mat', sink, 'coreg_matrix_file')
        wf.connect(coreg, 'outputNode.inv_coreg_mat', sink, 'coreg_inverse_matrix_file')
    if resample:
        if ibrainmask:
            wf.connect(resampImages, ('out_file', getElementFromList, 0, -1), sink, 'resampled_images')
            wf.connect(binBrainmask, 'out_file', sink, 'resampled_brainmask')
        else:
            wf.connect(resampImages, 'out_file', sink, 'resampled_images')
        # keep resample matrix
        wf.connect(resampImages, 'out_matrix_file', sink, 'resample_matrix_file')
    if reorient:
        wf.connect(reorImages, 'out_file', sink, 'reoriented_images')

    return wf

def getBrainmask(name='getBrainmask',
                 base_dir='',
                 data_dir='',
                 subjects=None,
                 in_dat_tmp={'ref_main': '%s/*%s.nii.gz'},
                 in_dat_tmp_arg={'ref_main': [['subject_id', 'T1']]},
                 method='fs',
                 method_options={'supply_fsbrainmask': False,
                                 'do_recon1': True,
                                 'fs_subjects_dir': '',
                                 'fs_3T': True},
                 qc=True):
    '''
    WF to generate a brainmask in the same space as 'ref_main'.

    Input arguments:
    - name: (str) name of the WF (default: 'getBrainmask')
    - base_dir: (str) working directory for the WF
    - data_dir: (str) directory containing input data with patterns specified by 
    in_dat_tmp and in_dat_tmp_arg
    - subjects: (list) list of subject ID
    - in_dat_tmp: (str or dict) nipype DataGrabber input field template, which is a dict.
    If string is provided, assumes that all keys in the in_dat_tmp_arg has the same template.
    - in_dat_tmp_arg: (dict) nipype DataGrabber input template arguments
        <Required item in the dict>
        - 'ref_main': specifies the path to reference image (usually T1) to generate the brain mask
        If method = 'fs' and performing autorecon1;
        - 'orig_main': specifies the path to image (usually T1) to perform auto-recon1 with
        If method = 'fs' and providing brainmask.mgz;
        - 'fsbrainmask': specifies the path to freesurfer-generated brainmask.mgz
    - method: (one of 'fs', 'bet', 'spm') method for obtaining brain mask
    - method_options: (dict) parameters for brain mask generation in each method
        <Required item in the dict>
        If method = 'fs' (Freesurfer);
        - 'supply_fsbrainmask': (bool) whether to pass brainmask.mgz as input rather than performing recon-all1 or
          using the Freesurfer SUBJECTS_DIR structure to grab brainmask.mgz
        - 'do_recon1': (bool) if not supplying the brainmask, whether to perform autorecon1 or simply
          use the precomputed brainmask in Freesurfer SUBJECTS_DIR
        - 'fs_subjects_dir': (str) the Freesurfer SUBJECTS_DIR. If supply_brainmask, this simply has to be 
          existing folder, and it won't be used.
        - 'fs_3T': (bool) it may not change the results, but add '-3T' flag when performing autorecon1.
        If method = 'bet' (FSL BET);
        No required method options, but any input parameters for nipype BET can be specified.
        If method = 'spm' (SPM Segment): 
        No required method options, 'spm_standalone' and 'mcr' that specifies paths to standalone spm and mcr
        can be provided if using spm with mcr.
        Also any input parameters for SPM NewSegment (Segment in SPM12) can be specified.
    - qc: (bool) saves brainmask plot and for SPM, fuzzy overlap score with tissue prior.
    '''

    wf = Workflow(name)
    wf.base_dir = base_dir

    # Make sure ref space image is provided that can be used to reslice fs brainmask
    if 'ref_main' not in in_dat_tmp_arg.keys():
        raise ValueError('Please provide input template for ref_main used to reslice fs brainmask')
    
    # Check available methods are chosen
    methods = ['fs', 'bet', 'spm']
    if method not in methods:
        raise ValueError('Please choose method from the currently available options: {}'.format(methods))
    # if using freesurfer and supplying fsbrainmask, make sure it is provided
    if method == 'fs':
        if method_options['supply_fsbrainmask']:
            if 'fsbrainmask' not in in_dat_tmp_arg.keys():
                raise ValueError('Please provide input template for fsbrainmask')
    # if performing recon1, provide orig_T1
        elif method_options['do_recon1']:
            if 'orig_T1' not in in_dat_tmp_arg.keys():
                raise ValueError('Please provide input template for orig_T1 used for recon1')

    # Iterate over the list of subject names
    infosource = Node(IdentityInterface(fields=['subject_id'], mandatory_inputs=True),
                      name="infosource")
    infosource.iterables = [('subject_id', subjects)]

    # Grab input data
    # If in_dat_tmp is string, replace it with dictionary
    if isinstance(in_dat_tmp, str):
        field_tmp = {k: in_dat_tmp for k in in_dat_tmp_arg.keys()}
        in_dat_tmp = field_tmp

    scanList = Node(DataGrabber(infields=['subject_id'],
                                outfields=[*in_dat_tmp_arg]),
                    name="scanList")
    scanList.inputs.base_directory = data_dir
    scanList.inputs.raise_on_empty = True
    scanList.inputs.sort_filelist = True
    scanList.inputs.template = '*'
    scanList.inputs.field_template = in_dat_tmp
    scanList.inputs.template_args = in_dat_tmp_arg
    wf.connect(infosource, "subject_id", scanList, "subject_id")

    # SPM subWF gunzips file first thing so make sure input is zipped. This node
    # will not change input if already nii.gz format)
    GZinImages = Node(BinaryMaths(), name="GZinImages")
    GZinImages.inputs.operand_value = 0
    GZinImages.inputs.operation = 'add'
    GZinImages.inputs.output_type = 'NIFTI_GZ'
    wf.connect(scanList, 'ref_main', GZinImages, "in_file")

    # if method='spm', perform SPM NewSegment (> spm8)
    if method == 'spm':
        if 'spm_standalone' in method_options.keys():
            spm_standalone = method_options.pop('spm_standalone')
            if not 'mcr' in method_options.keys():
                raise ValueError('Please provide mcr if using spm standalone')
            mcr = method_options.pop('mcr')
        else:
            spm_standalone, mcr = "", ""
        
        spmBrainmask = prepSpmBrainmask(spm_standalone=spm_standalone,
                                        mcr=mcr,
                                        spm_seg_options=method_options,
                                        spm_tpm_file=op.join(method_options['paths'], "tpm", "TPM.nii"),
                                        qc=qc)
        wf.connect(GZinImages, "out_file", spmBrainmask, 'inputNode.ref_main')
        
    # Put the processed brainmask in sink
    sink = Node(DataSink(), name='Sink')
    sink.inputs.base_directory = base_dir
    sink.inputs.container = '{}Sink'.format(name)
    sink.inputs.parameterization = True

    if method == 'spm':
        wf.connect(spmBrainmask, 'outputNode.SPMbrain', sink, 'spm_brain')
        wf.connect(spmBrainmask, 'outputNode.SPMbrainmask', sink, 'spm_brainmask')
        wf.connect(spmBrainmask, 'outputNode.SPM_tissue_maps', sink, 'spm_tissue_maps')
        wf.connect(spmBrainmask, 'outputNode.SPM_normalized_tissue_maps', sink, 'spm_normalized_tissue_maps')
        wf.connect(spmBrainmask, 'outputNode.SPM_inverse_deformation_field', sink, 'spm_inverse_deformation_field')
        wf.connect(spmBrainmask, 'outputNode.SPM_forward_deformation_field', sink, 'spm_forward_deformation_field')
       
        if qc:
            wf.connect(spmBrainmask, 'outputNode.SPM_tissue_fuzzy_score', sink, 'spm_tissue_fuzzy_score')

    return wf

def cropImages(name="cropImages",
               use_crop_info=True,
               apply_to_fnames=[],
               crop_dims=None,
               getCrop_sh_version=False):

    cropWF = Workflow(name)
    
    #inputNode
    in_fields = ['apply_to_images']
    if use_crop_info:
        in_fields.append('crop_info_txt')
    else:
        in_fields.append('brainmask')
    inputNode = Node(IdentityInterface(fields=in_fields), name='inputNode')

    if use_crop_info:
        # Get dimension from crop_info_txt
        XYZvalues = Node(Function(input_names=["crop_info_txt"],
                                  output_names=["dims"],
                                  function=readCropInfo),
                         name="XYZvalues")
        cropWF.connect(inputNode, "crop_info_txt", XYZvalues, "crop_info_txt")

    else:
        # We get the gravity center of the brain using the binarized brainmask.
        getBBcenter = Node(ImageStats(op_string= '-C'), name="getBBcenter")
        cropWF.connect(inputNode, 'brainmask', getBBcenter, 'in_file')
    
        # We calculate where we need to crop the image using the dimensions it is supposed to make (crop_dim) 
        # and the gravity center. We make sure that gravity center and center of the bounding box do match.
        XYZvalues = Node(Function(input_names=["center", "crop_dimensions", "sh_version"],
                                  output_names=["dims", "out_file"],
                                  function=getCropInfo),
                         name="XYZvalues")
        XYZvalues.inputs.crop_dimensions = crop_dims
        XYZvalues.inputs.sh_version = getCrop_sh_version
        cropWF.connect(getBBcenter, "out_stat", XYZvalues, "center")

    # Perform crop on the apply_to_images
    cropImages = MapNode(ExtractROI(),
                         name="cropImages",
                         iterfield=["in_file", "roi_file"])
    cropImages.inputs.x_size = crop_dims[0]
    cropImages.inputs.y_size = crop_dims[1]
    cropImages.inputs.z_size = crop_dims[2]
    cropImages.inputs.output_type = "NIFTI_GZ"
    cropImages.inputs.roi_file = apply_to_fnames
    cropWF.connect(inputNode, "apply_to_images", cropImages, "in_file")
    cropWF.connect(XYZvalues, ('dims', getElementFromList, 0), cropImages, "x_min")
    cropWF.connect(XYZvalues, ('dims', getElementFromList, 1), cropImages, "y_min")
    cropWF.connect(XYZvalues, ('dims', getElementFromList, 2), cropImages, "z_min")

    # Node: fsconv.outputNode
    outputNode = Node(IdentityInterface(fields=['cropped_images', 'crop_info']),
                                        name="outputNode")
    cropWF.connect(cropImages, "roi_file", outputNode, "cropped_images")
    if use_crop_info:
        cropWF.connect(inputNode, 'crop_info_txt', outputNode, 'crop_info')
    else:
        cropWF.connect(XYZvalues, 'out_file', outputNode, 'crop_info')

    return cropWF
                        
def PreprocImages(name='PreprocImages',
                  SVN_dir='',
                  path_to_spm ='',
                  base_dir='',
                  data_dir='',
                  subjects=None,
                  in_dat_tmp={'T1': '%s/*%s.nii.gz'},
                  in_dat_tmp_arg={'T1': [['subject_id', 'T1']]},
                  im_names=['T1'],
                  crop_dims= [160, 214, 176] ,
                  use_crop_info=False,
                  getCrop_sh_version=False):
    '''
    WF to crop and intensity-normalize images

    Input arguments:
    - name: (str) name of the WF (default: 'PreprocImages')
    - SVN_dir: (str) path to the SVN repository
    - base_dir: (str) working directory for the WF
    - data_dir: (str) directory containing input data with patterns specified by 
    in_dat_tmp and in_dat_tmp_arg
    - subjects: (list) list of subject ID
    - in_dat_tmp: (str or dict) nipype DataGrabber input field template, which is a dict.
    If string is provided, assumes that all keys in the in_dat_tmp_arg has the same template.
    - in_dat_tmp_arg: (dict) nipype DataGrabber input template arguments
        <Required item in the dict>
        - All the images specified in im_names
        If use_crop_info = False;
        - 'brainmask': input template for brain mask that will be will be used to compute crop info
        If use_crop_info = True;
        - 'crop_info': input template for crop_info.txt that will be used to crop images
    - im_names: (list of string) names of images to be processed
    - crop_dims: (list or tuple of x, y, z) dimensions of cropped image
    - use_crop_info: (bool) whether to use pre-computed crop coordinates by providing crop_info (default: False)
    - getCrop_sh_version: (bool) if computing crop coordinates using a brain mask, whether to replicate
    shell-version of the earlier processing pipeline that used slightly different calculation of crop coordinates
    (default: False)
    
    '''

    wf = Workflow(name)
    wf.base_dir = base_dir

    # Make sure im_names match keys in in_dat_tmp_arg
    for im in im_names:
        if im not in in_dat_tmp_arg.keys():
            raise ValueError('Please provide input template for {}'.format(im))
    # Make sure brainmask is provided
    if 'brainmask' not in in_dat_tmp_arg.keys():
        raise ValueError('Please provide input template for brainmask')
   
    all_im_names = im_names + ['brainmask']
    # If using crop info rather than brainmask to compute crop, make sure crop info is provided
    if use_crop_info:
        if 'crop_info' not in in_dat_tmp_arg.keys():
            raise ValueError('Please provide input template for crop info if not computing it from brainmask')
    

    # Iterate over the list of subject names
    infosource = Node(IdentityInterface(fields=['subject_id'], mandatory_inputs=True),
                      name="infosource")
    infosource.iterables = [('subject_id', subjects)]

    # Grab input data
    # If in_dat_tmp is string, replace it with dictionary
    if isinstance(in_dat_tmp, str):
        field_tmp = {k: in_dat_tmp for k in in_dat_tmp_arg.keys()}
        in_dat_tmp = field_tmp

    scanList = Node(DataGrabber(infields=['subject_id'],
                                outfields=[*in_dat_tmp_arg]),
                    name="scanList")
    scanList.inputs.base_directory = data_dir
    scanList.inputs.raise_on_empty = True
    scanList.inputs.sort_filelist = True
    scanList.inputs.template = '*'
    scanList.inputs.field_template = in_dat_tmp
    scanList.inputs.template_args = in_dat_tmp_arg
    wf.connect(infosource, "subject_id", scanList, "subject_id")

    imList = Node(Function(input_names=['item{}'.format(i) for i in range(len(all_im_names))],
                           output_names=['out_list'],
                           function=createListofItems),
                  name="imList")
    for i, im_name in enumerate(all_im_names):
        wf.connect(scanList, im_name, imList, "item{}".format(i))

    # 1) Crop the coregistered images
    # Prepare file list for crop!
    crop_fnames = ["{}_cropped.nii.gz".format(im) for im in all_im_names]
    cropping = cropImages(use_crop_info=use_crop_info,
                          apply_to_fnames=crop_fnames,
                          crop_dims=crop_dims,
                          getCrop_sh_version=getCrop_sh_version)
    # if not using crop info, we only need to supply image list to be cropped with brainmask
    wf.connect(imList, "out_list", cropping, "inputNode.apply_to_images")
    # if using crop_info, supply this, and if not, supply brainmask
    if use_crop_info:
        wf.connect(scanList, "crop_info", cropping, "inputNode.crop_info_txt")
    else:
        wf.connect(scanList, "brainmask", cropping, "inputNode.brainmask")

    # 2) Intensity-normalize inside the brain using the cropped images
    intNorm = intNormImages(base_dir=op.join(wf.base_dir, wf.name),
                            SVN_dir=SVN_dir,
                            path_to_spm = path_to_spm,
                            im_names=im_names,
                            unzip=True)
    wf.connect(cropping, ("outputNode.cropped_images", getElementFromList, 0, -1), intNorm, "inputNode.apply_to_images")
    wf.connect(cropping, ("outputNode.cropped_images", getElementFromList, -1), intNorm, "inputNode.brainmask")
    wf.connect(infosource,"subject_id", intNorm, "inputNode.subject_id")
   
    # We put the output of interest in a Sink folder.
    sink = Node(DataSink(), name='Sink')
    sink.inputs.base_directory = base_dir
    sink.inputs.container = '{}Sink'.format(name)
    sink.inputs.parameterization = True
    wf.connect(cropping, 'outputNode.cropped_images', sink, 'cropped_images')
    wf.connect(cropping, 'outputNode.crop_info', sink, 'crop_info')
    wf.connect(intNorm, 'outputNode.intensity_normed_images', sink, 'cropped_intensity_normed_images')
          
    return wf
