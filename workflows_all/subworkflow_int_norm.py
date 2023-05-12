import os.path as op
import numpy as np
from nipype.interfaces.base import (traits, File, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec,
                                    InputMultiPath, OutputMultiPath) 
from nipype.interfaces.matlab import MatlabCommand
from string import Template

class CustomIntensityNormalizationInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    brain_mask = File(exists=True, mandatory=True)
    out_file = File()
    svn_dir = traits.Str(mandatory=True, desc='Path to your SVN directory.')
    output_dir = traits.Str(mandatory=True, desc='Path to output directory.')
    path_to_spm = traits.Str(mandatory=True, desc='Path to SPM directory.')
    
class CustomIntensityNormalizationOutputSpec(TraitedSpec):
    out_file = File(exists=True)

class CustomIntensityNormalization(BaseInterface):
    """ Denoising algorithm for T1/T2flair images in MATLAB """

    input_spec = CustomIntensityNormalizationInputSpec
    output_spec = CustomIntensityNormalizationOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_file=self.inputs.in_file,
                 svn_dir=self.inputs.svn_dir,
                 path_to_spm = self.inputs.path_to_spm,
                 brain_mask=self.inputs.brain_mask,
                 output_dir=self.inputs.output_dir,
                 out_file=self.inputs.out_file)
        script = Template("""
                addpath('$svn_dir')
                intensity_normalization('$in_file','$brain_mask','$output_dir','$out_file', '$path_to_spm')
                exit;
                """).substitute(d)
     
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


