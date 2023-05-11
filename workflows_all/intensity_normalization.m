function intensity_normalization_v2(input_image, brain_mask, output_dir, output_image, path_to_spm)

	addpath(path_to_spm)

	fprintf('%s \n', 'About to normalize your image between 0 and 1.')

	filename = split(input_image, '/');
	filename = string(filename(length(filename)));
	filename = split(filename, '.');
	filename = string(filename(1));

	% open and read the volume
	vol0 = spm_vol(input_image);
	pipo = spm_read_vols(vol0);

	% open and read the mask
	vol1 = spm_vol(brain_mask);
	mask = spm_read_vols(vol1);

	% rescale between 0 and the 99th percentile 
    b= find(mask(:) > 0);
	tab = pipo;
	scale_factor = prctile(tab(b),99); % 99th percentile
	min_tab=0;
	max_tab=1.3; 
	tab= tab/scale_factor;
	 
	% everything below 0 equals to 0
	% and above 1.3 equals to 1.3
	tmp = find(tab < min_tab);
	tab(tmp) = min_tab;
	tmp = find(tab> max_tab);
	tab(tmp) = max_tab;

	% normalizing between [0 1]
	tab = (tab - min(tab(:))) / ( max(tab(:)) - min(tab(:)));

	% writing final normalized volume
	V3=vol0;
    V3.dt = [16 0];
	file_output = strcat(output_dir, '/', output_image);
	V3.fname = char(file_output);
	spm_write_vol(V3, tab);
    
end
