function convert_data()
	source_root_dir = '.';
	target_root_dir = 'viton_resize';
	flag_resize = true;
	fine_height = 256;
	fine_width = 192;

	% transfer data root
	if ~exist(target_root_dir,'dir');
		mkdir(target_root_dir);
	end

	modes = {'train', 'test'};
	for i = 1:length(modes);
		fprintf('Start convert %s\n', modes{i});
		convert(source_root_dir, target_root_dir, modes{i}, flag_resize, fine_height, fine_width);
	end
end

function convert(source_root_dir, target_root_dir, mode, flag_resize, fine_height, fine_width)
	cmap = importdata('human_colormap.mat');
	point_num = 18;
	
	% make new dir
	dir_list = {'cloth', 'cloth-mask', 'image', 'image-parse', 'pose'};
	if ~exist([target_root_dir '/' mode],'dir');
		mkdir([target_root_dir '/' mode]);
	end
	for i = 1:length(dir_list);
		if ~exist([target_root_dir '/' mode '/' dir_list{i}],'dir');
			mkdir([target_root_dir '/' mode '/' dir_list{i}]);
		end
	end

	% read train pairs
	[im_names, cloth_names] = textread(['viton_' mode '_pairs.txt'],'%s %s\n');
	N = length(im_names);
	for i = 1:N;
		imname = im_names{i} ;
		cname = cloth_names{i};
		fprintf('%d/%d: %s %s\n', i, N, imname, cname);
	   
		% generate cloth mask
		im_c = imread([source_root_dir '/' 'women_top/' cname]);
		
		% generate parsing result
		im = imread([source_root_dir '/' 'women_top/' imname]);
		h = size(im,1);
		w = size(im,2);
		s_name = strrep(imname,'.jpg','.mat');
		segment = importdata([source_root_dir '/' 'segment/' s_name]);
		segment = segment';
	
	    if h > w
	        segment = segment(:,1:int32(641.0*w/h));
	    else
	        segment = segment(1:int32(641.8*h/w),:);
	    end
	    segment = imresize(segment, [h,w], 'nearest');
	

	    % load pose
	    pose = importdata([source_root_dir '/' 'pose/' s_name]);
	    key_points = zeros(point_num,3);
	    for j = 1:point_num
	        index = int32(pose.subset(j))+1;
	        if index ~= 0
	            key_points(j,:) = pose.candidate(index,1:3);
	        end       
	    end 

		% save cloth & image, resize the results
		if flag_resize;
			im_c = imresize(im_c, [fine_height, fine_width], 'bilinear');
			imwrite(im_c, [target_root_dir '/' mode '/cloth/' cname]);

			im = imresize(im, [fine_height, fine_width], 'bilinear');
			imwrite(im, [target_root_dir '/' mode '/image/' imname]);
			
			segment = imresize(segment, [fine_height, fine_width], 'nearest');

			for j = 1:point_num
				key_points(j,1) = key_points(j,1) / w * fine_width;
				key_points(j,2) = key_points(j,2) / h * fine_height;
			end
		else
			copyfile([source_root_dir '/' 'women_top/' cname], ...
				[target_root_dir '/' mode '/cloth/' cname]);

			copyfile([source_root_dir '/' 'women_top/' imname] , ...
				[target_root_dir '/' mode '/image/' imname]);
		end

		% save cloth mask
		mask = double((im_c(:,:,1) <= 250) & (im_c(:,:,2) <= 250) & (im_c(:,:,3) <= 250));
		mask = imfill(mask);
		mask = medfilt2(mask);
		imwrite(mask, [target_root_dir '/' mode '/cloth-mask/' cname]);
	   
		% save parsing result
	    segment = uint8(segment);
	    pname = strrep(imname, '.jpg', '.png');
	    imwrite(segment,cmap,[target_root_dir '/' mode '/image-parse/' pname]);
	    
	    % save the pose info
	    key_name = strrep(imname, '.jpg', '_keypoints.json');
	    f = fopen([target_root_dir '/' mode '/pose/' key_name], 'w');
	    fprintf(f,'{"version": 1.0, "people": [{"face_keypoints": [], "pose_keypoints": ');
	    
	    key_points = reshape(key_points', 1, 54);
	    str_key_points = mat2str(key_points);
	    str_key_points = strrep(str_key_points,' ', ', ');
	    fprintf(f,str_key_points);
	    fprintf(f,', "hand_right_keypoints": [], "hand_left_keypoints": []}]} ');  
	    fclose(f);
	    
	end

end

