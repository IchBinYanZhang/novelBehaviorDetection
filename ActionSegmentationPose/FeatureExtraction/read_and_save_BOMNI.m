clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));
dataset_path = '/home/yzhang/Videos/Dataset_BOMNI'; %% only scenario1, side view
video_list = importdata([dataset_path '/scenario1/video_list2.txt']);
action_list = {'"walking"','"sitting"', '"drinking"','"washing-hands"','"opening-closing-door"','"fainted"'};


Precision = [];
Recall = [];
CptTime = [];

   
is_show_tracking = 1;
is_saving_features = 0;
rescale_factor = 1;
roi_width = 40;
roi_height = 40;

k = 1;
for vv = 1 : 1
% for vv = 1 : length(video_list)
    video = VideoReader([dataset_path '/scenario1/' video_list{vv} '.mp4']);
    annotation = importdata([dataset_path '/annotations/scenario1/' video_list{vv} '.dat']);
    fprintf('-- processing: %s\n',video_list{vv});
    
    %%% feature extraction and tracking visualization
    pattern = [];
    yt = [];
    idx_frame = 0;
    background = [];
    idx_frame_max = length(annotation)-1;
    for idx_frame = 1:idx_frame_max
        
        img_name = [dataset_path '/scenario1/frame_and_flow/' video_list{vv} sprintf('/image_%05d.jpg', idx_frame) ];
        frame = imresize(imread(img_name), rescale_factor);
        labels = strsplit(annotation{idx_frame},' ');
        bbox = rescale_factor*[str2num(labels{2}) str2num(labels{3}) ...
            str2num(labels{4})-str2num(labels{2}) str2num(labels{5})-str2num(labels{3})];
        is_lost = str2num(labels{7});
        action = labels{end};


        if is_lost 
    %             disp('- skip this frame without person..')
            continue;
        end

        %%% read mask from mask R-CNN
        mask_name_list = dir([dataset_path '/scenario1/frame_and_flow/' video_list{vv} sprintf('/image_%05d_mask*.jpg', idx_frame) ]);
        frame_mask = zeros(size(frame,1), size(frame,2));
        overlay = 0;
        for ii = 1:length(mask_name_list)
            frame_mask_c = imread([dataset_path '/scenario1/frame_and_flow/' video_list{vv} sprintf('/image_%05d_mask_%d.jpg',idx_frame, ii-1)]);
            frame_mask_c = imresize(frame_mask_c,rescale_factor);
            frame_mask_c = rgb2gray(frame_mask_c);
            frame_mask_c = imbinarize(frame_mask_c);
            frame_mask_crop = imcrop(frame_mask_c, bbox);

            if sum(frame_mask_crop(:))/prod(bbox(3:4))>overlay
                frame_mask = frame_mask_c;
                overlay = sum(frame_mask_crop(:))/prod(bbox(3:4));
            end
        end

        if sum(frame_mask(:))==0
            continue;
        end

        %%% find numerical label
        yt = [yt; find(cellfun(@(x) strcmp(x,action), action_list, 'UniformOutput',1))];
    %       

        %%% extract pattern from frame
        %%%% extract roi of the mask. We redefine the bbox, since the
        %%%% annotation is not accurate
        [rr, cc] = find(frame_mask);
        roi(1) = min(cc);
        roi(2) = min(rr);
        roi(3) = max(cc) - min(cc);
        roi(4) = max(rr) - min(rr);
        mm = imresize(imcrop(frame_mask,roi), [roi_height, roi_width]);
        [D,~] = bwdist(mm); D = max(D(:))-D;
        pattern0 = D(:);
        pattern = [pattern; pattern0'];


        %%% visualize tracking
    %         imwrite(uint8(frame),'img.png');
        if is_show_tracking
            close all;
            B = imoverlay(frame, frame_mask);
            figure(1); imshow(uint8(B));
            hold on;
            rectangle('Position',...
                [bbox(1) bbox(2) bbox(3) bbox(4)],...
                'EdgeColor','red',...
                'LineWidth',2);
            text(bbox(1),bbox(2)-10,action,...
                'HorizontalAlignment','left',...
                'Color','red');
    %             figure(2);plot(motion);xlim([1 idx_frame_max]);
    %             figure(3);plot(shape);xlim([1 idx_frame_max]);

            
            truesize;pause(0.1);
            saveas(gcf, sprintf('FIG%d.png',k)); % will create FIG1, FIG2,...
            k = k+1;
        end
    end
    
    %%% save patterns to file
    if is_saving_features
        X = pattern;
        outfilename = sprintf([dataset_path '/scenario1/features/%s.mat'], video_list{vv});
        save(outfilename, 'X','yt');
    end
    
end




    
    
