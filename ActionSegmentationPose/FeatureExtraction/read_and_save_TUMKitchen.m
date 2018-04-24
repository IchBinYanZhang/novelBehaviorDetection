clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../aca'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));
dataset_path = '/mnt/hdd/Dataset_TUMKitchen';
video_list = importdata([dataset_path '/video_list.txt']);



%%% scenario configuration - before each running, check here!"
feature_list = {'jointLocs','relativeAngle','quaternion'};
bodypart_list = {'rightArm','leftArm','torso'};

for vv = 1:length(video_list)
    for bb = 1:length(bodypart_list)
        for ff = 1:length(feature_list)
            bodypart = bodypart_list{bb};
            feature = feature_list{ff};
            %------------------ fearture extraction and annotation----------------%
            skeleton = importdata([dataset_path '/' video_list{vv} '/poses.csv']);    
            [xt, xl, xr] = calPatternFromSkeleton(skeleton, feature);
            anno = importdata([dataset_path '/' video_list{vv} '/labels.csv']);
            [yt_left_arm, yt_right_arm, yt_torso] = readTUMKitchenAnnotation(anno);
            if strcmp(bodypart,'leftArm')
                n_clusters = 9;
                yt = yt_left_arm;
                pattern = xl;
            elseif strcmp(bodypart,'rightArm')
                n_clusters = 9;
                yt = yt_right_arm;
                pattern = xr;
            elseif strcmp(bodypart,'torso')
                n_clusters = 2;
                yt = yt_torso;
                pattern = xt;
            end

            %------------------ save features to file------------------%
            outfile = sprintf([dataset_path '/features/TUMkitchen_%s_%s_v%d.mat'],...
                               bodypart, feature, vv);
            X = pattern;
            save(outfile,'X');
        end
    
    end
end




    
    
