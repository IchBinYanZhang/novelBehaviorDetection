clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../aca'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));
dataset_path = '/mnt/hdd/Dataset_MPIIHDM05';
addpath(genpath('HDM05-Parser/parser'));
addpath(genpath('HDM05-Parser/animate'));
addpath(genpath('HDM05-Parser/quaternions'));


dataset_info = split(importdata([dataset_path '/annotation.txt']));
video_list = unique(dataset_info(:,1));

%%% scenario configuration - before each running, check here!"
feature_list = {'jointLocs','relativeAngle','quaternion'};


is_show = 0; 
%%% scenario configuration - end"
for ff = 1:length(feature_list)
    feature = feature_list{ff};
    vid = 0;
for vv = 1:length(video_list)    
    
    % only consider scenarios 03
    if ~contains(video_list{vv},'_03-')
        continue;
    end
    vid = vid+1; 
    %------------------ fearture extraction and annotation----------------%
    fprintf('------- processing: %s\n', video_list{vv});
    %%% find annotation
    bds = dataset_info(find( strcmp(dataset_info(:,1),video_list{vv})), 2:3 );
    ytt = unique(sort(reshape(cellfun(@str2num,bds),[],1)));
    
    %%% parse the skeleton
    [skel, mot] = readMocap([dataset_path '/' video_list{vv} '.c3d'],[],false);
    x = (cell2mat(mot.jointTrajectories))';
    yt = zeros(size(x,1),1);
    yt(ytt) = 100;
    if strcmp(feature, 'jointLocs')
        pattern = x;
    else
        n_joints = size(x,2)/3;
        xt = x;
        rat = [];
        qut = [];
   
        for j = 1:n_joints-2
            p1 = xt(:,3*j-2: 3*j);
            p2 = xt(:,3*(j+1)-2: 3*(j+1));
            p3 = xt(:,3*(j+2)-2: 3*(j+2));
            av = p2-p1;
            bv = p3-p2;
            a = av./repmat(sqrt(sum(av.^2,2))+1e-6,1,3);
            b = bv./repmat(sqrt(sum(bv.^2,2))+1e-6,1,3);

            %%% compute relative angles
            c = diag(a*b');
            rat = [rat acos(c)];

            %%% compute quaternions
            yaw = acos((a(:,1).*b(:,1)+a(:,2).*b(:,2))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,2).^2).*sqrt(b(:,1).^2+b(:,2).^2) ));
            pitch = acos((a(:,1).*b(:,1)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,3).^2).*sqrt(b(:,1).^2+b(:,3).^2) ));
            roll = acos((a(:,2).*b(:,2)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,2).^2+a(:,3).^2).*sqrt(b(:,2).^2+b(:,3).^2) ));
            qut = [qut angle2quat(yaw, pitch,roll)];
        end

        if strcmp(feature,'relativeAngle')
            pattern = rat;
        elseif strcmp(feature,'quaternion')
            pattern = qut;
        end
    end

    X = pattern;
    outfile = sprintf([dataset_path '/features/HDM05_%s_v%d.mat'], feature,vid );
    save(outfile,'X');
        
    end
end



    
    
