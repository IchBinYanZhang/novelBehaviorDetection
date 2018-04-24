clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../aca'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));
%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';

subject = 1:20;
feature = 'BodyPose';

pose_feature_set = {'jointLocs', 'relativeAngle', 'quaternions'};
time_weights = 0.3*1e-2;

for pp = 1:length(pose_feature_set)

    pose_feature = pose_feature_set{pp};

    for ss = subject

        for qq = 1:2
            feature_file = sprintf([dataset, '/',feature,'/PoseFeature_sub%02d_seq%02d.mat'], ss,qq);
            data = load(feature_file);
            gt_file = sprintf([dataset,'/sub%02d/seq%02d_label.mat'], ss,qq);
            load(gt_file);
            gtlabE =  extractGTFormat(label);
            n_frames = label(end,3)-1;

            if strcmp(pose_feature, 'jointLocs')
                pattern = data.pattern.jointLocs;
            elseif strcmp(pose_feature,'relativeAngle')
                pattern = data.pattern.relativeAngle;
            elseif strcmp(pose_feature, 'quaternions')
                pattern = data.pattern.quaternions;
            end
            X = pattern;
            outfile = sprintf('CMUMAD_%s_sub%02d_seq%02d.mat', pose_feature, ss,qq);
            save(outfile, 'X');

        end

    end

end
