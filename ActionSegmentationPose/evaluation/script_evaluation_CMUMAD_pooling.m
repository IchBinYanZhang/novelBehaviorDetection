%%% (1) The script is specific to investigate functions of temporal pooling.
%%% (2) Script for CMUMAD

clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));

%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';
subject = 1:20;
feature = 'BodyPose';
% method_set = { 'ours'};
pose_feature_set = {'quaternions'};
method_set = {'ktc-s'};
% method_set = {'ours','sliding_window','ktc-s'};
% pose_feature_set = {'jointLocs', 'relativeAngle', 'quaternions'};


for mm = 1:length(method_set)
    for pp = 1:length(pose_feature_set)
        
        method = method_set{mm};
        pose_feature = pose_feature_set{pp};
        fprintf('================ perform %s ===== %s =========\n', method, pose_feature);

        Pre = [];
        Rec = [];
        CMat = zeros(2,2);
        CptTime = [];
        
        for ss = 1:20
            for qq = 1:2
                
                fprintf('-- processing: %d,%d\n',ss,qq);
                feature_file = sprintf([dataset, '/',feature,'/PoseFeature_sub%02d_seq%02d.mat'], ss,qq);
                data = load(feature_file);
                gt_file = sprintf([dataset,'/sub%02d/seq%02d_label.mat'], ss,qq);
                load(gt_file); %  subject 1, sequence   
                gtlabE =  extractGTFormat(label); % true event-based labels (obtain by loading a true label file)

                if strcmp(pose_feature, 'jointLocs')
                    pattern = data.pattern.jointLocs;                    
                elseif strcmp(pose_feature,'relativeAngle')
                    pattern = data.pattern.relativeAngle;
                elseif strcmp(pose_feature, 'quaternions')
                    pattern = data.pattern.quaternions;
                end

                n_clusters = 36;
                
                %%%% start dynamic clustering
                startTime = tic;
                time_window = 30;
                sigma = 0.09;
                dist_type = 0; % Euclidean distance
                verbose = 0;
                disp('--- dynamic clustering on input features');

                [idx1, c_locs] = incrementalClustering(double(pattern), time_window,sigma,0);
                if strcmp(method,'ours')        
                    idx = fun_feature_aggregation(pattern,idx1,c_locs,'ours','CMUMAD');
                elseif strcmp(method,'sliding_window')
                    idx = fun_feature_aggregation_slidingwindow(pattern,c_locs,0.8e-5);
                elseif strcmp(method,'ktc-s')
                    idx = fun_feature_aggregation_kernelizedcut(pattern,c_locs,8e-6,'batch');
                end


                CptTime = [CptTime toc(startTime)];
                res = evaluation_metric_CMUMAD(gtlabE, double(idx'), 0.5, false); % tol range = 0.5 overlaping
                
                Pre = [Pre res.Segmentation.Pre];
                Rec = [Rec res.Segmentation.Rec];
                CMat = CMat + res.NovelBehavior.ConMat;               
            end            
        end

        fprintf('- segmentation: avg_pre = %f\n',mean(Pre));
        fprintf('- segmentation: avg_rec = %f\n',mean(Rec));
        fprintf('- novelBehavior: avg_pre = %f\n',CMat(1,1)/(CMat(2,1)+CMat(1,1)));
        fprintf('- novelBehavior: avg_rec = %f\n',CMat(1,1)/(CMat(1,2)+CMat(1,1)));
        fprintf('- avg_runtime = %f seconds\n',mean(CptTime));
       
    end
end
