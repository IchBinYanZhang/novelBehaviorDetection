%%% Evaluation on the CMUMAD dataset
%%% This script is developed for evaluating the performance of {Dirichlet
%%% process mixture model},{Dirichlet process mixture model -
%%% aggregation} and {Growing Neural gas}. Since these methods are
%%% implemented in Python, this matlab script reads the results of the
%%% Python scripts and perform evaluation.

%%% developed by Yan Zhang (yz-cnsdqz.github.io)



close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../aca'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));

%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';
subject = 1:20;
pose_feature_set = {'idtfv','VGG16','jointLocs', 'relativeAngle', 'quaternions'};
time_weights = 0.3*1e-2;

is_save = 0;
is_show_result = 0;

% method_list = {'DPMM','DPMM-A','GNG'};
method_list = {'GNG'};

for mm = 1:length(method_list)
    method = method_list{mm};
    for pp = 1:length(pose_feature_set)

        cpt_time = [];
        Pre = [];
        Rec = [];
        CMat = zeros(2,2);
        pose_feature = pose_feature_set{pp};
        fprintf('================ perform ===== %s =========\n', pose_feature);

        for ss = subject


            for qq = 1:2

                gt_file = sprintf([dataset,'/sub%02d/seq%02d_label.mat'], ss,qq);
                load(gt_file);
                gtlabE =  extractGTFormat(label); 
                n_frames = label(end,3)-1;
                
                if strcmp(method, 'DPMM-A')
                    idx_file = sprintf([dataset, '/%s_segmentation_results/CMUMAD_%s_sub%02d_seq%02d_DPMM_SampleLabels.txt'],'DPMM',pose_feature,ss,qq);
                    idx = importdata(idx_file);
                else
                    idx_file = sprintf([dataset, '/%s_segmentation_results/CMUMAD_%s_sub%02d_seq%02d_%s_SampleLabels.txt'],method,pose_feature,ss,qq,method);
                    idx = importdata(idx_file);
                end
              
                if strcmp(method, 'DPMM-A')
                    pattern_file = sprintf([dataset, '/%s_segmentation_results/CMUMAD_%s_sub%02d_seq%02d_DPMM_SampleEncoding.txt'],'DPMM',pose_feature,ss,qq);
                    pattern = importdata(pattern_file);
                
                
                    tt = tic;
                    idx = fun_feature_aggregation_no_encoding(pattern,idx, 'kmeans','CMUMAD',35,1); 
                    tt = toc(tt);
                    cpt_time = [cpt_time tt];
                end
                res = evaluation_metric_CMUMAD(gtlabE, double(idx), 0.5, is_show_result); % tol range = 15
                Pre = [Pre res.Segmentation.Pre];
                Rec = [Rec res.Segmentation.Rec];
                CMat = CMat + res.NovelBehavior.ConMat;
            end            
        end

        fprintf('====== method: %s==================\n',method);
        fprintf('- segmentation: avg_pre = %f\n',mean(Pre));
        fprintf('- segmentation: avg_rec = %f\n',mean(Rec));
        fprintf('- novelBehavior: avg_pre = %f\n',CMat(1,1)/(CMat(2,1)+CMat(1,1)));
        fprintf('- novelBehavior: avg_rec = %f\n',CMat(1,1)/(CMat(1,2)+CMat(1,1)));
        fprintf('- avg_runtime = %f seconds\n',mean(CptTime));

        if is_save
            result_filename = sprintf('CMUMAD_Result_%s_%s.mat',pose_feature,method);
            save(result_filename, 'Pre','Rec','CMat','CptTime');
        end

    end
end
