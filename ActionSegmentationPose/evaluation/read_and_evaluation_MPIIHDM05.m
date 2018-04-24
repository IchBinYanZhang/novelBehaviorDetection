%%% Evaluation on the HDM05 dataset
%%% This script is developed for evaluating the performance of {Dirichlet
%%% process mixture model},{Dirichlet process mixture model -
%%% aggregation} and {Growing Neural gas}. Since these methods are
%%% implemented in Python, this matlab script reads the results of the
%%% Python scripts and perform evaluation.

%%% developed by Yan Zhang (yz-cnsdqz.github.io)





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
method = 'DPMM-A'; % choose from {'DPMM', 'DPMM-A'}

is_show = 0; 
%%% scenario configuration - end"

for ff = 1:length(feature_list)
    feature = feature_list{ff};
    vid = 0;
    CptTime = [];
    Pre = [];
    Rec = [];
    CMat = zeros(2,2);

for vv = 1:length(video_list)    
    
    % only consider scenarios 03
    if ~contains(video_list{vv},'_03-')
        continue;
    end
    vid = vid+1; 
    %------------------ fearture extraction and annotation----------------%
    %fprintf('------- processing: %s\n', video_list{vv});
    %%% find annotation
    bds = dataset_info(find( strcmp(dataset_info(:,1),video_list{vv})), 2:3 );
    ytt = unique(sort(reshape(cellfun(@str2num,bds),[],1)));
    n_clusters = 15;

    
    [skel, mot] = readMocap([dataset_path '/' video_list{vv} '.c3d'],[],false);
    x = (cell2mat(mot.jointTrajectories))';
    yt = zeros(size(x,1),1);
    yt(ytt) = 100;
       
    % input the DPMM result
    idx_file = sprintf([dataset_path '/DPMM_segmentation_results/HDM05_%s_v%d_DPMM_SampleLabels.txt'],feature,vid);
    idx = importdata(idx_file);
   
    if strcmp(method, 'DPMM-A')
        % aggregation and clustering
        X_file = sprintf([dataset_path '/DPMM_segmentation_results/HDM05_%s_v%d_DPMM_SampleEncoding.txt'],feature,vid);
        X = importdata(X_file);
        starttime = tic;
        idx = fun_feature_aggregation_no_encoding(X,idx, 'kmeans','HDM05',13,2); 
        eps_time = toc(starttime);
        CptTime = [CptTime eps_time];
    end
    % evaluation
    res = evaluation_metric_TUMKitchen(yt, double(idx), 15, is_show); % tol range = 15
    Pre = [Pre res.Segmentation.Pre];
    Rec = [Rec res.Segmentation.Rec];
    CMat = CMat + res.NovelBehavior.ConMat;
end
fprintf('====== method: %s======feature: %s============\n',method,feature);
fprintf('- segmentation: avg_pre = %f\n',mean(Pre));
fprintf('- segmentation: avg_rec = %f\n',mean(Rec));
fprintf('- novelBehavior: avg_pre = %f\n',CMat(1,1)/(CMat(2,1)+CMat(1,1)));
fprintf('- novelBehavior: avg_rec = %f\n',CMat(1,1)/(CMat(1,2)+CMat(1,1)));
if strcmp(method, 'DPMM-A')
    fprintf('- avg_runtime = %f seconds\n',mean(CptTime));
end
end



    
    
