clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../TSC'));
addpath(genpath('../../aca'));
addpath(genpath('../../mexIncrementalClustering'));
dataset_path = '/home/yzhang/Videos/Dataset_BOMNI'; %% only scenario1, side view
video_list = importdata([dataset_path '/scenario1/video_list2.txt']);
action_list = {'"walking"','"sitting"', '"drinking"','"washing-hands"','"opening-closing-door"','"fainted"'};


Pre = [];
Rec = [];
CMat = zeros(2,2);
CptTime = [];
CMat_fainted = zeros(2,2);

method_list = {'ours'};   
is_show_result = 0;
is_save = 0;

for mm = 1:length(method_list)
    method = method_list{mm};
    for vv = 1 : length(video_list)
%     for vv = 3:3

        %%% read features from file
        obj = load( sprintf([dataset_path, '/scenario1/features/%s.mat'], video_list{vv})  );
        pattern = double(obj.X);
        yt = obj.yt;

        %%% evaluation using different methods
        n_clusters = length(unique(yt));
        startTime = tic;
        if strcmp(method, 'kmeans')
            [idx,C] = kmeans(pattern,n_clusters);
            uid = idx(1);
            idx(idx==uid) = 10e6;
            idx(idx==1) = uid;
            idx(idx==10e6) = 1;
            idx = idx-1; %%% 0-based label
            idx = calLocalFeatureAggregationAndClustering(pattern,idx,C, n_clusters); 


        elseif strcmp(method, 'regularSampling')
                %%% regular sampling
            prd_boundary = 1:30:size(pattern,1);
            idx = zeros(size(pattern,1),1);
            idx(prd_boundary) = 1;

        elseif strcmp(method, 'spectralClustering')
            idx = spectralClustering(pattern,500,n_clusters);
            uid = idx(1);
            idx(idx==uid) = 10e6;
            idx(idx==1) = uid;
            idx(idx==10e6) = 1;
            idx = idx-1; %%% 0-based label



        elseif strcmp(method, 'TSC')
            %%%---Parameter settings---%%%
            paras = [];
            paras.lambda1 = 0.01;
            paras.lambda2 = 15;
            paras.n_d = 80;
            paras.ksize = 7;
            paras.tol = 1e-4;
            paras.maxIter = 12;
            paras.stepsize = 0.1;

            %%%---Learn representations Z---%%%
    %             disp('--first pca to 100d; otherwise computation is prohibitively expensive.');
    %             [comp,XX,~] = pca(pattern, 'NumComponents',100);
            [D, Z, err] = TSC_ADMM(pattern',paras);
            disp('clustering via graph cut..');
    %             nbCluster = length(unique(label));
            vecNorm = sum(Z.^2);
            W2 = (Z'*Z) ./ (vecNorm'*vecNorm + 1e-6);
            [oscclusters,~,~] = ncutW(W2,n_clusters);
            idx = denseSeg(oscclusters, 1);

        elseif strcmp(method, 'ACA')
            idx = calACAOrHACA(pattern,n_clusters, 'ACA');
            idx(end) = []; %%% remove redudant frame


        elseif strcmp(method, 'HACA')
            idx = calACAOrHACA(pattern,n_clusters, 'HACA');
            idx(end) = []; %%% remove redudant frame

            
            
        elseif strcmp(method,'dclustering')
            time_window = 30;
            sigma = 30;

            [idx1, C] = incrementalClustering(double(pattern), time_window,sigma,0);
            idx = calLocalFeatureAggregationAndClustering_TUMKitchen(pattern,idx1,C, n_clusters,'normal_action'); 
        
        
            
            
        elseif strcmp(method,'ours')
            sigma = 30;
            dist_type = 0; % Euclidean distance
            verbose = 0;
    %                     disp('--online learn the clusters and labels..');
    %                     [idx, C] = incrementalClustering(double(pattern), time_window,sigma,0,0,1.0);
    %                     [idx1, C] = incrementalClusteringAndOnlineAgg(double(pattern), time_window, sigma, 0, 0, 1.0);
            [idx1, c_locs, c_stds, c_ex2, c_sizes] = dynamicEM(double(pattern), sigma, dist_type,verbose);
            %%% uncomment the following for online processing
    %                     disp('--postprocessing, merge clusters');
%             idx = fun_feature_aggregation(pattern,idx1,c_locs,'ours','BOMNI');
%             idx = fun_feature_aggregation_slidingwindow(pattern,c_locs,2.5e-2);
            idx = fun_feature_aggregation_kernelizedcut(pattern,c_locs,1e-3,'sequential');


        end
        CptTime = [CptTime toc(startTime)];
        res = evaluation_metric_TUMKitchen(yt, double(idx), 7, 1); % tol range = 7
        cmat_fainted = evaluation_fainted_detection_BOMNI(yt, double(idx));
        Pre = [Pre res.Segmentation.Pre];
        Rec = [Rec res.Segmentation.Rec];
        CMat = CMat + res.NovelBehavior.ConMat;
        CMat_fainted = CMat_fainted + cmat_fainted;
        

    end
    
    fprintf('====== method: %s==================\n',method);
    fprintf('- segmentation: avg_pre = %f\n',mean(Pre));
    fprintf('- segmentation: avg_rec = %f\n',mean(Rec));
    fprintf('- novelBehavior: avg_pre = %f\n',CMat(1,1)/(CMat(2,1)+CMat(1,1)));
    fprintf('- novelBehavior: avg_rec = %f\n',CMat(1,1)/(CMat(1,2)+CMat(1,1)));
    fprintf('- avg_runtime = %f seconds\n',mean(CptTime));
    fprintf('- fainting detection: avg_pre = %f\n',CMat_fainted(1,1)/(CMat_fainted(2,1)+CMat_fainted(1,1)));
    fprintf('- fainting detection: avg_rec = %f\n',CMat_fainted(1,1)/(CMat_fainted(1,2)+CMat_fainted(1,1)));

    if is_save
        result_filename = sprintf('BOMNI_Result_MaskDistTrans_%s.mat',method);
        save(result_filename, 'yt','idx','Pre','Rec','CMat','CptTime');
    end
end





 