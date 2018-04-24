%%% (1) script to evaluate the abnormality recognition in TUMKitchen
%%% (2) frontend of the evaluation pipeline


clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));
dataset_path = '/mnt/hdd/Dataset_TUMKitchen';
video_list = importdata([dataset_path '/video_list.txt']);


%%% scenario configuration - before each running, check here!"
% feature_list = {'jointLocs','relativeAngle','quaternion'};
feature_list = {'quaternion'}; 

% method_list = {'spectralClustering','TSC','ACA'};
method_list = {'ours'};

% bodypart_list = {'rightArm','leftArm','torso'};
bodypart_list = {'leftArm'};

is_save = 0;
is_show = 0; 
%%% scenario configuration - end"



for bb = 1:length(bodypart_list)
for ff = 1:length(feature_list)
for mm = 1:length(method_list)


bodypart = bodypart_list{bb};
feature = feature_list{ff};
method = method_list{mm};


Pre = [];
Rec = [];
CMat = zeros(2,2);
CptTime = [];


for vv = 1:length(video_list)
% for vv = 1:1
    fprintf(' -- processing video %d\n',vv);
    
    
% for vv = 10:10    
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
    %------------------ run methods and make evaluations------------------%
    
    startTime = tic;
    if strcmp(method, 'kmeans')
%         [idx,C] = kmeans(pattern,n_clusters);
        [idx,C] = kmeans(pattern,50);
        
%         idx = calLocalFeatureAggregationAndClustering_TUMKitchen(pattern,idx,C, n_clusters,'normal_action'); 

    elseif strcmp(method, 'spectralClustering')
        idx = spectralClustering(pattern,500,n_clusters);
        idx = idx-1; %%% 0-based label

        
    elseif strcmp(method, 'TSC')
        %%%---Normalize the data---%%%
%             X = normalize(pattern);

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
%         disp('--first pca to 100d; otherwise computation is prohibitively expensive.');
%         [comp,XX,~] = pca(pattern, 'NumComponents',100);
        [D, Z, err] = TSC_ADMM(pattern',paras);
        disp('clustering via graph cut..');
        vecNorm = sqrt(sum(Z.^2));
        W2 = (Z'*Z) ./ (vecNorm'*vecNorm + 1e-6);
        [oscclusters,~,~] = ncutW(W2,n_clusters);
        idx = denseSeg(oscclusters, 1);
        idx = idx-1; %%% 0-based label


    elseif strcmp(method, 'ACA')
        idx = calACAOrHACA(pattern,n_clusters, 'ACA');
        idx(end) = []; %%% remove redudant frame
        
    elseif strcmp(method,'dclustering')
        time_window = 30;
        sigma = 0.005;

        [idx1, C] = incrementalClustering(double(pattern), time_window,sigma,0);
        idx = calLocalFeatureAggregationAndClustering_TUMKitchen(pattern,idx1,C, n_clusters,'normal_action'); 
        

    elseif strcmp(method,'ours')
        sigma = 1e-3;
        time_window = 25;
        dist_type = 0;
        verbose = 0;

%         [idx, c_locs, c_stds, c_ex2, c_sizes] = dynamicEM(double(pattern), sigma, dist_type,verbose);
        [idx, c_locs] = incrementalClustering(double(pattern), time_window,sigma,0);

%         idx = fun_feature_aggregation(pattern,idx,c_locs,'ours','TUMKitchen'); 
%         idx = fun_feature_aggregation_slidingwindow(pattern,c_locs,n_clusters);
        idx = fun_feature_aggregation_kernelizedcut(pattern,c_locs,1e-6,'batch');

    end
    
    CptTime = [CptTime toc(startTime)];
    res = evaluation_metric_TUMKitchen(yt, double(idx), 7, is_show); % tol range = 7
    Pre = [Pre res.Segmentation.Pre];
    Rec = [Rec res.Segmentation.Rec];
    CMat = CMat + res.NovelBehavior.ConMat;
end
fprintf('=================TUMKitchen====%s==%s===%s=========\n',bodypart,feature,method);
fprintf('- segmentation: avg_pre = %f\n',mean(Pre));
fprintf('- segmentation: avg_rec = %f\n',mean(Rec));
% fprintf('- segmentation: avg_f-measure = %f\n',2*mean(Pre)*mean(Rec)/(mean(Pre)+mean(Rec)));
fprintf('- novelBehavior: avg_pre = %f\n',CMat(1,1)/(CMat(2,1)+CMat(1,1)));
fprintf('- novelBehavior: avg_rec = %f\n',CMat(1,1)/(CMat(1,2)+CMat(1,1)));
fprintf('- avg_runtime = %f seconds\n',mean(CptTime));

if is_save
    result_filename = sprintf('TUMKitchen_Result_%s_%s_%s.mat',bodypart,feature,method);
    save(result_filename, 'yt','idx','Pre','Rec','CMat','CptTime');
end



end
end
end
    
    
