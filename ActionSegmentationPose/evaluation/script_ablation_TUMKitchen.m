% clear all;
% close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../aca'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));
dataset_path = '/mnt/hdd/Dataset_TUMKitchen';
video_list = importdata([dataset_path '/video_list.txt']);


% moving_variance_window_set = round([0.25 0.5 1 2 4 6 8 10]*25);
gaussian_smooth_kernel_set = [0.001 2.5 4.5 6.5 8.5 10.5 12.5 25];
% peak_width_weight_set = 0.1:0.1:1;


%%% scenario configuration - before each running, check here!"
feature_list = {'jointLocs'};

% method_list = {'spectralClustering','TSC','ACA'};
method_list = {'ours'};
% bodypart_list = {'rightArm','leftArm','torso'};

bodypart_list = {'leftArm'};
% method_list = {'ours'};
% feature_list = {'jointLocs'};
% 
% bodypart = bodypart_list{1};
% feature = feature_list{1};
% method = method_list{end};

is_save = 0;
is_show = 0; 
%%% scenario configuration - end"



for bb = 1:length(bodypart_list)
for ff = 1:length(feature_list)
for mm = 1:length(method_list)


bodypart = bodypart_list{bb};
feature = feature_list{ff};
method = method_list{mm};


mean_Pre=  [];
mean_Rec = [];
mean_novPre = [];
mean_novRec = [];

for gaussian_smooth_kernel =  gaussian_smooth_kernel_set

Pre = [];
Rec = [];
CMat = zeros(2,2);
CptTime = [];


for vv = 1:length(video_list)
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
%             disp('--first pca to 100d; otherwise computation is prohibitively expensive.');
%             [comp,XX,~] = pca(pattern, 'NumComponents',100);
        [D, Z, err] = TSC_ADMM(pattern',paras);
        disp('clustering via graph cut..');
%             nbCluster = length(unique(label));
        vecNorm = sqrt(sum(Z.^2));
        W2 = (Z'*Z) ./ (vecNorm'*vecNorm + 1e-6);
        [oscclusters,~,~] = ncutW(W2,n_clusters);
        idx = denseSeg(oscclusters, 1);
        idx = idx;

%         uid = idx(1);
%         idx(idx==uid) = 10e6;
%         idx(idx==1) = uid;
%         idx(idx==10e6) = 1;
        idx = idx-1; %%% 0-based label


    elseif strcmp(method, 'ACA')
        idx = calACAOrHACA(pattern,n_clusters, 'ACA');
        idx(end) = []; %%% remove redudant frame

%         uid = idx(1);
%         idx(idx==uid) = 10e6;
%         idx(idx==1) = uid;
%         idx(idx==10e6) = 1;
%         idx = idx-1; %%% 0-based label
%         save('ACA_idx_TUMKitchen.mat','idx');

    elseif strcmp(method, 'HACA')
        idx = calACAOrHACA(pattern,36, 'HACA');
        idx(end) = []; %%% remove redudant frame
        
        
        
    elseif strcmp(method,'dclustering')
        time_window = 30;
        sigma = 0.005;

        [idx1, C] = incrementalClustering(double(pattern), time_window,sigma,0);
        idx = calLocalFeatureAggregationAndClustering_TUMKitchen(pattern,idx1,C, n_clusters,'normal_action'); 
        

    elseif strcmp(method,'ours')
        sigma = 10e3;
        dist_type = 0;
        verbose = 0;

%         disp('--online learn the clusters and labels..');
        [idx, c_locs, c_stds, c_ex2, c_sizes] = dynamicEM(double(pattern), sigma, dist_type,verbose);
%         idx2 = zeros(size(idx1));
%         for kk = 2:length(idx1)
%             if idx1(kk)~=idx1(kk-1)
%                 idx2(kk) = idx1(kk)+randi(length(unique(idx1)))-1;
%             else
%                 idx2(kk) = idx2(kk-1);
%             end
%         end
%         uid = mode(idx2);
%         idx2(idx2==uid) = 10e6;
%         idx2(idx2==0) = uid;
%         idx2(idx2==10e6) = 0;
%         disp('--postprocessing, merge clusters');
        idx = fun_feature_aggregation(pattern,idx,c_locs,'ours',gaussian_smooth_kernel,'TUMKitchen'); 
%         idx = idx1;
    end
    
    CptTime = [CptTime toc(startTime)];
    res = evaluation_metric_TUMKitchen(yt, double(idx), 15, is_show); % tol range = 15
    Pre = [Pre res.Segmentation.Pre];
    Rec = [Rec res.Segmentation.Rec];
    CMat = CMat + res.NovelBehavior.ConMat;
end
% fprintf('=================TUMKitchen====%s==%s===%s=========\n',bodypart,feature,method);
% fprintf('- segmentation: avg_pre = %f\n',mean(Pre));
% fprintf('- segmentation: avg_rec = %f\n',mean(Rec));
% % fprintf('- segmentation: avg_f-measure = %f\n',2*mean(Pre)*mean(Rec)/(mean(Pre)+mean(Rec)));
% fprintf('- novelBehavior: avg_pre = %f\n',CMat(1,1)/(CMat(2,1)+CMat(1,1)));
% fprintf('- novelBehavior: avg_rec = %f\n',CMat(1,1)/(CMat(1,2)+CMat(1,1)));
% fprintf('- avg_runtime = %f seconds\n',mean(CptTime));

if is_save
    result_filename = sprintf('TUMKitchen_Result_%s_%s_%s.mat',bodypart,feature,method);
    save(result_filename, 'yt','idx','Pre','Rec','CMat','CptTime');
end


mean_Pre = [mean_Pre; mean(Pre)];
mean_Rec = [mean_Rec; mean(Rec)];
mean_novPre = [mean_novPre; CMat(1,1)/(CMat(2,1)+CMat(1,1))];
mean_novRec = [mean_novRec; CMat(1,1)/(CMat(1,2)+CMat(1,1))];

end

figure(1);
set(gcf, 'Color','w');
c = lines;
plot(moving_variance_window_set, mean_Pre, '-s','Color',c(1,:), 'LineWidth',3.5);hold on;
plot(moving_variance_window_set, mean_Rec, '-s','Color',c(2,:), 'LineWidth',3.5);hold on;
plot(moving_variance_window_set, mean_novPre, '--s','Color',c(1,:), 'LineWidth',3.5);hold on;
plot(moving_variance_window_set, mean_novRec, '--s','Color',c(2,:), 'LineWidth',3.5);
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
lgd=legend ('precision (segmentation)','recall (segmentation)','precision (novelty detection)','recall (novelty detection)' ); lgd.FontSize = 15;
xlabel('window size for motion energy computation','FontSize',18);
ylabel('propotion','FontSize',18);
ylim([0 1]);
title('torso','FontSize',18);grid on;

end
end
end
    
    
