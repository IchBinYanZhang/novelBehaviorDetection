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
% method_set = { 'ours'};
pose_feature_set = {'quaternions'};
method_set = {'ours'};
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
                startTime = tic;
                if strcmp(method, 'kmeans')
                    [idx,C] = kmeans(pattern,36);
                    uid = idx(1);
                    idx(idx==uid) = 10e6;
                    idx(idx==1) = uid;
                    idx(idx==10e6) = 1;
                    idx = idx-1; %%% 0-based label
                    idx = calLocalFeatureAggregationAndClustering(pattern,idx,C, 36); 

                    
                elseif strcmp(method, 'regularSampling')
                        %%% regular sampling
                    prd_boundary = 1:30:size(pattern,1);
                    idx = zeros(size(pattern,1),1);
                    idx(prd_boundary) = 1;
                    
                elseif strcmp(method, 'spectralClustering')
                    idx = spectralClustering(pattern,1,n_clusters);
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
                    [oscclusters,~,~] = ncutW(W2,36);
                    idx = denseSeg(oscclusters, 1);
                    idx = idx';

                elseif strcmp(method, 'ACA')
                    idx = calACAOrHACA(pattern,36, 'ACA');
                    idx(end) = []; %%% remove redudant frame
                    

                elseif strcmp(method, 'HACA')
                    idx = calACAOrHACA(pattern,36, 'HACA');
                    idx(end) = []; %%% remove redudant frame

                    
                elseif strcmp(method, 'dclustering')
                    time_window = 30;
                    sigma = 0.09;
                    
%                     disp('--online learn the clusters and labels..');
                    [idx, C] = incrementalClustering(double(pattern), time_window,sigma,0);

                    %%% uncomment the following for online processing
%                     disp('--postprocessing, merge clusters');
                    idx = calLocalFeatureAggregationAndClustering(pattern,idx,C, 36); 
                    
                    
                elseif strcmp(method,'ours')
                    
                    fprintf('================ video %d, %d =========\n', ss,qq);

                    idx = [];
                    sigma = 0.1;
                    dist_type = 0; % Euclidean distance
                    verbose = 0;

                    n_batches = 3; n_frames = size(pattern,1);
                    
                    for bb = 1:n_batches
                        range = round(n_frames/n_batches);
                        lb = 1;
                        ub = min(n_frames, bb*range);
                        pp = pattern(lb:ub, :);
                        
%                         fprintf('-iteration %d\n',bb);
                        if bb==1
                            [idx1, c_locs1, c_stds1, c_ex21, c_sizes1] = dynamicEM(double(pp), sigma, dist_type,verbose);
                        else
                            [idx1, c_locs1, c_stds1, c_ex21, c_sizes1] = dynamicEM(double(pp), sigma, dist_type,verbose,c_locs, c_stds, c_ex2, c_sizes);
                        end
                    %%% uncomment the following for online processing
                        idx = fun_feature_aggregation_online(double(pp),idx1,c_locs1,idx,'ours','CMUMAD');
                        clear idx1;
                        c_locs = c_locs1; c_stds =  c_stds1; c_ex2 = c_ex21; c_sizes = c_sizes1;
%                         idx = [idx; idx0];
%                     idx = fun_feature_aggregation_slidingwindow(pattern,c_locs,1e-4);
                    end
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
       
%         result_filename = sprintf('CMUMAD_Result_%s_%s.mat',pose_feature,method);
%         save(result_filename, 'Pre','Rec','CMat','CptTime');
    end
end
