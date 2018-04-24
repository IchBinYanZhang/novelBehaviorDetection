clear all;
close all;
% clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));

%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';
subject = 1:20;
method_set = {'ours'};


for mm = 1:length(method_set)
    method = method_set{mm};
    fprintf('====CMUMAD:VGG16========= evaluation method: %s==================\n',method);

    Pre = [];
    Rec = [];
    CMat = zeros(2,2);
    CptTime = [];

    for ss = subject
        pre = 0;
        rec = 0;
        acc = 0;
     
        
        for qq = 1:2
            fprintf('-- processing: person %d, sequence %d\n',ss,qq);
            
            gt_file = sprintf([dataset,'/sub%02d/seq%02d_label.mat'], ss,qq);
            load(gt_file); %  subject 1, sequence 
            gtlabE=  extractGTFormat(label); % true event-based labels (obtain by loading a true label file)
     
            feature_file = sprintf([dataset '/features/CMUMAD_VGG16_sub%02d_seq%02d.mat'],ss,qq);
            load(feature_file);
            pattern=X; % test frame-based labels produced by SVM+DP (baseline)
            
    %         pattern = [pattern; pattern(end,:)];
            n_clusters = 36;
            startTime = tic;
            if strcmp(method, 'kmeans')
                [idx,C] = kmeans(pattern,n_clusters);
                uid = idx(1);
                idx(idx==uid) = 10e6;
                idx(idx==1) = uid;
                idx(idx==10e6) = 1;
                idx = idx-1; %%% 0-based label
    %             idx = calLocalFeatureAggregationAndClustering(pattern,idx,C, 36); 




            elseif strcmp(method, 'spectralClustering')
                idx = spectralClustering(pattern,1,n_clusters);
                uid = idx(1);
                idx(idx==uid) = 10e6;
                idx(idx==1) = uid;
                idx(idx==10e6) = 1;
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
                disp('--first pca to 100d; otherwise computation is prohibitively expensive.');
                [comp,XX,~] = pca(pattern, 'NumComponents',100);
                [D, Z, err] = TSC_ADMM(XX',paras);
                disp('clustering via graph cut..');
    %             nbCluster = length(unique(label));
                vecNorm = sum(Z.^2);
                W2 = (Z'*Z) ./ (vecNorm'*vecNorm + 1e-6);
                [oscclusters,~,~] = ncutW(W2,36);
                idx = denseSeg(oscclusters, 1);
                idx = idx';

                uid = idx(1);
                idx(idx==uid) = 10e6;
                idx(idx==1) = uid;
                idx(idx==10e6) = 1;
                idx = idx-1; %%% 0-based label





            elseif strcmp(method, 'ACA')
                
                addpath(genpath('../../aca'));

                idx = calACAOrHACA(pattern,36, 'ACA');
                idx(end) = []; %%% remove redudant frame
                
                uid = idx(1);
                idx(idx==uid) = 10e6;
                idx(idx==1) = uid;
                idx(idx==10e6) = 1;
                idx = idx-1; %%% 0-based label
                rmpath(genpath('../../aca'));

 



            elseif strcmp(method, 'HACA')
                [comp,XX,~] = pca(pattern, 'NumComponents',100);
                addpath(genpath('../../aca'));
                idx = calACAOrHACA(pattern,36, 'HACA');
                idx(end) = []; %%% remove redudant frame
                rmpath(genpath('../../aca'));




            elseif strcmp(method, 'KTC-S')
                disp('--first pca to 100d; otherwise computation is prohibitively expensive.');
                [comp,XX,~] = pca(pattern, 'NumComponents',100);
                sigma_s = 0.001;
                sigma_t = 0.001;
                time_window = 50;
                idx = calKernelizedTemporalSegment(XX,time_window,sigma_s,sigma_t);

                
                
                
            elseif strcmp(method,'dclustering')
                time_window = 30;
                sigma = 5;
                is_temporal_reg = 0;

%                 disp('--online learn the clusters and labels..');
                [idx, C] = incrementalClustering(double(pattern), time_window,sigma,0);
                %disp('--postprocessing, merge clusters');
                idx = calLocalFeatureAggregationAndClustering(pattern,idx,C, 36); 
                
                

            elseif strcmp(method,'ours')
                    time_window = 30;
                    sigma = 5;
                    dist_type = 0; % Euclidean distance
                    verbose = 0;
 
%                     [idx1, c_locs, c_stds, c_ex2, c_sizes] = dynamicEM(double(pattern), sigma, dist_type,verbose);
                    [idx1, c_locs] = incrementalClustering(double(pattern), time_window,sigma,0);

                    %%% uncomment the following for online processing
%                     disp('--postprocessing, merge clusters');
%                     idx = fun_feature_aggregation(pattern,idx1,c_locs,'ours','CMUMAD');
%                     idx = fun_feature_aggregation_slidingwindow(pattern,c_locs,2.5e-3);
                    idx = fun_feature_aggregation_kernelizedcut(pattern,c_locs,2.5e-3,'batch');


            end
            CptTime = [CptTime toc(startTime)];
            res = evaluation_metric_CMUMAD(gtlabE, double(idx'), 0.5, false); % tol range = 0.5

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

    result_filename = sprintf('CMUMAD_Result_%s_%s.mat','vgg16',method);
    save(result_filename, 'Pre','Rec','CMat','CptTime');
    
end


