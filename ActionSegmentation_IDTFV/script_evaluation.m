clear all;
close all;
%clc;
addpath(genpath('../eval_package'));
addpath(genpath('../TSC'));
addpath(genpath('../mexIncrementalClustering'));

%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';
subject = 1:20;
method_set = {'ours'};



for mm = 1:length(method_set)
    method = method_set{mm};
    fprintf('======CUMAD:IDTFV======= evaluation method: %s==================\n',method);

    Pre = [];
    Rec = [];
    CMat = zeros(2,2);
    CptTime = [];

    for ss = subject
     
        
        for qq = 1:2
            gt_file = sprintf([dataset,'/sub%02d/seq%02d_label.mat'], ss,qq);
            load(gt_file); %  subject 1, sequence 
            gtlabE=  extractGTFormat(label); % true event-based labels (obtain by loading a true label file)
     
            feature_file = sprintf([dataset '/features/CMUMAD_idtfv_sub%02d_seq%02d.mat'],ss,qq);
            load(feature_file);
            pattern=X; % test frame-based labels produced by SVM+DP (baseline)
            clear X;        
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
                [comp,XX,~] = pca(pattern, 'NumComponents',100);
                addpath(genpath('../aca'));

                idx = calACAOrHACA(XX,36, 'ACA');
                idx(end) = []; %%% remove redudant frame
                rmpath(genpath('../aca'));
                uid = idx(1);
                idx(idx==uid) = 10e6;
                idx(idx==1) = uid;
                idx(idx==10e6) = 1;
                idx = idx-1; %%% 0-based label
 



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


            elseif strcmp(method, 'dclustering')
            
                time_window = 30;
                sigma = 0.00006; 
 
    %             disp('--online learn the clusters and labels..');
                [idx, C] = incrementalClustering(double(pattern), time_window,sigma,0);
                
                
                
                
            elseif strcmp(method,'ours')
                sigma = 0.00006; 
                dist_type = 0;
                verbose = 0;
 
    %             disp('--online learn the clusters and labels..');
                [idx, c_locs, c_stds, c_ex2, c_sizes] = dynamicEM(double(pattern), sigma,dist_type,verbose);

    %             disp('--postprocessing, merge clusters');
    %             idx = calLocalFeatureAggregationAndClustering(pattern,idx,C, 36); 

            end
            CptTime = [CptTime toc(startTime)];
            res = evaluation_metric_CMUMAD(gtlabE, double(idx'), 0.5, false); % tol range = 15
    %         pre = pre+res.Prec/2;
    %         rec = rec+res.Rec/2;
    %         acc = acc+res.Acc/2;
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

    result_filename = sprintf('CMUMAD_Result_%s_%s.mat','idtfv',method);
    save(result_filename, 'Pre','Rec','CMat','CptTime');
    
end


