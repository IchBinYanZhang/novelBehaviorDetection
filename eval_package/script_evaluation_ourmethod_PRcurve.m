clear all;
close all;
clc;

%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';
subject = [1 2 3 4 5];
feature = 'FV';
method = 'ours';

SSS = (1:10)*1e-3;
WWW = 500;



Precision =[];
Recall = [];
NCluster = [];

for sss = SSS
    for www = WWW
        fprintf('- time window = %f; sigma = %f\n',www,sss);
        avg_pre = 0;
        avg_rec = 0;
        avg_fs = 0;
        avg_ridx = 0;
        avg_ncluster = 0;
        for ss = subject
            pre = 0;
            rec = 0;
            fs = 0;
            ridx = 0;
            ncluster = 0;
            feature_file = sprintf([dataset, '/',feature,'/CMUMAD_EvaluationResults_sub%02d.mat'], ss);
            load(feature_file);
            for qq = 1:2
                if qq ==1
                    qqs = 2;
                else
                    qqs = 1;
                end
                gt_file = sprintf([dataset,'/sub%02d/seq%02d_label.mat'], ss,qq);
                load(gt_file); %  subject 1, sequence 
                gtlabE=  extractGTFormat(label); % true event-based labels (obtain by loading a true label file)
                n_frames = label(end,3)-1;
                pattern=eval_res{qqs}.stip_T_encoded{1}.feature; % test frame-based labels produced by SVM+DP (baseline)
                pattern = prdInterpolation(pattern, 50, n_frames); % stride and time window is fixed.

                n_clusters = 36;
                if strcmp(method, 'kmeans')
                    [idx,C] = kmeans(pattern,n_clusters);
                elseif strcmp(method, 'spectralClustering')
                    idx = spectralClustering(pattern,0.1,n_clusters);
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
                elseif strcmp(method, 'KTC-S')
                    disp('--first pca to 100d; otherwise computation is prohibitively expensive.');
                    [comp,XX,~] = pca(pattern, 'NumComponents',100);
                    sigma_s = 0.001;
                    sigma_t = 0.001;
                    time_window = 500;
                    idx = calKernelizedTemporalSegment(XX,time_window,sigma_s,sigma_t);
                elseif strcmp(method,'ours')
                    time_window = www;
                    sigma = sss;
                    is_temporal_reg = 0;
                    idx = incrementalClustering(double(pattern), time_window,sigma,is_temporal_reg,0,1);
                    
                end
                res = funEvalDetection(gtlabE, double(idx'), 15, false); % tol range = 15
                pre = pre+res.precision/2;
                rec = rec+res.recall/2;
                fs = fs+res.f_score/2;
                ridx = ridx + res.RandIdx/2;
                ncluster = ncluster + length(unique(idx))/2;
            end
            avg_pre = avg_pre+pre/5;
            avg_rec = avg_rec + rec/5;
            avg_fs = avg_fs + fs/5;
            avg_ridx = avg_ridx + ridx/5;
            avg_ncluster = avg_ncluster + ncluster/5;
            fprintf('--subject %02d: precision=%f, recall=%f, avg_ncluster=%f\n',ss,pre,rec,ncluster);
        end

        Precision = [Precision; avg_pre];
        Recall = [Recall; avg_rec];
        NCluster = [NCluster; avg_ncluster];
    end
end



