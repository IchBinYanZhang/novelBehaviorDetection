clear all;
% close all;
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
feature_list = {'quaternion'};
% method_list = {'spectralClustering','TSC','ACA'};
method_list = {'ours'};


for ff = 1:length(feature_list)

for mm = 1:length(method_list)


feature = feature_list{ff};
method = method_list{mm};

if strcmp(method, 'spectralClustering') && strcmp(feature, 'jointLocs')
    continue;
end


is_show = 0; 
%%% scenario configuration - end"

Pre = [];
Rec = [];
CMat = zeros(2,2);
CptTime = [];

for vv = 1:length(video_list)    


    % only consider scenarios 03
    if ~contains(video_list{vv},'_03-')
        continue;
    end
    
    %------------------ fearture extraction and annotation----------------%
%     fprintf('------- processing: %s\n', video_list{vv});
    %%% find annotation
    bds = dataset_info(find( strcmp(dataset_info(:,1),video_list{vv})), 2:3 );
    ytt = unique(sort(reshape(cellfun(@str2num,bds),[],1)));
    
    %%% parse the skeleton
    [skel, mot] = readMocap([dataset_path '/' video_list{vv} '.c3d'],[],false);
    x = (cell2mat(mot.jointTrajectories))';
    yt = zeros(size(x,1),1);
    yt(ytt) = 100;
    if strcmp(feature, 'jointLocs')
        pattern = x;
    else
        n_joints = size(x,2)/3;
        xt = x;
        rat = [];
        qut = [];
   
        for j = 1:n_joints-2
            p1 = xt(:,3*j-2: 3*j);
            p2 = xt(:,3*(j+1)-2: 3*(j+1));
            p3 = xt(:,3*(j+2)-2: 3*(j+2));
            av = p2-p1;
            bv = p3-p2;
            a = av./repmat(sqrt(sum(av.^2,2))+1e-6,1,3);
            b = bv./repmat(sqrt(sum(bv.^2,2))+1e-6,1,3);

            %%% compute relative angles
            c = diag(a*b');
            rat = [rat acos(c)];

            %%% compute quaternions
            yaw = acos((a(:,1).*b(:,1)+a(:,2).*b(:,2))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,2).^2).*sqrt(b(:,1).^2+b(:,2).^2) ));
            pitch = acos((a(:,1).*b(:,1)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,3).^2).*sqrt(b(:,1).^2+b(:,3).^2) ));
            roll = acos((a(:,2).*b(:,2)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,2).^2+a(:,3).^2).*sqrt(b(:,2).^2+b(:,3).^2) ));
            qut = [qut angle2quat(yaw, pitch,roll)];
        end

        if strcmp(feature,'relativeAngle')
            pattern = rat;
        elseif strcmp(feature,'quaternion')
            pattern = qut;
        end
    end
        
    n_clusters = 15; %% this number refer to http://resources.mpi-inf.mpg.de/HDM05/2009_MuellerBa_MotionAnnotation_SCA.pdf
    %------------------ run methods and make evaluations------------------%
    
    startTime = tic;
    if strcmp(method, 'kmeans')
%         [idx,C] = kmeans(pattern,n_clusters);
        [idx,C] = kmeans(pattern,50);
        
%         idx = calLocalFeatureAggregationAndClustering_TUMKitchen(pattern,idx,C, n_clusters,'normal_action'); 

    elseif strcmp(method, 'spectralClustering')
        if strcmp(feature,'jointLocs')
            sigma = 500;
        elseif strcmp(feature,'relativeAngle')
            sigma = 1;
        elseif strcmp(feature,'quaternion')
            sigma = 1;
        end
        idx = spectralClustering(pattern,sigma,n_clusters);
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

    elseif strcmp(method,'ours')
        sigma = 0.025;
        dist_type = 0; % Euclidean distance
        verbose = 0;
%                     disp('--online learn the clusters and labels..');
%                     [idx, C] = incrementalClustering(double(pattern), time_window,sigma,0,0,1.0);
%                     [idx1, C] = incrementalClusteringAndOnlineAgg(double(pattern), time_window, sigma, 0, 0, 1.0);
        [idx1, c_locs, c_stds, c_ex2, c_sizes] = dynamicEM(double(pattern), sigma, dist_type,verbose);
        %%% uncomment the following for online processing
%                     disp('--postprocessing, merge clusters');
%         idx = fun_feature_aggregation(pattern,idx1,c_locs,'ours', 'HDM05');
        idx = fun_feature_aggregation_slidingwindow(pattern,c_locs,3e-5);

    end
    CptTime = [CptTime toc(startTime)];
    res = evaluation_metric_TUMKitchen(yt, double(idx), 7, false); % tol range = 7

    Pre = [Pre res.Segmentation.Pre];
    Rec = [Rec res.Segmentation.Rec];
    CMat = CMat + res.NovelBehavior.ConMat;
end
fprintf('====HDM05-sports========%s========%s================\n',feature, method)
fprintf('- segmentation: avg_pre = %f\n',mean(Pre));
fprintf('- segmentation: avg_rec = %f\n',mean(Rec));
fprintf('- novelBehavior: avg_pre = %f\n',CMat(1,1)/(CMat(2,1)+CMat(1,1)));
fprintf('- novelBehavior: avg_rec = %f\n',CMat(1,1)/(CMat(1,2)+CMat(1,1)));
fprintf('- avg_runtime = %f seconds\n',mean(CptTime));

result_filename = sprintf('HDM05_Result_%s_%s.mat',feature,method);
save(result_filename, 'Pre','Rec','CMat','CptTime');

end
end


    
    
