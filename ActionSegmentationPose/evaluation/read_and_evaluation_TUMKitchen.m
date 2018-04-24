clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../aca'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));
dataset_path = '/mnt/hdd/Dataset_TUMKitchen';
video_list = importdata([dataset_path '/video_list.txt']);



%%% scenario configuration - before each running, check here!"
feature_list = {'jointLocs','relativeAngle','quaternion'};
bodypart_list = {'leftArm','rightArm','torso'};
% feature_list = {'quaternion'};
method_list = {'GNG'};
% method_list = {'GNG'};

is_show = 0;


for bb = 1:length(bodypart_list)
    for ff = 1:length(feature_list)
        for mm = 1:length(method_list)
        
        
        method = method_list{mm};
        bodypart = bodypart_list{bb};
        feature = feature_list{ff};
        
        Pre = [];
        Rec = [];
        CptTime = [];
        CMat = zeros(2,2);
        
        for vv = 1:length(video_list)
            %------------------ read annotation----------------%
            anno = importdata([dataset_path '/' video_list{vv} '/labels.csv']);
            [yt_left_arm, yt_right_arm, yt_torso] = readTUMKitchenAnnotation(anno);
            if strcmp(bodypart,'leftArm')
                n_clusters = 9;
                yt = yt_left_arm;
            elseif strcmp(bodypart,'rightArm')
                n_clusters = 9;
                yt = yt_right_arm;
            elseif strcmp(bodypart,'torso')
                n_clusters = 2;
                yt = yt_torso;
            end


            
% %             %------------------ aggregation and clustering------------------%
            if strcmp(method, 'DPMM-A')
                idx_file = sprintf([dataset_path '/DPMM_segmentation_results/TUMkitchen_%s_%s_v%d_DPMM_SampleLabels.txt'],...
                           bodypart, feature, vv);
                idx = importdata(idx_file);

                X_file = sprintf([dataset_path '/DPMM_segmentation_results/TUMkitchen_%s_%s_v%d_DPMM_SampleEncoding.txt'],...
                               bodypart, feature, vv); 
                X = importdata(X_file);
                starttime = tic;
                
                if strcmp(bodypart, 'torso')
                    n_still = 1;
                    n_move = 1;
                elseif strcmp(bodypart, 'rightArm')
                    n_still = 1;
                    n_move = n_clusters-n_still;
                elseif strcmp(bodypart, 'leftArm')
                    n_still = 1;
                    n_move = n_clusters-n_still;
                end
                
                idx = fun_feature_aggregation_no_encoding(X,idx, 'kmeans','TUMKitchen',n_move,n_still); 
                eps_time = toc(starttime);
                CptTime = [CptTime eps_time];
            else
                idx_file = sprintf([dataset_path '/%s_segmentation_results/TUMkitchen_%s_%s_v%d_%s_SampleLabels.txt'],...
                           method, bodypart, feature, vv, method);
                idx = importdata(idx_file);
            end

            
            %------------------ evaluation ---------------------------%
            res = evaluation_metric_TUMKitchen(yt, double(idx), 7, is_show); % tol range = 15
            Pre = [Pre res.Segmentation.Pre];
            Rec = [Rec res.Segmentation.Rec];
            CMat = CMat + res.NovelBehavior.ConMat;
        end
        
        fprintf('====== method: %s======feature: %s=====bodypart:%s =======\n',method,feature,bodypart);
        fprintf('- segmentation: avg_pre = %f\n',mean(Pre));
        fprintf('- segmentation: avg_rec = %f\n',mean(Rec));
        fprintf('- novelBehavior: avg_pre = %f\n',CMat(1,1)/(CMat(2,1)+CMat(1,1)));
        fprintf('- novelBehavior: avg_rec = %f\n',CMat(1,1)/(CMat(1,2)+CMat(1,1)));
        if strcmp(method, 'DPMM-A')
            fprintf('- avg_runtime = %f seconds\n',mean(CptTime));
        end
        
        end
    end
end




    
    
