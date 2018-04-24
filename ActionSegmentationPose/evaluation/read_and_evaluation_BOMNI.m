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
method_list = {'DPMM-A'};
% method_list = {'GNG'};



is_show_result = 1;
is_save = 0;



for mm = 1:length(method_list)
    method = method_list{mm};
    Pre = [];
    Rec = [];
    CMat = zeros(2,2);
    CptTime = [];
    CMat_fainted = CMat;
    for vv = 1 : length(video_list)

        %%% read annotations
        obj = load( sprintf([dataset_path, '/scenario1/features/%s.mat'], video_list{vv})  );
        yt = obj.yt;


        if strcmp(method, 'DPMM-A')
            idx_file = sprintf([dataset_path, '/scenario1/DPMM_segmentation_results/BOMNI_%s_DPMM_SampleLabels.txt'], video_list{vv});
            idx = importdata(idx_file);

            pattern_file = sprintf([dataset_path, '/scenario1/DPMM_segmentation_results/BOMNI_%s_DPMM_SampleEncoding.txt'],video_list{vv});
            pattern = importdata(pattern_file);
            tt = tic;
%             idx = fun_feature_aggregation_no_encoding(pattern,idx, 'kmeans','BOMNI',4,2); 
            idx = calLocalFeatureAggregationWithEncodedFeatures_TUMKitchen(pattern,idx,6);

            tt = toc(tt);
            CptTime = [CptTime tt];
        else
            idx_file = sprintf([dataset_path, '/scenario1/%s_segmentation_results/BOMNI_%s_%s_SampleLabels.txt'],method, video_list{vv}, method);
            idx = importdata(idx_file);
        end

        res = evaluation_metric_TUMKitchen(yt, double(idx), 7, is_show_result); % tol range = 15
        Pre = [Pre res.Segmentation.Pre];
        Rec = [Rec res.Segmentation.Rec];
        CMat = CMat + res.NovelBehavior.ConMat;
        cmat_fainted = evaluation_fainted_detection_BOMNI(yt, double(idx));
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
        result_filename = sprintf('BOMNI_Result_MaskDistTrans_%s.mat','DPMM-A');
        save(result_filename, 'Pre','Rec','CMat','CptTime');
    end
end