clear all;
close all;
clc;
addpath(genpath('../../eval_package'));

%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';
subject = 1:20;
method_list = {'DPMM','DPMM-A'};




for mm = 1:length(method_list)
    method = method_list{mm};

    Pre = [];
    Rec = [];
    CMat = zeros(2,2);
    CptTime = [];
for ss = subject
    pre = 0;
    rec = 0;

    for qq = 1:2
        gt_file = sprintf([dataset,'/sub%02d/seq%02d_label.mat'], ss,qq);
        load(gt_file);
        gtlabE=  extractGTFormat(label);
        n_frames = label(end,3);

        idx_file = sprintf([dataset, '/DPMM_segmentation_results/CMUMAD_VGG16_sub%02d_seq%02d_DPMM_SampleLabels.txt'],ss,qq);
        idx = importdata(idx_file);
        
        if strcmp(method, 'DPMM-A')
            X_file = sprintf([dataset, '/DPMM_segmentation_results/CMUMAD_VGG16_sub%02d_seq%02d_DPMM_SampleEncoding.txt'],ss,qq);
            X = importdata(X_file);
        
            startTime = tic;
            idx = fun_feature_aggregation_no_encoding(X,idx, 'kmeans','CMUMAD',35,1); 
            CptTime = [CptTime toc(startTime)];
        end
        
        res = evaluation_metric_CMUMAD(gtlabE, double(idx'), 15, false); % tol range = 15
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

result_filename = sprintf('CMUMAD_Result_%s_%s.mat','VGG16',method);
save(result_filename, 'idx','Pre','Rec','CMat','CptTime');
    

end



