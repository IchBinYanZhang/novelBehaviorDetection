clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));

%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';
subject = 1:20;

feature = 'FV2';


fprintf('============= save idtfv features to disk ==================\n');


for ss = subject
    pre = 0;
    rec = 0;
    acc = 0;
    feature_file = sprintf([dataset, '/',feature,'/CMUMAD_EvaluationResults_sub%d.mat'], ss);
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
        n_frames = label(end,3);

        pattern=eval_res{qqs}.stip_T_encoded{1}.feature; % test frame-based labels produced by SVM+DP (baseline)
        X = prdInterpolation(pattern, 50, n_frames); % stride and time window is fixed.

        outfile = sprintf('CMUMAD_idtfv_sub%02d_seq%02d.mat',ss,qq );
        save(outfile, 'X');
    end
end



