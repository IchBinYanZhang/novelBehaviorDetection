% The MAD database example codes
% Citation: 
% Dong Huang, Yi Wang, Shitong Yao and F. De la Torre. Sequential Max-Margin Event Detectors, ECCV 2014

% This is an example of using funEvalDetection to compute event-based
% detection results.
clc; close all; clear;

subject = 'sub01';
gt_file = sprintf('/home/yzhang/Videos/Dataset_CMUMAD/%s/seq01_label.mat', subject);
load(gt_file); %  subject 1, sequence 1
gtlabE=  extractGTFormat(label); % true event-based labels (obtain by loading a true label file)
n_frames = label(end,3)-1;
pattern=eval_res{2}.stip_T_encoded{1}.feature; % test frame-based labels produced by SVM+DP (baseline)
pattern = prdInterpolation(pattern, 50, n_frames);

thr=0.5; % overlap threshold between a true event and a detect event
% Result= funEvalDetection(gtlabE, tslab, thr);
X = double(pattern);

W = 1:50:1000;
S = 1e-5*[1 2.5 5 7.5 10 25 50 75 100 250 500 750 1000 2500 5000 7500 10000];
use_temporal_reg = 1;


precision = zeros(length(W), length(S));
recall = zeros(length(W), length(S));
f_measure = zeros(length(W), length(S));

for i = 1:length(W)
    for j = 1:length(S)
        fprintf('-- time_window=%f  sigma = %f\n', W(i), S(j));
        labels_output= incrementalClustering(X', W(i), S(j), use_temporal_reg, 0);
        Result= funEvalDetection(gtlabE, labels_output, thr);
        precision(i,j) = Result.precision;
        recall(i,j) = Result.recall;
        f_measure(i,j) = 2* Result.precision * Result.recall /(Result.precision + Result.recall);
        fprintf('--- precision=%f  recall = %f  f_score=%f \n', precision(i,j),recall(i,j),f_measure(i,j));

    end
end