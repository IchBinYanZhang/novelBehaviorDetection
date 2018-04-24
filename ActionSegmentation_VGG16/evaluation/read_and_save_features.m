clear all;
close all;
clc;
addpath(genpath('../../eval_package'));
addpath(genpath('../../TSC'));
addpath(genpath('../../mexIncrementalClustering'));

%% evaluate script
dataset = '/home/yzhang/Videos/Dataset_CMUMAD';
subject = 1:20;
feature = 'VGG';





for ss = subject
    for qq = 1:2

        feature_file = sprintf([dataset, '/',feature,'/CMUMAD_sub%02d_seq%02d.mat'], ss,qq);
        data = load(feature_file);
        gt_file = sprintf([dataset,'/sub%02d/seq%02d_label.mat'], ss,qq);
        load(gt_file); %  subject 1, sequence 
        gtlabE=  extractGTFormat(label); % true event-based labels (obtain by loading a true label file)
        n_frames = label(end,3)-1;
        pattern1 = double(data.pattern');

        X = [pattern1; pattern1(end,:)];

        outfile = sprintf('CMUMAD_VGG16_sub%02d_seq%02d.mat',ss,qq );
        save(outfile,'X');

    end
end


