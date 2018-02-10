%%% this script aims to obtain the ground truth poses and visualize it in
%%% the dataset of TUMPatientMocap

clear all;
clc;
close all;


%% retrieve data from files

dataset_path = '/mnt/hdd/Dataset_PatientMocap';
filepath1 = [dataset_path '/' 'trainM/trainM.txt'];
filepath2 = [dataset_path '/' 'testM/testM.txt'];

%%% choose which file to open
fh = fopen(filepath2);

%%% choose which video to visualize
idx_video_seq = 4;

patterns = [];
poses = [];
idx_frame = 1;

while ~feof(fh)
    line = fgets(fh);
    if ~contains(line, sprintf('_seq%d',idx_video_seq))
        fprintf('-- read %s. SKIP!\n',line);    
        line = fgets(fh);
        line = fgets(fh);
        line = fgets(fh);
        
    else
        fprintf('-- read %s. READ!\n',line);    
        if contains(line,'feature')
            patterns = [patterns; str2num(fgets(fh))];        
        end
        
        if contains(line,'label')
            poses = [poses; str2num(fgets(fh))];        
        end
    end
end
        
%% visualize the pose information
fig = figure(1);
n_frames = size(poses,1);
n_joints = size(poses,2)/3;

for ii = 1:n_frames
    for jj = 1:n_joints
        name = sprintf('joint:%d',jj);
        scatter3(poses(ii, jj*3-2), poses(ii, jj*3-1), poses(ii, jj*3),'filled');hold on;
        text(poses(ii, jj*3-2), poses(ii, jj*3-1), poses(ii, jj*3),name);
    end
    pause(0.03);
    hold off;
end






    
