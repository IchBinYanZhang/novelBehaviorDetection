clear all;
clc;
close all;

subject = 6:20;
sequence = 1:2;
dataset_path = '/home/yzhang/Videos/Dataset_CMUMAD';
for sub = subject
    for seq = sequence
        jointLocs = [];
        relativeAngle = [];
        quaternions = [];
        
        fprintf([ '- processing: ' dataset_path, '/sub%02d/seq%02d_sk.mat\n'],sub,seq);
        filename = sprintf([dataset_path, '/sub%02d/seq%02d_sk.mat'],sub,seq);
        load(filename);
        n_frames = length(skeleton);
        for ii = 1:n_frames
            jointLocs = [jointLocs; reshape(skeleton{ii}',1,[])];
            [ang,qua] = calRotationFromPosition(skeleton{ii});
            relativeAngle = [relativeAngle; ang'];
            quaternions = [quaternions; reshape(qua',1,[])];
        end
        pattern.jointLocs = jointLocs;
        pattern.relativeAngle = relativeAngle;
        pattern.quaternions = quaternions;
        outfilename = sprintf('PoseFeature_sub%02d_seq%02d.mat', sub,seq);
        save(outfilename,'pattern');
    end
end


       
