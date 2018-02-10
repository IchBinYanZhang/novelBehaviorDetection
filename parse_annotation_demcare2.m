function labels = parse_annotation_demcare2(varargin)
%%% This function aims to retrieve data and annotation for DemCare2. 
%%% When is_show = true and the video recording is specified, annotation is
%%% visualized.

dataset_path = '/home/yzhang/Videos/Dataset_DemCare/DS1/DemCare2';
annotation = {'ER','SB','DB','PS','ES','CU','SP','EP','UC','RP','HS','TV'};


video_seq = varargin{1};
if nargin > 1
    modality = varargin{2}; %% {'Depth','RGB'};
    is_show = varargin{3}; 
    
else
    modality = 'RGB'; %% {'Depth','RGB'};
    is_show = false; 
    
end

%%% read annotation data
dataset_info = importdata([dataset_path, '/KinectSequences.txt']);
seq_gt = strsplit(dataset_info{video_seq});
seq_gt2 = strsplit(seq_gt{2},',');
frame_list = dir(sprintf([dataset_path '/s%d/%s*.jpeg'], video_seq,modality));
n_frames = length(frame_list);

%%% rearange the annotation into framewise labeling
labels = zeros(n_frames,1); %% the label 0 denotes the background actions, which are not annotated by the dataset creater.

for ii = 1:n_frames
    ll = labeling_one_frame(ii, seq_gt2);
    if ~isempty(ll)
        labels(ii) = find(cellfun(@(x) strcmp(x, ll), annotation));
    end

end

end




function label = labeling_one_frame(idx_frame, seq_gt2)

label = [];
for ii = 1:length(seq_gt2)-1
    info = strsplit(seq_gt2{ii},{':','&'});
    if isempty(info{2})
        continue;
    end
    n_chunks= length(info);
    for jj = 2:n_chunks
        info2 = strsplit(info{jj},'-');
        lb = str2num(info2{1});
        ub = str2num(info2{2});
    
        if idx_frame >= lb && idx_frame <= ub
            label = info{1};
            break;
        end
    end
end
end



