function Result= evaluation_metric_CMUMAD(gtlabE, tslab, thr, is_show)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluation is two-fold:
%   (1) boundary-based precision-recall
%   (2) confusion matrix for novel behavior recognition
%       Each input sample is encoded to two states: a) in new cluster and
%       b) in old cluster
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INPUTS:
% gtlab: frame-level ground truth label (obtain by loading a true label file)
% tslab: frame-level label obtained by your algorithm
% thr: threshold for segment boundary matching
% 
% OUTPUTS:
% Result.Segmentation.
%       .Pre: correctly detected events over all detected events (dct_NT/dct_N)
%       .Rec: correctly detected events over all ground truth events (dct_NT/tru_N)
% Result.NovelBehavior.
%       .ConMat: confusion matrix
%       .Pre: precision of novel behavior
%       .Rec: recall of novel behavior



%%% parse ground truth files
gtlabE(gtlabE(:,1)==36, 1)=0; 
class_N=length(unique(gtlabE(:,1)));
e = cumsum(gtlabE(:,2));
s = [1; e(1:end-1)+1];
seglab = [s e];


%%% unify the length of annotation and detection
%%% in some videos of CMUMAD, annotation is longer than feature sequence
gtlabE(seglab(:,1)>=length(tslab)) = [];
seglab(seglab(:,1)>=length(tslab)) = [];

%%% consider all the labels including static poses
% sel = (gtlabE(:,1)~=0);
gtlab = gtlabE;
seglab = seglab;
gtlab_label = [gtlab(:,1) seglab];

%%% evaluation of segmentation, without considering the labels. 
%%% boundary based evaluation
bd_offset = thr;
tp = 0;
gt_idx = seglab(:,2);
prd_idx = find(diff(tslab));
gt_idx(end) = [];
true_N = length(gt_idx);
prd_N = length(prd_idx);
for i = 1:length(gt_idx)
    lb = max(1, gt_idx(i)-bd_offset);
    ub = min(length(tslab), gt_idx(i)+bd_offset);
    segment = tslab(lb:ub);
    prd_idx0 = find(diff(segment));
    if ~isempty(prd_idx0)
       tp = tp+1;
    end
end
if prd_N==0
    Result.Segmentation.Pre = 0;
    Result.Segmentation.Rec = 0;
else
    Result.Segmentation.Pre = tp/prd_N;
    Result.Segmentation.Rec = tp/true_N;
end


%%% evaluation of states classification.
%%% Each sample is encoded to two states: a) in old cluster and b) in new
%%% cluster

%%%% encode gtlab to state labels %%%%
n_frames = length(tslab);
yt = 2*ones(n_frames,1);
used_label = [];
for i = 1:size(seglab,1)
    if sum( ismember(used_label, gtlab_label(i,1) ))==0 %% new cluster
        yt(gtlab_label(i,2):gtlab_label(i,3)) = 1;
        used_label = [used_label; gtlab_label(i,1)];
    end
end

%%%% encode tslab to state labels %%%%
ytp = 2*ones(n_frames,1);
changeframe = 1;
j = 1;
used_label = [];
while j < n_frames
    jump = tslab(j+1)-tslab(j);
    if jump ~= 0  %% a boundary is detected
        seg = tslab(changeframe:j);
        mj_label = mode(seg);
        if sum( ismember(used_label, mj_label ))==0 %% new cluster
            ytp(changeframe : j) = 1;
            used_label = [used_label; mj_label];
        end
        changeframe = j;
    end
    j = j+1;
end


%%%% compute the confusion matrix 
Result.NovelBehavior.ConMat = confusionmat(yt, ytp);
Result.NovelBehavior.Pre = Result.NovelBehavior.ConMat(1,1)/sum(ytp==1);
Result.NovelBehavior.Rec = Result.NovelBehavior.ConMat(1,1)/sum(yt==1);




% Show Bar-------
if is_show
    f = figure('Units', 'normalized', 'Position', [0,0.5,.8,0.2]);

    param.height = 1;
    param.class_N = class_N;   
    map = [[148,131,166],
            [129,232,51],
            [158,48,219],
            [118,236,100],
            [209,63,232],
            [77,190,55],
            [92,37,181],
            [233,232,41],
            [153,84,238],
            [198,235,72],
            [76,84,224],
            [150,183,40],
            [229,77,222],
            [95,229,135],
            [176,58,190],
            [65,161,58],
            [222,51,173],
            [168,224,110],
            [67,35,137],
            [225,224,88],
            [153,100,221],
            [228,193,48],
            [88,112,213],
            [219,138,31],
            [43,35,98],
            [224,222,128],
            [152,53,152],
            [103,238,183],
            [235,48,119],
            [46,191,137],
            [237,64,32],
            [99,236,224],
            [200,51,39],
            [89,204,230],
            [228,104,26],
            [119,150,226],
            [221,167,65],
            [219,116,218],
            [101,191,108],
            [226,76,160],
            [77,133,37],
            [119,75,156],
            [182,225,147],
            [103,29,92],
            [178,160,55],
            [186,138,217],
            [41,88,22],
            [165,49,121],
            [114,192,142],
            [219,55,80],
            [39,160,130],
            [223,92,131],
            [75,143,83],
            [214,118,178],
            [143,165,77],
            [163,44,84],
            [86,194,186],
            [223,103,61],
            [93,171,220],
            [134,53,21],
            [180,225,222],
            [154,47,49],
            [175,224,183],
            [49,21,48],
            [233,209,149],
            [86,82,135],
            [140,111,34],
            [74,119,174],
            [176,103,48],
            [75,148,165],
            [218,108,99],
            [53,101,66],
            [223,174,225],
            [39,50,27],
            [230,145,175],
            [100,110,43],
            [149,83,130],
            [220,222,192],
            [97,31,52],
            [146,190,203],
            [78,38,28],
            [188,193,226],
            [106,73,33],
            [227,188,191],
            [40,46,63],
            [230,152,102],
            [61,92,117],
            [191,160,102],
            [51,99,94],
            [218,148,137],
            [97,159,139],
            [182,100,118],
            [125,145,109],
            [141,79,71],
            [114,138,145],
            [144,121,87],
            [107,79,92],
            [187,173,146],
            [87,89,63],
            [163,126,127]]/255;
    cmap = colormap(map);
    cmap(1,:) = .9*[1 1 1];
    colormap(cmap);

    im_true = labelConv(gtlabE, 'slab2flab');
    im_test = tslab;

    gt = subplot(2,1,1);
    imagesc(im_true);
    set(gt, 'XTick', []);
    set(get(gca,'XLabel'),'String','Frame')
    set(gt, 'XTickLabel', []);
    set(gt, 'YTick', []);
    set(get(gca,'YLabel'),'String','True')
    set(gt, 'Layer', 'bottom');
    axis on

    ts = subplot(2,1,2);
    imagesc(im_test);
    % ft2 = title('');
    % set(ft2, 'FontSize', 10);
    set(gcf, 'Color','white');
    set(ts, 'XTick', []);
    set(get(gca,'XLabel'),'String','Frame')
    set(ts, 'XTickLabel', []);
    set(ts, 'YTick', []);
    set(get(gca,'YLabel'),'String','Detected')
    set(ts, 'Layer', 'bottom');
    axis on
    end
end



function label = labelConv(lab, mode)
%
% Convert from frame-level label to segment-level label, or vice versa.
%
% Description 
% label = labelConv(lab, mode) convert between frame-level label and
% segment-level label according to the mode.
%
% Inputs ------------------------------------------------------------------
%   o lab  : Frame-level label or segment-level label. Segment-level label
%            must be N*2, the first column is the label, the second column
%            should be segment length.
%   o mode : 2 mode. 'flab2slab' or 'slab2flab'. 
% Outputs -----------------------------------------------------------------
%   o label: label after conversion
% 
% By: Shitong Yao  // yshtng(at)gmail.com    
% Last modified: 18 July 2012
% 
if nargin < 2
    error('Two input arguments required!'); 
elseif nargin > 2
    error('Too many input arguments!');
end

if strcmpi(mode, 'flab2slab')
    % Frame-level label to segment-level label
    lab = [lab NaN];
    slab = zeros(length(lab),2);
    frame_count = 0;
    seg_count = 0;
    for i = 1:length(lab)-1
        frame_count = frame_count + 1;        
        if lab(i) ~= lab(i+1)   
            seg_count = seg_count + 1;
            slab(seg_count,:) = horzcat(lab(i), frame_count);
            frame_count = 0;   
            if i+1 == length(lab)
                break; 
            end
        end
    end
    label = slab(1:seg_count,:);  
elseif strcmpi(mode, 'slab2flab')
    % Segment-level label to frame-level label
    flab = zeros(1, sum(lab(:,2)));
    m = 0;
    for i = 1:size(lab,1)
        flab(1,m+1:m+lab(i,2)) = repmat(lab(i,1), 1, lab(i,2));
        m = m + lab(i,2);
    end
    label = flab;
else
    error('No such mode!');
end

end


