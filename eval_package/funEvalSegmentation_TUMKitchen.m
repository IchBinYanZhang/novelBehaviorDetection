
% Evaluation for action segmentation based on occurrence and boundary.
% The evaluation is implemented according to the following work: 
% Dong Huang, Yi Wang, Shitong Yao and F. De la Torre. Sequential Max-Margin Event Detectors, ECCV 2014
% The script is modified by yan.zhang@uni-ulm.de

function Result= funEvalSegmentation_TUMKitchen(gt, prd, thr, is_show, varargin)
% INPUTS:
%
% gt: frame-level ground truth label (obtain by loading a true label file)
% prd: segmentation results, labels are given by clustering
% thr: threshold of overlap ratio between. thr >=1, we match boundaries not
%      regions.
% 
% OUTPUTS:
% Result.tru_N: total number of gt segments
%       .dct_N: total number of detected segments
%       .dct_NT:number of correctly partitioned actions
%       .Prec: correctly detected actions over all detected actions (dct_NT/dct_N)
%       .Rec: correctly detected actions over all ground truth actions (dct_NT/tru_N)

%%%-------------------evaluation method---------------------------------%%%
%%% (1) Evaluation is based on recall and precision.
%%% (2) In contrast to CMUMAD that actions only appear once, in this dataset
%%%     an action can repeat several times. Therefore, we cannot define the
%%%     true positives as in CMUMAD.
%%% (3) In this case, we only match the regions of overlaping for computing
%%%     recall and precision if thr < 1 and match the boundary if thr > 1. Nothing more.

diff_gt = diff(gt);
diff_prd = diff(prd);




used_label = [];
if thr < 1
    tru_N = length(find(diff_gt))+1;
    dct_N = length(find(diff_prd))+1;
    dct_NT = 0;
    n_frames = length(gt);
    changeframe = 1;
    idx_change_left = -1;
    idx_change_right = -1;
    label_left = -1;
    label_right = -1;
    for ii = 1:tru_N
        for jj = changeframe:n_frames-1
           if abs(diff_gt(jj)) > 1e-6
               seg_gt = gt(changeframe:jj);
               seg_prd = prd(changeframe:jj);
               mj_label = mode(seg_prd);
               seg_prd(seg_prd==mj_label) = seg_gt(1);
               ratio = sum(seg_prd==seg_gt(1))/length(seg_gt);
               
                %%% the first detection
                if isempty(used_label)
                    if ratio > thr 
                        dct_NT = dct_NT + 1;
                    end
                    used_label = [used_label; [seg_gt(1),mj_label] ];
                else

                    %%% when a new ground truth action is evaluated
                    if isempty(find(used_label(:,1)==seg_gt(1)))
                        %%% if the detected label is also new, true positive++ and we
                        %%% update the label mapping
                        if ratio > thr && isempty( find(used_label(:,2)==mj_label) )
                            dct_NT = dct_NT + 1;
                            used_label = [used_label; [seg_gt(1),mj_label] ];
                        end

                    %%% when a gt action which has occured before..    
                    else
                        
                    %%% if the detected label matches to one of the
                    %%% previous corresponding labels
                        if ratio > thr && ismember(mj_label, used_label(used_label(:,1)==seg_gt(1),2))
                            dct_NT = dct_NT + 1;
                        end

                    end
                end
               
               changeframe = jj+1;
               break;
           end
        end
        if dct_NT >= dct_N
            break;
        end
    end
elseif thr >=1
    tru_idx = find(diff_gt);
    dec_idx = find(diff_prd);
    tru_N = length(find(diff_gt));
    dct_N = length(find(diff_prd));
    dct_NT = 0;
    n_frames = length(gt);
    for ii = 1:tru_N
        idx = tru_idx(ii);
        lb = max(1,idx-thr);
        ub = min(idx+thr, n_frames-1);
        seg_prd = diff_prd(lb:ub);
        if sum(find(seg_prd)) > 0
            dct_NT = dct_NT+1;
        end
    end
end
    
    
    
% Output------
Result.tru_N= tru_N; %total number of events 
Result.dct_N= dct_N; %total number of detected events
Result.dct_NT=dct_NT; %number of correctly detection events
Result.Prec= dct_NT/dct_N; %correctly detected events over all detected events (dct_NT/dct_N)
Result.Rec=dct_NT/tru_N;%correctly detected events over all ground truth events (dct_NT/tru_N)


% Show Bar-------
if is_show
    f = figure('Units', 'normalized', 'Position', [0,0.5,.8,0.2]);  
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
    
    colormap(cmap);

%     im_true = labelConv(gtlabE, 'slab2flab');
%     im_test = tslab;
    im_true = gt';
    im_test = prd';
    gtt = subplot(2,1,1);
    imagesc(im_true);
    % ft1 = title('');
    % set(ft1, 'FontSize', 10);
    set(gtt, 'XTick', []);
    set(get(gca,'XLabel'),'String','Frame')
    set(gtt, 'XTickLabel', []);
    set(gtt, 'YTick', []);
    set(get(gca,'YLabel'),'String','True')
    set(gtt, 'Layer', 'bottom');
    axis on
    title([' Results (',num2str(thr),' overlap): ',...
           ': Precision=', num2str(Result.Prec),...
           '; Recall=', num2str(Result.Rec)])

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
    if nargin == 8
        method = varargin{1};
        bodypart = varargin{2};
        feature = varargin{3};
        video_idx = varargin{4};
        figname = sprintf('%s_%s_%s_%d',method,bodypart, feature, video_idx);
        savefig(f, figname);
    end
    
end
end

% function a = findMostFrequentNumber(vec)
% ele = unique(vec);
% num = zeros(length(ele),1);
% a = 0;
% for i = 1: length(ele)
%     num = sum(vec == ele(i));
%     if num > a
%         a = num;
%     end
% end
% 
% end


function randIdx = calRandIdx(prd, gt)
a = 0; 
b = 0; 
c = 0; 
d = 0;

for ii = 1:length(gt)
    for jj = 1:length(gt)
        if prd(ii)==prd(jj)
            if gt(ii) == gt(jj)
                a = a+1;
            else
                c = c+1;
            end
        else
            if gt(ii) == gt(jj)
                b = b+1;
            else
                d = d+1;
            end
        end
    end
end
randIdx=  (a+b)/(a+b+c+d);
end





function [pp,rr,ff] = evaluate(prd, gt, delta)

%%% first, we find the index of the segment boundaries
prdf = abs(diff(prd));
gtf = abs(diff(gt));

idx_prdf = find(abs(diff(prd))>0.1);
idx_gtf = find(abs(diff(gt))>0.1);

% fprintf('--- n_segment_gt=%f  n_segment_prd = %f  \n', length(idx_gtf), length(idx_prdf));


pp = 0;
tp = 0;

%%% compute recall
for i = 1:length(idx_gtf)
    idx = idx_gtf(i);
    lb = max(1, idx-delta);
    ub = min(length(prdf), idx+delta);
    score_seg = prdf(lb:ub);
    if ~isempty(find(score_seg > 0.1))
        tp=tp+1;
    end
end
rr = tp/length(idx_gtf);

%%% compute precision
% for i = 1:length(idx_prdf)
%     idx = idx_prdf(i);
%     lb = max(1, idx-delta);
%     ub = min(length(gtf), idx+delta);
%     score_seg = gtf(lb:ub);
%     if ~isempty(find(score_seg > 0.1))
%         pp=pp+1;
%     end
% end
pp = tp/length(idx_prdf);
if isempty(idx_prdf)
    pp = 0;
    rr = 0;
end

%%% compute f measure
ff = 2*pp*rr/(pp+rr);

end

% function label = labelConv(lab, mode)
% %
% % Convert from frame-level label to segment-level label, or vice versa.
% %
% % Description 
% % label = labelConv(lab, mode) convert between frame-level label and
% % segment-level label according to the mode.
% %
% % Inputs ------------------------------------------------------------------
% %   o lab  : Frame-level label or segment-level label. Segment-level label
% %            must be N*2, the first column is the label, the second column
% %            should be segment length.
% %   o mode : 2 mode. 'flab2slab' or 'slab2flab'. 
% % Outputs -----------------------------------------------------------------
% %   o label: label after conversion
% % 
% % By: Shitong Yao  // yshtng(at)gmail.com    
% % Last modified: 18 July 2012
% % 
% if nargin < 2
%     error('Two input arguments required!'); 
% elseif nargin > 2
%     error('Too many input arguments!');
% end
% 
% if strcmpi(mode, 'flab2slab')
%     % Frame-level label to segment-level label
%     lab = [lab NaN];
%     slab = zeros(length(lab),2);
%     frame_count = 0;
%     seg_count = 0;
%     for i = 1:length(lab)-1
%         frame_count = frame_count + 1;        
%         if lab(i) ~= lab(i+1)   
%             seg_count = seg_count + 1;
%             slab(seg_count,:) = horzcat(lab(i), frame_count);
%             frame_count = 0;   
%             if i+1 == length(lab)
%                 break; 
%             end
%         end
%     end
%     label = slab(1:seg_count,:);  
% elseif strcmpi(mode, 'slab2flab')
%     % Segment-level label to frame-level label
%     flab = zeros(1, sum(lab(:,2)));
%     m = 0;
%     for i = 1:size(lab,1)
%         flab(1,m+1:m+lab(i,2)) = repmat(lab(i,1), 1, lab(i,2));
%         m = m + lab(i,2);
%     end
%     label = flab;
% else
%     error('No such mode!');
% end
% 
% end
