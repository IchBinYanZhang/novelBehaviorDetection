
% Evaluation for action segmentation based on occurrence and boundary.
% The evaluation is implemented according to the following work: 
% Dong Huang, Yi Wang, Shitong Yao and F. De la Torre. Sequential Max-Margin Event Detectors, ECCV 2014
% The script is modified by yan.zhang@uni-ulm.de

function Result= evaluation_metric_TUMKitchen(gt, prd, thr, is_show)
% INPUTS:
%
% gt: frame-level ground truth label (obtain by loading a true label file)
% prd: segmentation results, labels are given by clustering
% thr: threshold of overlap ratio between. thr >=1, we match boundaries not
%      regions.
% 
% OUTPUTS:
% Result.Segmentation.
%       .Pre: correctly detected events over all detected events (dct_NT/dct_N)
%       .Rec: correctly detected events over all ground truth events (dct_NT/tru_N)
% Result.NovelBehavior.
%       .ConMat: confusion matrix
%       .Pre: precision of novel behavior
%       .Rec: recall of novel behavior



%%% evaluation of temporal segmentation
diff_gt = diff(gt);
diff_prd = diff(prd);

tru_idx = find(diff_gt);
dec_idx = find(diff_prd);
tru_N = length(tru_idx);
dct_N = length(dec_idx);
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


if dct_N==0
    Result.Segmentation.Pre = 0;
    Result.Segmentation.Rec = 0;
else
    Result.Segmentation.Pre = dct_NT/dct_N;
    Result.Segmentation.Rec = dct_NT/tru_N;
end


%%% evaluation of states classification.
%%% Each sample is encoded to two states: a) in old cluster and b) in new
%%% cluster

%%%% encode gtlab to state labels %%%%

yt = 2*ones(n_frames,1);
changeframe = 1;
j = 1;
used_label = [];
while j <= n_frames
    if j ~= n_frames
        jump = gt(j+1)-gt(j);
        if jump ~= 0  %% a boundary is detected
            seg = gt(changeframe:j);
            mj_label = mode(seg);
            if sum( ismember(used_label, mj_label ))==0 %% new cluster
                yt(changeframe : j) = 1;
                used_label = [used_label; mj_label];
            end
            changeframe = j;
        end
        
    else
        seg = gt(changeframe:j);
        mj_label = mode(seg);
        if sum( ismember(used_label, mj_label ))==0 %% new cluster
            yt(changeframe : j) = 1;
            used_label = [used_label; mj_label];
        end
    end
    j = j+1;
end


%%%% encode tslab to state labels %%%%
ytp = 2*ones(n_frames,1);
changeframe = 1;
j = 1;
used_label = [];
while j <= n_frames
    if j ~= n_frames
        jump = prd(j+1)-prd(j);
        if jump ~= 0  %% a boundary is detected
            seg = prd(changeframe:j);
            mj_label = mode(seg);
            if sum( ismember(used_label, mj_label ))==0 %% new cluster
                ytp(changeframe : j) = 1;
                used_label = [used_label; mj_label];
            end
            changeframe = j;
        end
        
    else
        seg = prd(changeframe:j);
        mj_label = mode(seg);
        if sum( ismember(used_label, mj_label ))==0 %% new cluster
            ytp(changeframe : j) = 1;
            used_label = [used_label; mj_label];
        end
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
           ': Precision=', num2str(Result.Segmentation.Pre),...
           '; Recall=', num2str(Result.Segmentation.Rec)])

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



