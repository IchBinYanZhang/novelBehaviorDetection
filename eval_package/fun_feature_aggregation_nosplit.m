function idx = fun_feature_aggregation_nosplit(X,Xl,C,method,varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function consists of two steps: feature aggregation and clustering.
% Feature aggregation = feature encoding + temporal pooling.
% Stationary regions are moving regions are separately aggregated.
% stationary states and moving action patterns are separately clustered.
% Action parsing is combinition of the two.

% '_nosplit' means this funtion dynamic-clusters all windows together
% this function is used for ablation study
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset=varargin{1};

if strcmp(dataset, 'CMUMAD')

    %%% set params CMUMAD%%%
    moving_variance_window = 60;
    gaussian_filter_sigma = 12.5;
    min_peak_distance = 60;
    sigma_dclustering = 1e-8;
    sigma_dclustering_stationary = 0.75;
    peak_width_weight = 0.65;
    
elseif strcmp(dataset, 'TUMKitchen')
    %%% set params TUMKitchen%%%
    %%% jointlocation, rightarms + leftarms %%%
    moving_variance_window = 25;
    gaussian_filter_sigma = 6.5;
    min_peak_distance = 25;
    sigma_dclustering = 0.03;
    sigma_dclustering_stationary = 0.03;
    peak_width_weight = 0.5;
    %%%
elseif strcmp(dataset,'HDM05') % to tune
    moving_variance_window = 25;
    gaussian_filter_sigma = 6.5;
    min_peak_distance = 25;
    sigma_dclustering = 0.05;
    sigma_dclustering_stationary = 5;
    peak_width_weight = 0.75;

elseif strcmp(dataset,'BOMNI')
    %%% set params BOMNI scenario1 %%%
    moving_variance_window = 30;
    gaussian_filter_sigma = 7.5;
    min_peak_distance = 45;
    sigma_dclustering = 0.1;
    sigma_dclustering_stationary = 1e-5;
    peak_width_weight = 0.5;
%%%
end



%% aggregate action patterns
%%% moving variance computation
Xll = movvarcat(Xl, moving_variance_window);

%%% find the peaks in the label space to determine the time window
labels = imgaussfilt(Xll,gaussian_filter_sigma);
[pks_label,locs_label, pks_width] = findpeaks(labels, 'MinPeakDistance',min_peak_distance);

%%% use the time window to aggregate features

bds = [1];
for pp = 1:length(pks_label)
    lb = max(1,locs_label(pp)-round(peak_width_weight*pks_width(pp)));
    ub = min(size(X,1), locs_label(pp)+round(peak_width_weight*pks_width(pp)));
    if lb > bds(end)
        bds = [bds lb];
    end
    bds = [bds ub];
end

features = zeros(length(bds)-1, size(C,1));
for ii = 1:length(bds)-1
    XX = X(bds(ii) : bds(ii+1), :);
    dist = pdist2(XX, C);
    ff = exp(-0.1*dist)./ repmat(sum(exp(-0.1*dist),2), 1, size(C,1));
    ff2 = sum(ff,1);
    features(ii,:) = ff2/(1e-6+norm(ff2,2));
end


if strcmp(method, 'kmeans')
    nc = varargin{2};
    if size(features,1) <= nc
        iidx = 1:size(features,1);
    else
        iidx = kmeans(features, nc);
    end    
elseif strcmp(method,'ours')

    [iidx, ~,~,~] = dynamicEM(features,...
        sigma_dclustering,0,0);
else 
    error('[Error] calLocalFeatureAggregationAndClustering(): method is incorrect!');
end

idx = zeros(size(Xl));

for ii = 1:length(bds)-1
    idx(bds(ii): bds(ii+1)) = iidx(ii);
end



end


function Xll = movvarcat(Xl, moving_variance_window)

n_samples = length(Xl);
Xll = zeros(size(Xl));
width = round(moving_variance_window/2);
for ii = 1:n_samples
    lb = max(1, ii-width);
    ub = min(n_samples, ii+width);
    seg = Xl(lb:ub);
    n_variations = length(find(diff(seg)));
    Xll(ii) = n_variations/length(seg);
end

end








































