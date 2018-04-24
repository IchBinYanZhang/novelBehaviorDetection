function idx = fun_feature_aggregation(X,Xl,C,method,varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Motion energy-based pooling
% This function consists of two steps: feature aggregation and clustering.
% Feature aggregation = feature encoding + temporal pooling.
% Stationary regions are moving regions are separately aggregated.
% stationary states and moving action patterns are separately clustered.
% Action parsing is combinition of the two.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset=varargin{1};

if strcmp(dataset, 'CMUMAD')

    %%% set params CMUMAD%%%
    moving_variance_window = 60;
    gaussian_filter_sigma = 12.5;
    min_peak_distance = 60;
    sigma_dclustering = 1e-8;
    sigma_dclustering_stationary = 100;
    peak_width_weight = 0.65;
    W_dclustering = 1;
elseif strcmp(dataset, 'TUMKitchen')
    %%% set params TUMKitchen%%%
    %%% jointlocation, rightarms + leftarms %%%
    moving_variance_window = 25;
    gaussian_filter_sigma = 6.5;
    min_peak_distance = 25;
    sigma_dclustering = 0.03;%arms 1 for torso
    sigma_dclustering_stationary =0.03; %arms, 1 for torso
    peak_width_weight = 0.5;
    W_dclustering = 5;
    %%%
elseif strcmp(dataset,'HDM05') % to tune
    moving_variance_window = 30;
    gaussian_filter_sigma = 6.5;
    min_peak_distance = 30;
    sigma_dclustering = 0.05;
    sigma_dclustering_stationary = 0.05;
    peak_width_weight = 1;

elseif strcmp(dataset,'BOMNI')
    %%% set params BOMNI scenario1 %%%
    moving_variance_window = 30;
    gaussian_filter_sigma = 6.5;
    min_peak_distance = 30;
    sigma_dclustering = 0.1;
    sigma_dclustering_stationary = 1e-5;
    peak_width_weight = 0.3;
%%%
end



%% aggregate action patterns
%%% moving variance computation
Xll = movvarcat(Xl, moving_variance_window);

%%% find the peaks in the label space to determine the time window
labels = imgaussfilt(Xll,gaussian_filter_sigma);
[pks_label,locs_label, pks_width] = findpeaks(labels, 'MinPeakDistance',min_peak_distance);

%%% use the time window to aggregate features
features = zeros(length(pks_label), size(C,1));
for pp = 1:length(pks_label)
    lb = max(1,locs_label(pp)-round(peak_width_weight*pks_width(pp)));
    ub = min(size(X,1), locs_label(pp)+round(peak_width_weight*pks_width(pp)));
    XX = X(lb : ub, :);
    dist = pdist2(XX, C);
    ff = exp(-0.1*dist)./ repmat(sum(exp(-0.1*dist),2), 1, size(C,1));
    features(pp,:) = sum(ff,1);
    features(pp,:) = features(pp,:)/(1e-6+norm(features(pp,:),2));
end


if strcmp(method, 'kmeans')
    nc = varargin{2};
    if size(features,1) <= nc
        iidx = 1:size(features,1);
    else
        iidx = kmeans(features, nc);
    end    
elseif strcmp(method,'ours')
%     [iidx, ~,~,~] = dynamicEM(features,...
%         sigma_dclustering,0,0);
      [iidx, ~] = incrementalClustering(features, W_dclustering,sigma_dclustering,0);

else 
    error('[Error] calLocalFeatureAggregationAndClustering(): method is incorrect!');
end

iidx = iidx+1;
idx = zeros(size(Xl));
for pp = 1:length(pks_label)
    lb = max(1,locs_label(pp)-round(peak_width_weight*pks_width(pp)));
    ub = min(size(X,1), locs_label(pp)+round(peak_width_weight*pks_width(pp)));
    idx(lb:ub)=iidx(pp);
end


%% aggregate stationary body configs, checking number of different body configs
% if ~strcmp(dataset, 'CMUMAD')
    %%% in case of CUMMAD, we only interests on moving actions
idx2 = idx;
idx2(idx2~=0) = 1;
idx2 = 1-idx2;
idx3 = bwlabel(idx2);
n_regions = length(unique(idx3))-1;
features_stationary = zeros(n_regions, size(C,1));

for kk = 1:n_regions
    XX = X(idx3==kk, :);
    dist = pdist2(XX, C);
    ff = exp(-0.1*dist)./ repmat(sum(exp(-0.1*dist),2), 1, size(C,1));
    features_stationary(kk,:) = sum(ff,1);
    features_stationary(kk,:) = features_stationary(kk,:)/(1e-6+norm(features_stationary(kk,:),2));
end

if strcmp(method, 'kmeans')
    nc = varargin{3};
    if size(features,1) <= nc
        iidx = 1:size(features_stationary,1);
    else
        iidx = kmeans(features_stationary, nc);
    end

elseif strcmp(method,'ours')

%     [iidx, ~,~,~] = dynamicEM(features_stationary,...
%         sigma_dclustering_stationary,0,0);
      [iidx, ~] = incrementalClustering(features_stationary, W_dclustering,sigma_dclustering_stationary,0);

else 
    error('[Error] calLocalFeatureAggregationAndClustering(): method is incorrect!');
end

max_label = max(idx)+3;


for kk = 1:n_regions
    idx(idx3==kk) = iidx(kk)-max_label;
end
% end
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








































