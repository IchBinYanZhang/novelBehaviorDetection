function idx = fun_feature_aggregation_kernelizedcut(X,C,sigma_dclustering, cut_method)

%%% ------------------kernelized cut-based pooling------------------
%%% Input:  X - input feature array. Samples are arranged row-wisely.
%%%         C - cluster centers for feature encoding. The output from the
%%%             first DC
%%%         sigma_dclustering - the radius of the second DC module
%%%         cut_method - 'sequential' or 'batch'
%%% Output: idx - the final parsing result, arranged in a 1D sequence



if ~strcmp(cut_method, 'sequential') && ~strcmp(cut_method, 'batch')
    error('[error] fun_feature_aggregation_kernelizedcut: cut_method must be either sequential or batch!');
end


W = 100;S = 1; alpha = 1.5; % if cmumad
% W = 30; S=1; alpha = 1.2; % if tumkitchen
% W = 30; S=1; alpha = 1.2; % if BOMNI


n_frames = size(X,1);


%% step1: sliding window method to get the encoded pattern
disp('--- sliding window aggregation');

%%% use the time window to aggregate features
features = zeros(n_frames,size(C,1));
for ii = 1:S:n_frames
    lb = max(1,ii-round(W/2));
    ub = min(n_frames,ii+round(W/2));
    XX = X(lb : ub, :);
    dist = pdist2(XX, C);
    ff = exp(-0.1*dist)./ repmat(sum(exp(-0.1*dist),2), 1, size(C,1));
    features(ii,:) = sum(ff,1);
    features(ii,:) = features(ii,:)/norm(features(ii,:));
end

%% step2: apply the kernelized cut to derive the labels
fprintf('--- spatiotemporal kernelized cut, %s\n',cut_method);
[~,cut_locs] = calKernelizedTemporalSegment(features, alpha*W, 0.01, 1, cut_method);

%%% re-agg the encoded features depending on the cut_locs
cut_locs = [1 cut_locs n_frames];
features = zeros(length(cut_locs)-1,size(C,1));
for ii = 1:length(cut_locs)-1
    lb = max(1,cut_locs(ii));
    ub = min(n_frames,cut_locs(ii+1));
    XX = X(lb : ub, :);
    dist = pdist2(XX, C);
    ff = exp(-0.1*dist)./ repmat(sum(exp(-0.1*dist),2), 1, size(C,1));
    features(ii,:) = sum(ff,1);
    features(ii,:) = features(ii,:)/norm(features(ii,:));
end


%% step3: apply dynamic clustering on the new action patterns
disp('--- dynamic clustering on new action patterns')

[iidx, ~] = incrementalClustering(features, 1,sigma_dclustering,0);
% [iidx, ~,~,~] = dynamicEM(features,...
%         sigma_dclustering,0,0);

idx = zeros(n_frames,1);

for ii = 1:length(cut_locs)-1
    lb = max(1,cut_locs(ii));
    ub = min(n_frames,cut_locs(ii+1));
    idx(lb : ub, :) = iidx(ii);
end

end

