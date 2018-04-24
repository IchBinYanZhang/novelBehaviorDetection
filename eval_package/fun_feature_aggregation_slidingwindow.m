function iidx = fun_feature_aggregation_slidingwindow(X,C,sigma_dclustering)


%%% sliding window method
W = 100;S = 1; 
n_frames = size(X,1);


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


%  [iidx, ~,~,~] = dynamicEM(features,...
%         sigma_dclustering,0,0);
[iidx, ~] = incrementalClustering(features, 30,sigma_dclustering,0);

% iidx = spectralClustering(features,1,sigma_dclustering);

end

    
