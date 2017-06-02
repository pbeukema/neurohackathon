%% Load data
% Principal features across subjects/timepoints
load('feature_analysis.mat');
% Electrode Distance from fovia
load('distances.mat');
%% Evaluate feature count per timestep
srate  = 256;
nelecs = 128;
ntimepoints = 307;
featureCnt = zeros(nelecs,ntimepoints);
for ts = 1:ntimepoints
    featureSet = [];
    % concatonate features at TS across all students
    for student = 1:16
        featureSet = horzcat(featureSet,which_features{student,ts});
    end
    % count features
    [ai,~,~] = unique(featureSet);
    for ielec = 1:length(ai)
        elecCount = sum(ai(ielec) == featureSet);
        featureCnt(ai(ielec),ts) = elecCount;
    end
end
%%  Sort electrodes by distances
elec = 1:nelecs;
[~,ind] = sort(distance,'ascend');
sortedelec = elec(ind);
sortedFeatureCnt = featureCnt(ind,:);
NsortedFeatureCnt = bsxfun(@times,sortedFeatureCnt,mean(acc_linear,1));
NsortedFeatureCnt = NsortedFeatureCnt./max(max(NsortedFeatureCnt));
%% Visualize results
% sorted
cmap = redgreencmap(max(max(featureCnt)));
hm1 = imagesc(sortedFeatureCnt,'colormap',cmap);
addTitle(hm1,'Principle Electrode Activation (Sorted)');
addXLabel(hm1,'Time (sample)');
addYLabel(hm1,'Electrodes (sorted)');
% sorted and Normalized by classification accuracy
hm2 = imagesc(NsortedFeatureCnt);
title('Normalized Principle Electrode Activation (Sorted)');
xlabel('Time (sample)');
ylabel('Normlized Electrodes (sorted)');
set(gca,'fontsize',15);
% Unsorted
hm3 = HeatMap(featureCnt);
addTitle(hm3,'Principle Electrode Activation (UnSorted)');
addXLabel(hm3,'Time (sampe)');
addYLabel(hm3,'Sorted Electrodes');
%% Number of principle Features
figure; hold on;
plot(mean(n_features)')
ylim = get(gca,'ylim');
stimulusOnset = 0.2/(1/256); %200 ms stim onset
plot([stimulusOnset stimulusOnset],[ylim(1) ylim(end)],'k--')
set(gca,'xlim',[1 307],'fontsize',15);
title('mean Principle Electrode Activation');
xlabel('Time (sample)')
ylabel('Number of electrodes');