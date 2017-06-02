function run_classifiers();
%These scripts implement different classifiers
%LDA, Naive Bayes for sake of comparison

%To do: consider PCA/ICA
%To do: add permutation test for chance to confirm if actually 1/6
%To do: add distance from origin as additional feature vector

%Replicate existing analysis with linear classifier. Then shift to Bayes
load('badelectrodes.mat');

origin = [mean(x), mean(y), min(z)];
distance = sqrt((origin(1)-x).^2 + (origin(2)-y).^2 + (origin(3)-z).^2);
n_subs = 16

%Fit linear classifier (LDA)
for sub = 1:n_subs;
    load(sprintf('/Users/plb/Desktop/grover_data/data/s%sdat.mat', int2str(sub)));
 
    del_elec = badelectrodes.(s{sub});% load bad electrodes for deletion 
    parfor timepoint = 1:307;
        class_labels =  repmat([1:6]',80,1);
        this_data = dat(:, timepoint, :);
        this_data = squeeze(this_data)';
        this_data(:,del_elec) = []; % del bad electrodes
        lda = fitcdiscr(this_data(:,[1:size(this_data,2)]),class_labels, 'DiscrimType','linear');
        cvmodel = crossval(lda, 'KFold', 10);
        loss = kfoldLoss(cvmodel);
        acc_linear(sub, timepoint) = 1 - loss;
    end
end
%Plot the results:
subplot(1,2,1)
errorbar([1:307], mean(acc_linear), std(acc_linear)./sqrt(16));



% Fit Naive Bayes Classifier
for sub = 1:16;
    load(sprintf('/Users/plb/Desktop/grover_data/data/s%sdat.mat', int2str(sub)));
    % load bad electrodes. 
    del_elec = badelectrodes.(s{sub});
    
    parfor timepoint = 1:307;
        class_labels =  repmat([1:6]',80,1);
        this_data = dat(:, timepoint, :);
        this_data = squeeze(this_data)';
        this_data(:,del_elec) = []; % del bad electrodes
        lda = fitcnb(this_data(:,[1:size(this_data,2)]),class_labels);
        cvmodel = crossval(lda, 'KFold', 10);
        loss = kfoldLoss(cvmodel);
        acc_nb(sub, timepoint) = 1 - loss;
        
    end
end
subplot(1,2,1)
errorbar([1:307], mean(acc_nb), std(acc_nb)./sqrt(16));


% Implement linear classifier using sequential feature selection:
for sub = 1:16;
    load(sprintf('/Users/plb/Desktop/grover_data/data/s%sdat.mat', int2str(sub)));
    s = fieldnames(badelectrodes);
    del_elec = badelectrodes.(s{sub});  % del bad electrodes. 
    
    parfor timepoint = 1:307;
        class_labels =  repmat([1:6]',80,1);
        this_data = dat(:, timepoint, :);
        this_data = squeeze(this_data)';
        this_data(:,del_elec) = []; % del bad electrodes
        cvp = cvpartition(class_labels,'k',10); 
        [fs,history] = sequentialfs(@classf,this_data(:,[1:size(this_data,2)]),class_labels,'cv',cvp);
        n_features(sub, timepoint) = sum(fs);
        which_features{sub,timepoint} = find(fs);
    end
    
    save(sprintf('sub%s_accuracy.mat', sub), 'which_features', 'n_features', 'acc_linear');
end





    
