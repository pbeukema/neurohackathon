
% This function implements Representational Similarity Analysis on Glover EEG data
% Generate candidate RDMs around fovea and around furthest pole
% Attempts to answer question - How separable are classes as you 
% move away from fovea?

load('badelectrodes.mat');
s = fieldnames(badelectrodes);

fovea_elec = 81; % for posterity
load('distances.mat'); %generated using 3 dimensional distance to origin (computed as most posterior node).
[h,p]=sort(distance);  %sort electrodes based on distance from origin.

figure;
for which_electrode = 1:length(p);
    this_electrode = p(which_electrode);
    for sub = 1:16;
        load(sprintf('/Users/plb/Desktop/grover_data/data/s%sdat.mat', int2str(sub)));
        % load bad electrodes. 
        del_elec = badelectrodes.(s{sub});
        class_labels =  repmat([1:6]',80,1);
        this_data = dat(:, :, :);
        this_data(del_elec, :, :) = []; % del bad electrodes
        
        %if its a bad electrode, fill with nan, and pass to the next one:
        if any(del_elec==this_electrode);
            avg_dist(:,:, sub) = nan(6,6);
            continue;
        end
        
        %Grab data from fovea electrode -
        lowl = squeeze(this_data(this_electrode, :, [1:6:480]))';
        midl = squeeze(this_data(this_electrode, :, [2:6:480]))';
        highl = squeeze(this_data(this_electrode, :, [3:6:480]))';
        lowr = squeeze(this_data(this_electrode, :, [4:6:480]))';
        midr = squeeze(this_data(this_electrode, :, [5:6:480]))';
        highr = squeeze(this_data(this_electrode, :, [6:6:480]))';
        
        %Gen big matrix of all pairs
        all_data(1,:,:) = reshape(lowl, 1,307,80);
        all_data(2,:,:) = reshape(midl, 1,307,80);
        all_data(3,:,:) = reshape(highl, 1,307,80);
        all_data(4,:,:) = reshape(lowr, 1,307,80);
        all_data(5,:,:) = reshape(midr, 1,307,80);
        all_data(6,:,:) = reshape(highr, 1,307,80);

        %Compute cross-validated mahalanobis distances:
        num_seqs = nchoosek([1:80],2);
        for pair = 1:size(num_seqs,1);
            for seq1 = 1:6;
                for seq2 = 1:6;
                    x = num_seqs(pair,:);
                    foldm = (all_data(seq1,:,x(1)) - all_data(seq2,:,x(1)));
                    foldl = (all_data(seq1,:,x(2)) - all_data(seq2,:,x(2)));
                    d(seq1,seq2,pair) = foldm*foldl';
                end
            end
        end    
        %Compute RDM:
        dhat = mean(d,3);

        %normalize for consistency across subjects:
        normd = dhat - min(dhat(:));
        d_final = normd ./ max(normd(:));
        avg_dist(:,:, sub) = d_final;
    end
    mean_distances = nanmean(avg_dist,3);
    bad_variable_naming(:,:,this_electrode) =  mean_distances;
    subplot(8,16,this_electrode) = imagesc(mean_distances);
    doubleDM(1,this_electrode) = mean(mean_distances(:));   
end

    