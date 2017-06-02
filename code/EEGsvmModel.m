% Load electrode info
load('electrode_arrays.mat'); % electrode mapping for LD1,LD2,LD3,LD4,HD
load('trialorder.mat');       % trialorder
load('badelectrodes.mat');    % badelectrodes
% Info vars
nsubj = 16;
nelec = 128;
ntrials = 480;
ntimepoints = 307;
srate   = 256;      % hz
beforeOnset = 200;  % msec
afterOnset  = 1000; % msec
%% Load data for subject uinput (electrode chan,time instance,density)
depdata = repmat(trialorder,[nsubj 1]);
curdir  = pwd;
classes = {'ld1','ld2','ld3','ld4','hd'};
classrec = cell(length(classes));
KclassificationAccuracy      = zeros(ntimepoints,nsubj,length(classes));
KclassClassificationAccuracy = zeros(ntimepoints,5,6,length(classes));
alldat = zeros(16,nelec,ntimepoints,ntrials);
%% load all subject dat first
for subject = 1:16
    sid     = sprintf('s%d',subject);
    fprintf('Subject ID: %s\n',sid);
    load(horzcat(curdir,filesep,'data',filesep,sid,'dat.mat'));
    % eliminate bad electrodes
    sidBadElectrodes = false(1,nelec);
    sidBadElectrodes(badelectrodes.(sid)) = true;
    dat(sidBadElectrodes,:,:) = NaN;
    alldat(subject,:,:,:) = dat;
end    
%%  Multi-Class SVM 
% Loop through each timepoint and record Classification accuracy (90:10)
for subject = 1:16 
    t = templateSVM('Standardize',1,'KernelFunction','linear');
    for DensityRecClass = 1:5 % USER INPUT [CLASS DENSITY RECORDINGS]
        fprintf('Subject ID: %d,Density Class: %d\n',subject,DensityRecClass);
        % Separate Electrodes based EEG Density recording class
        dclass    = electrode_arrays.(classes{DensityRecClass});
        inputData = alldat(subject,dclass,:,:);
        for i = 1:ntimepoints
            %fprintf('iter: %d\n',i);
            tempmat = reshape(inputData(1,:,i,:),size(inputData,2),ntrials);
            tempmat = tempmat(all(~isnan(tempmat),2),:); % Bad array removed from analysis
            recMDL2  = fitcecoc(tempmat',trialorder,'KFold',2,'learners',t);
            %CMdl     = recMDL.Trained{1};
            %testInds = test(recMDL.Partition);
            %XTest = tempmat(:,testInds);
            %YTest = trialorder(testInds);
            %yhat  = predict(CMdl,XTest');
            yhat = kfoldPredict(recMDL2);
            KclassificationAccuracy(i,subject,DensityRecClass) = sum(yhat==trialorder)/length(yhat);
            for j = 1:6
                Xclass = tempmat(:,trialorder==j);
                Yclass = trialorder(trialorder==j);
                %yclasshat = predict(CMdl,Xclass');
                yclasshat = yhat(trialorder==j);
                KclassClassificationAccuracy(i,subject,j,DensityRecClass) = sum(yclasshat==Yclass)/length(yclasshat);
            end
        end
    end
end
%% Visualizing results (validation error)
figure; hold on;
for densityRecClass = 1:5
    plot(mean(classificationAccuracy(:,:,densityRecClass),2))
end
% bar chart of svm class accuracy
recStimClass = zeros(6,5);
recStimClassSTD = zeros(6,5);
for densityRecClass = 1:5
    for densityStimClass = 1:6
        [recStimClass(densityStimClass,densityRecClass),ind] = max(mean(classClassificationAccuracy(:,:,densityStimClass,densityRecClass),2));
        recStimClassSTD(densityStimClass,densityRecClass) = std(classClassificationAccuracy(ind,:,densityStimClass,densityRecClass));
    end
end
val = repmat(1:6,[5,1]);
val(1,:) = val(1,:)-0.3;
val(2,:) = val(2,:)-0.16;
val(4,:) = val(4,:)+0.16;
val(5,:) = val(5,:)+0.31;
figure; hold on;
bid = bar(1:6,recStimClass(1:6,:));
errorbar(val',recStimClass(1:6,:),recStimClassSTD(1:6,:),'k.')
set(gca,'xtick',1:6,'xlim',[0 7])
% label bar chart
ylabel('Classification Accuracy');
xticklbl = {'Left LSD','Left MSD','Left HSD','Right LSD','Right MSD','Right HSD'};
set(gca,'ylim',[0 1],'xticklabel',xticklbl,'XTickLabelRotation',45,'fontsize',15);
legend(classes);
title('SVM Classification Accuracy with linear kernel')