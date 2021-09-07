initProject;
load('AUs_select.mat');
load('CK_8066_Data_forTesting.mat');
testingdata = Features;
AU_label = AU_matrix_binary;

load('ModelMaster.mat');
load('MasterMatrix.mat');

f1 = zeros(1,26);
for m1 = select_disfa_au%(1:10)
    m1
    test_label = AU_label(:,m1)';
    test_label = test_label + 1;
    [rate,classEstimate,prob] = KSDA_Testing_MasterModel(testingdata,test_label,modelKSDA,masterMatrix,m1,'mahal');
    stats = confusionmatStats(test_label,classEstimate);
    f1(1,m1) = stats.Fscore(2);
end
    
