clc
clearvars
clear all

%% Read Data File
load 'DOE_permutations.mat' 'P_analysis'
log_filename = 'varout_opt_log.csv';
% data = readmatrix(, 'HeaderLines',1);
data = importdata(['Optimization_studies/',log_filename],',',1);

Index_a = find(strcmp(data.colheaders,'n_f_th')); % attribute
Index_c = find(strcmp(data.colheaders,'weight')); % cost
resiliance = data.data(:,Index_a);
weight = data.data(:,Index_c);

bb_extra_param = {P_analysis, weight, resiliance};

%%

x0 = [1, 0, 1, -1, -1]';

neighbors(x0)

lb = [-100, -100, -100, -100, -100]';
ub = [100, 100, 100, 100, 100]';
opts = nomadset('display_degree',2,'display_all_eval',1,'history_file','history.txt','max_bb_eval',500,'bb_output_type','OBJ','f_target',-4,'bb_input_type','[C C C C C]','neighbors_mat',@neighbors); 

% Start optimization
[x,fval] = nomad(@fun,x0,lb,ub,opts,bb_extra_param);
