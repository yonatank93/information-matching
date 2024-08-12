% Find the optimal PMU placements using the MISDP version of information-matching method.
% We will use this result as an initial guess when we solve the regular SDP version of the information-matching.
% 
% For the information on how to formulate the MISDP problem, see the following webpage:
% https://www.opt.tu-darmstadt.de/scipsdp/#download


clc
clear

% SETUP
nbuses = 14;
nparams = 2 * nbuses;                   % Number of parameters
buses = 1:nbuses;                       % All buses

% CANDIDATE CONFIGURATIONS
area_file = 'models/data/area_partition.json';
json_str = fileread(area_file);
area_data = jsondecode(json_str);
area_buses = area_data.AreaB;            % Buses within the area of interest

configs = transpose(area_buses);
nconfigs = length(configs);       % Number of candidate configurations
xtype = repelem('B', nconfigs);     % Type of the optimizing variables

% DEFINE THE TARGET FIM
lam_tol = 1e-4;                         % If lambda is too small, then the optimal result is all zero
diag_fimJ = ones(nparams, 1) * lam_tol; % Initialize the diagonal values of target FIM
for bus=buses% Set the diagonal element that don't correspond to the candidates to zero
    if ~ismember(bus, configs)
        diag_fimJ(2*bus) = 0;
        diag_fimJ(2*bus-1) = 0;
    end
end
fimJ = diag(diag_fimJ);
sdp = [fimJ(:)];                       % Additional constant matrix needs to be placed at the beginning

% LOAD THE FIMS OF THE CONFIGURATIONS
for i=1:nconfigs
    filepath = sprintf(string('FIMs/fim_bus%i.csv'), configs(i));
    Im = csvread(filepath);
    sdp = [sdp Im(:)];
end
    

% FORMULATE THE PROBLEM
% opti('f', f, 'sdcone', sdp, 'xtype', xtype)
% Assume variable to optimize is x
% ('f', f) -> Linear objective function, minimize f.x
% ('sdcone', sdp) -> Semi-definite constraint, sdp.x >> 0
% ('xtype', xtype) -> Constraint on the type of x, the input value should be a series of string: I=integer, C=continuous, B=binary
f = ones(nconfigs, 1);                  % Represents the objective function to minimize, min f.x
misdp = opti('f', f, 'sdcone', sdp, 'xtype', xtype)
[x, fval, exitflag, info] = solve(misdp)


% POST-PROCESSING
% Retrieve the locations of optimal buses
opt_idx = find(x>1e-4);
opt_buses = configs(opt_idx)
% Write to a file
writematrix(opt_buses, 'data/misdp_optimal_buses_areaB.txt', 'Delimiter', ',')
