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
nconfigs = nbuses;                % Number of candidate configurations
xtype = repelem('B', nconfigs);          % Type of the optimizing variables

% DEFINE THE TARGET FIM
lam_tol = 1e-4;                         % If lambda is too small, then the optimal result is all zero
fimJ = eye(nparams) * lam_tol;
sdp = [fimJ(:)];                       % Additional constant matrix needs to be placed at the beginning

% LOAD THE FIMS OF THE CONFIGURATIONS
for ii=1:nbuses
    filepath = sprintf(string('FIMs/fim_bus%i.csv'), ii);
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
opt_buses = find(x);
% Write to a file
writematrix(transpose(opt_buses), 'data/misdp_optimal_buses.txt', 'Delimiter', ',')
