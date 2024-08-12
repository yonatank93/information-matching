% Find the optimal PMU placements using the MISDP version of information-matching method.
% We will use this result as an initial guess when we solve the regular SDP version of the information-matching.
% 
% For the information on how to formulate the MISDP problem, see the following webpage:
% https://www.opt.tu-darmstadt.de/scipsdp/#download


clc
clear

% SETUP
nbuses = 39;
nparams = 2 * nbuses;                   % Number of parameters
buses = 1:nbuses;                       % All buses

% CANDIDATE CONFIGURATIONS
exclude_buses = [1, 9, 30:38];        % Buses to exclude from the candidates

configs = [];
for bus=buses
    if ~ismember(bus, exclude_buses)
        configs = [configs bus];
    end
end
nconfigs = length(configs);       % Number of candidate configurations
xtype = repelem('B', nconfigs);     % Type of the optimizing variables

% ITERATION
lam_tol_list = linspace(4, 6, 21) * 1e-2;

for j=1:length(lam_tol_list)
    % DEFINE THE TARGET FIM
    % lam_tol = 5e-2;% If lambda is too small, then the optimal result is all zero
    lam_tol = lam_tol_list(j)
    fimJ = eye(nparams) * lam_tol;
    sdp = [fimJ(:)];% Additional constant matrix needs to be placed at the beginning

    % LOAD THE FIMS OF THE CONFIGURATIONS
    for i=1:nconfigs
        filepath = sprintf('FIMs/fim_bus%i.csv', configs(i));
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
    opts = optiset('solver', 'SCIPSDP', 'display', 'off');
    misdp = opti('f', f, 'sdcone', sdp, 'xtype', xtype, 'options', opts);
    [x, fval, exitflag, info] = solve(misdp);


    % POST-PROCESSING
    % Retrieve the locations of optimal buses
    opt_idx = find(x>1e-4);
    opt_buses = sort(configs(opt_idx))
    nopt_buses = length(opt_buses);

    if nopt_buses == 8                  % We need at least 8 buses to identify the entire states
        % Write to a file
        writematrix(opt_buses, 'data/misdp_optimal_buses.txt', 'Delimiter', ',')
        break
    end
end
