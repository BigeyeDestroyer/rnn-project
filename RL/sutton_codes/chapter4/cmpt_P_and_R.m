function [R, P] = cmpt_P_and_R(lambdaRequests, lambdaReturns, max_n_cars, max_num_cars_can_transfer)
% This function computes the rewards and transition probabilities
%
% Inputs
%     lambdaRequests            : lambda for rental possion 
%     lambdaReturns             : lambda for return possion
%     max_n_cars                : maximum cars in each location
%     max_num_cars_can_transfer : maximum cars transfer each night
%
% Outputs
%     R : (max_n_cars + max_num_cars_can_transfer + 1, )
%         Rewards for all the possible states in the morning
%     P : (max_n_cars + max_num_cars_can_transfer + 1, max_n_n_cars + 1)
%         Probabilities from 'morning' to 'night' 

if(nargin == 0)
    lambdaRequests=4; 
    lambdaReturns=2; 
    max_n_cars=20; 
    max_num_cars_can_transfer=5; 
end

% the number of possible cars at any site first thing in the morning: 
% # of cars last night + # of cars moved during the night
nCM = 0: (max_n_cars + max_num_cars_can_transfer);

% Since the states we consider are at night: before car moving
% The Rewards we consider should include all the 
% ** activities in daytime **, from ** the end of yestoday ** to 
% ** the end of today **
R = zeros(1, length(nCM));
for n = nCM
    tmp = 0.0;
    for nreq = 0: (10 * lambdaRequests) % <- a value where the probability of request is very small.
        for nret = 0: (10 * lambdaReturns) % <- a value where the probability of returns is very small.
            tmp = tmp + 10 * min(n + nret, nreq) * poisspdf(nreq, lambdaRequests) ...
                * poisspdf(nret, lambdaReturns);
        end
    end
    R(n+1) = tmp;
end
% R(N) represents the expected rewards 
% we get when having N - 1 cars. 

% Probabilities from the ** state in morning **
% to the state ** at night ** right before moving cars
P = zeros(length(nCM), max_n_cars + 1); 
for nreq = 0: (10 * lambdaRequests) % <- a value where the probability of request is very small. 
    reqP = poisspdf(nreq, lambdaRequests); 
    % for all possible returns:
    for nret = 0: (10 * lambdaReturns) % <- a value where the probability of returns is very small. 
        retP = poisspdf(nret, lambdaReturns); 
        % for all possible morning states: 
        for n = nCM
            sat_requests = min(n, nreq); 
            new_n = max(0, min(max_n_cars, n+nret - sat_requests));
            P(n + 1, new_n + 1) = P(n + 1, new_n + 1) + reqP * retP;
        end
    end
end

if(PLOT_FIGS) 
    figure; 
    imagesc(0: max_n_cars, nCM, P); colorbar; 
    xlabel('num at the end of the day'); 
    ylabel('num in morning'); 
    axis xy; drawnow; 
end





