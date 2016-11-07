function [v_tmp] = jcr_rhs_state_value_bellman(na, nb, ntrans, V, gamma, Ra, Pa, Rb, Pb, max_num_cars_can_transfer)
% RHS_STATE_VALUE_BELLMAN - computes the right hand side of the bellman equation
% We have to consider the possible number of rentals at sites A/B
%                 and the possible number of returns at sites A/B
% This function is for policy-evaluation
% Moreover, we can use it for value-iteration
%
% Inputs
%      na     : number of cars at night for LOC_a, before moving
%      nb     : number of cars at night for LOC_b, before moving
%      ntrans : number of tranfered cars from a to b 
%      V      : (max_n, max_n), value matrix
%      gamma  : discount factor 
%      Ra     : (max_n + max_transfer + 1, ), rewards for cars' 
%               states in the morning
%      Pa     : (max_n + max_transfer + 1, max_n + 1), probabilities
%               from ** state in the morning ** to ** state at night **
%      Rb     : similar for LOC_b
%      Pb     : similar for LOC_b
% 
%      max_num_cars_can_transfer : int
% 
% Outputs
%     v_tmp   : The updated state-value  

% the maximum number of cars at each site (assume equal): 
max_n_cars = size(V, 1) - 1;

% restrict this action: 
%  -nb < ntrans < na
ntrans = max(-nb, min(ntrans, na)); 
% -max_num_cars_can_transfer < ntrans < +max_num_cars_can_transfer
ntrans = max(-max_num_cars_can_transfer, min(+max_num_cars_can_transfer, ntrans));

% Rewards for moving cars
v_tmp = -2 * abs(ntrans); 
na_morn = na - ntrans; % cars in the morning
nb_morn = nb + ntrans; 

% V(s) = \sum_{s'}P_{ss'}^{pi(s)}[R_{ss'}^{a} + gamma * V(s')]
for nna = 0: max_n_cars
  for nnb = 0: max_n_cars
    pa = Pa(na_morn + 1, nna + 1); 
    pb = Pb(nb_morn + 1, nnb + 1); 
    v_tmp = v_tmp + pa * pb * (Ra(na_morn + 1) + Rb(nb_morn + 1) + gamma * V(nna + 1, nnb + 1)); 
  end
end