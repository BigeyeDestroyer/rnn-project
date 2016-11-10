% Implements Monte Carlo ES (exploring starts) with first visit estimation to
% compute the action-value function for the black jack example.

close all;
clc; 

% a numerical approximation of +Inf
N_HANDS_TO_PLAY=5e6;

rng_seed = rng; % record the current random seed
                % which can be used to initialize 
                % a new seed as: rng(rng_seed); 

% nStates comprises 3 parts:
% player's hand : 12 ~ 21
% dealer's hand : ace ~ 13 (dealer only shows one card)
nStates = prod([21 - 12 + 1, 13, 2]);
nActions = 2; % 0 => stick; 1 => hit 
Q = zeros(nStates, nActions);  % the initial action-value function

% pol_pi = zeros(1, nStates); % our initial policy is to always stick "0"
%pol_pi = unidrnd(2, 1, nStates) - 1; % our initial policy is random 
pol_pi = ones(1, nStates); % our initial policy is to always hit "1"
firstSARewSum = zeros(nStates, nActions); 
firstSARewCnt = zeros(nStates, nActions); 

tic
for hi = 1: N_HANDS_TO_PLAY
    %%%%%%%%%%%%%%%%%%%%%
    %%% Player's Part %%%
    %%%%%%%%%%%%%%%%%%%%%
    stateseen = []; 
    deck = shufflecards(); % 1 x 52 int
    
    % the player gets the first two cards: 
    p = deck(1: 2); 
    deck = deck(3: end); 
    phv = handValue(p);  % player's hand value 
  
    % the dealer gets the next two cards (and shows his first card): 
    d = deck(1: 2); 
    deck = deck(3: end); 
    dhv = handValue(d);  % dealer's hand value
    cardShowing = d(1); 
    
    % disgard states who's initial sum is less than 12 (the decision is always to hit): 
    while(phv < 12) 
        p = [p, deck(1)];  % hit once more 
        deck = deck(2: end); 
        phv = handValue(p);
    end
    
    % accumulate/store the first state seen:
    stateseen(1, :) = stateFromHand(p, cardShowing);
    
    % implement the policy specified by pol_pi (keep hitting till we should "stick"):
    si = 1; 
    polInd = sub2ind([21 - 12 + 1, 13, 2], stateseen(si, 1) - 12 + 1, ...
        stateseen(si, 2), stateseen(si, 3) + 1);
    pol_pi(polInd) = unidrnd(2) - 1;  % FOR EXPLORING STARTS, TAKE AN INITIAL RANDOM POLICY!!! 
    
    pol_to_take = pol_pi(polInd);
    while(pol_to_take && (phv < 22))
        % 'hit' once more
        p = [p, deck(1)]; 
        deck = deck(2: end); 
        phv = handValue(p);
        stateseen(end + 1, :) = stateFromHand(p, cardShowing); 
        
        if(phv <= 21) % only then do we need to querry the next policy action when we have not gone bust
            si = si + 1; 
            %[ stateseen(si,1), stateseen(si,2), stateseen(si,3) ] 
            polInd = sub2ind([21 - 12 + 1, 13, 2], stateseen(si, 1) - 12 + 1, ...
                stateseen(si, 2), stateseen(si, 3) + 1); 
            pol_to_take = pol_pi(polInd);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%
    %%% Dealer's Part %%%
    %%%%%%%%%%%%%%%%%%%%%
    % implement the fixed deterministic policy of the dealer (hit until we have a hand value of 17): 
    while(dhv < 17)
        % 'hit' once more
        d = [d, deck(1)]; 
        deck = deck(2: end); 
        dhv = handValue(d);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Get reward and Update %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % determine the reward for playing this game:
    rew = determineReward(phv,dhv);
    
    % accumulate these values used in computing statistics on this action value function Q^{\pi}: 
    for si = 1: size(stateseen, 1)
        if( (stateseen(si,1)>=12) && (stateseen(si,1)<=21) ) % we don't count "initial" and terminal states
            staInd = sub2ind([21 - 12 + 1, 13, 2], stateseen(si, 1) - 12 + 1, ...
                stateseen(si, 2), stateseen(si, 3) + 1); 
            actInd = pol_pi(staInd) + 1; 
            % 'firstSARewCnt' and 'firstSARewSum' are collected 
            % through all the trials during simulation. 
            firstSARewCnt(staInd, actInd) = firstSARewCnt(staInd, actInd) + 1; 
            firstSARewSum(staInd, actInd) = firstSARewSum(staInd, actInd) + rew; 
            Q(staInd, actInd) = firstSARewSum(staInd, actInd) / firstSARewCnt(staInd, actInd); % <- take the average 
            [dum, greedyChoice] = max(Q(staInd, :));
            pol_pi(staInd) = greedyChoice - 1;
        end
    end
end % end number of hands loop 
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% plot the optimal state-value function V^{*} %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q is with size (num_state, num_act)
mc_value_fn = max(Q, [], 2);
mc_value_fn = reshape(mc_value_fn, [21 - 12 + 1, 13, 2]); 

% Plot 'no_usable_ace'
figure; 
mesh(1: 13, 12: 21, mc_value_fn(:, :, 1)); 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy; 
title('no usable ace'); 
drawnow; 
% fn = sprintf('state_value_fn_nua_%d_mesh.eps', N_HANDS_TO_PLAY); 
% saveas(gcf, fn, 'eps2'); 

% Plot 'usable_ace'
figure; 
mesh(1: 13, 12: 21, mc_value_fn(:, :, 2)); 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy; 
title('a usable ace'); 
drawnow; 
% fn = sprintf('state_value_fn_ua_%d_mesh.eps', N_HANDS_TO_PLAY); 
% saveas(gcf, fn, 'eps2'); 

% imagesc 'no_usable_ace'
figure;
imagesc(1: 13, 12: 21, mc_value_fn(:, :, 1));
caxis([-1, +1]); 
colorbar; 
xlabel('dealer shows');
ylabel('sum of cards in hand'); 
axis xy; 
title('no usable ace'); 
drawnow; 
% fn = sprintf('state_value_fn_nua_%d_img.eps', N_HANDS_TO_PLAY); 
% saveas(gcf, fn, 'eps2');

% imagesc 'usable_ace'
figure;
imagesc(1: 13, 12: 21, mc_value_fn(:, :, 2)); 
caxis([-1, +1]); 
colorbar; 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy; 
title('a usable ace'); 
drawnow; 
% fn = sprintf('state_value_fn_ua_%d_img.eps', N_HANDS_TO_PLAY); 
% saveas(gcf, fn, 'eps2'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% plot the optimal policy: %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pol_pi = reshape(pol_pi, [21 - 12 + 1, 13, 2]); 

% Plocy without ace
figure; 
imagesc(1: 13, 12: 21, pol_pi(:, :, 1)); 
colorbar; 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy; 
title('no usable ace'); 
drawnow; 
% fn = sprintf('bj_opt_pol_nua_%d_image.eps', N_HANDS_TO_PLAY); 
% saveas(gcf, fn, 'eps2'); 

% Plocy with ace
figure; 
imagesc(1: 13, 12: 21, pol_pi(:, :, 2)); 
colorbar; 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy; 
title('a usable ace'); 
drawnow; 
% fn = sprintf('bj_opt_pol_ua_%d_mesh.eps', N_HANDS_TO_PLAY); 
% saveas(gcf, fn, 'eps2'); 
