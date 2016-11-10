% Uses first-visit monte-carlo evaluation to compute
% the value function for the black jack example.

%close all;
clc; 

%N_HANDS_TO_PLAY=1e4;
%N_HANDS_TO_PLAY=2*1e4;
%N_HANDS_TO_PLAY=2*5e5;
N_HANDS_TO_PLAY=5e6;

% nStates comprises 3 parts:
% player's hand: 12 - 21
% dealer's hand: ace - 13 (dealer only shows one card)
nStates = prod([21 - 12 + 1, 13, 2]);
allStatesRewSum = zeros(nStates, 1);
allStatesNVisits = zeros(nStates, 1); 

tic
for hi = 1: N_HANDS_TO_PLAY
    if mod(hi, 10000) == 0
        fprintf('%d/%d-th hand ...\n', hi, N_HANDS_TO_PLAY);
    end
    stateseen = []; 
    deck = shufflecards(); % (1, 52) 
    
    % the player gets the first two cards: 
    p = deck(1: 2); 
    deck = deck(3: end); 
    % the dealer gets the next two cards and shows his first card: 
    d = deck(1: 2); 
    deck = deck(3: end); 
    dhv = handValue(d); 
    cardShowing = d(1); 
    
    % accumulate/store the first state seen: 
    % Each row a state 
    stateseen(1, :) = stateFromHand(p, cardShowing); 
    phv = handValue(p);
    
    % implement the policy of the player (hit until we have a hand value of 20 or 21): 
    while(phv < 20)
        p = [p, deck(1)]; % HIT, p is a (1, n_p) matrix
        deck = deck(2: end); 
        stateseen(end+1, :) = stateFromHand(p, cardShowing); 
        phv = handValue(p);       
    end
    
    % implement the policy of the dealer (hit until we have a hand value of 17):
    while(dhv < 17)
        d = [d, deck(1)]; 
        deck = deck(2: end); % HIT, d is a (1, n_d) matrix
        dhv = handValue(d); 
    end
    % determine the reward for playing this game:
    rew = determineReward(phv, dhv);
    
    % accumulate these values used in computing global statistics: 
    for si = 1: size(stateseen, 1)
        % we don't count "initial" and terminal states
        if((stateseen(si, 1) >= 12) && (stateseen(si, 1) <= 21)) 
            indx = sub2ind([21 - 12 + 1, 13, 2], stateseen(si, 1) - 12 + 1, stateseen(si, 2), stateseen(si, 3) + 1); 
            allStatesRewSum(indx)  = allStatesRewSum(indx) + rew; 
            allStatesNVisits(indx) = allStatesNVisits(indx) + 1; 
        end
    end
end % end number of hands loop
toc

mc_value_fn = allStatesRewSum ./ allStatesNVisits;

mc_value_fn = reshape(mc_value_fn, [21 - 12 + 1, 13, 2]); 

% plot the various graphs:  
% 1) value function with no_usable_ace
figure; 
mesh(1: 13, 12: 21, mc_value_fn(:, :, 1)); 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy;
title('no usable ace'); 
drawnow; 
% 2) value function with usable_ace
figure; 
mesh(1: 13, 12: 21, mc_value_fn(:, :, 2)); 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy;
title('a usable ace'); 
drawnow; 
% 3) colorbar for no_usable_ace
figure;
imagesc(1: 13, 12: 21, mc_value_fn(:, :, 1)); 
caxis([-1, +1]); 
colorbar; 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy; 
title('no usable ace'); 
drawnow; 
% 4) colorbar for usable_ace
figure;
imagesc(1: 13, 12: 21, mc_value_fn(:, :, 2)); 
caxis([-1, +1]); 
colorbar; 
xlabel('dealer shows'); 
ylabel('sum of cards in hand'); 
axis xy; 
title('a usable ace'); 
drawnow;