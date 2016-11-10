function [st] = stateFromHand(hand, cardShowing)
% Returns the state (a three vector of numbers for a given hand of cards): 
% [players current sum, dealar showing card, usable ace] 
% 
% Inputs:
%     hand        : (1, n_h), each a card number
%     cardShowing : (1, n_c), each a card number of the dealer
% 
% Outpus:
%     st : a three vector, [player_sum, dealer_sum, usable_ace]
% 
% Cards are enoumerated 1:52, such that
% 
% 1:13 => A, 2, 3, ..., 10, J, Q, K      (of C)
% 14:26                                  (of D)
%                                        (of H)
%                                        (of S)

[hv, usableAce] = handValue(hand);

cardShowing = mod(cardShowing - 1, 13) + 1; 

st = [hv, cardShowing, usableAce]; 
return; 