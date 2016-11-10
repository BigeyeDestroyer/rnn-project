function [hv, usableAce] = handValue(hand)
% Get the values for cards in hand
% which is to be further processed 
% by the 'stateFromHand' function
%
% Inputs:
%     hand : (1, n), each a card number 
% 
% Outputs:
%     hv        : int, hand value
%     usableAce : bool, whether ace is used

% compute 1: 13 indexing for each card: 
values = mod(hand - 1, 13) + 1; 

% map face cards (11, 12, 13)'s to 10's: 
values = min(values, 10);
sv = sum(values); 
% Promote soft ace
if (any(values == 1)) && (sv <= 11)
   sv = sv + 10;
   usableAce = 1; 
else
   usableAce = 0; 
end

hv = sv; 