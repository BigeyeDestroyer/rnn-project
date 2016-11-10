function [deck] = shufflecards()
% Returns a shuffled deck of cards
% Outputs:
%      deck : (1, 52), each number indicates a card

deck = randperm( 52 ); 