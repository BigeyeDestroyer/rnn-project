function [rew] = determineReward(phv, dhv)
% This function judges the winner. 
% 
% Inputs:
%     phv : player's hand value
%     dhv : dealer's hand value
% 
% Outputs:
%     rew : reward the player get

if (phv > 21) % player went bust
    rew = -1;
    return; 
end

if (dhv > 21) % dealer went bust
    rew = +1; 
    return; 
end

% No one bust here
if(phv == dhv) % a tie
    rew = 0; 
    return;
end

if(phv > dhv) % the larger hand wins
    rew = +1; 
else
    rew = -1;
end