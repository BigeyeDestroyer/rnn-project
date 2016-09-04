nB = 2000;
nA = 10;
nP = 2000;
sigmaReward = 1.0;
sigmaRW = 1.0;

%-----
% 1) Epsilon greedy (eps = 0.1) algorithm, using sample average
% 2) Epsilon greedy (eps = 0.1) algorithm with constant geometric 
%    factor alpha = 0.1
%-----

tEps = 0.1;
alpha = 0.1;

avgReward = zeros(2, nP);
perOptAction = zeros(2, nP);

allRewards_Eps = zeros(nB, nP);
pickedMaxAction_Eps = zeros(nB, nP);

allRewards_Alp = zeros(nB, nP);
pickedMaxAction_Alp = zeros(nB, nP);

for bi = 1: nB % pick a bandit 
    fprintf('%d/%d bandit ...\n', bi, nB);
    % Generate teh TRUE reward Q^{\star}
    qStarMeans = ones(1, nA);
    
    qT_Eps = zeros(1, nA); % <- initialize value function to zero 
    qN_Eps = zeros(1, nA); % <- keep track of the number of draws on each arm 
    
    qT_Alp = zeros(1, nA); 
    
    for pi = 1: nP % make a play 
        if (rand(1) <= tEps) % explore 
            [~, arm_Eps] = histc(rand(1), linspace(0, 1 + eps, nA + 1));
            arm_Alp = arm_Eps;
        else % exploit 
            [~, arm_Eps] = max(qT_Eps);
            [~, arm_Alp] = max(qT_Alp);
        end
        
        % determine if the arm selected is the best possible 
        [~, bestArm] = max(qStarMeans);
        if (arm_Eps == bestArm)
            pickedMaxAction_Eps(bi, pi) = 1;
        end
        if (arm_Alp == bestArm)
            pickedMaxAction_Alp(bi, pi) = 1;
        end
        
        % get the reward from the drawing arm 
        reward_Eps = qStarMeans(arm_Eps) + sigmaReward * randn(1);
        reward_Alp = qStarMeans(arm_Alp) + sigmaReward * randn(1);
        
        allRewards_Eps(bi, pi) = reward_Eps;
        allRewards_Alp(bi, pi) = reward_Alp;
        
        % update qT, qN incrementally 
        qT_Eps(arm_Eps) = qT_Eps(arm_Eps) + ...
            (reward_Eps - qT_Eps(arm_Eps)) / (qN_Eps(arm_Eps) + 1);
        qN_Eps(arm_Eps) = qN_Eps(arm_Eps) + 1;
        
        qT_Alp(arm_Alp) = qT_Alp(arm_Alp) + ...
            alpha * (reward_Alp - qT_Alp(arm_Alp));
        
        % To be non-stationary, qStarMeans follows a random walk 
        qStarMeans = qStarMeans + sigmaRW * randn(size(qStarMeans));
    end
end

avgReward(1, :) = mean(allRewards_Eps, 1);
avgReward(2, :) = mean(allRewards_Alp, 1);

perOptAction(1, :) = mean(pickedMaxAction_Eps, 1);
perOptAction(2, :) = mean(pickedMaxAction_Alp, 1);

% average rewards plot 
figure; 
hold on; 
all_hnds = plot(1: nP, avgReward);
legend(all_hnds, {'eps:0.1', 'fixed step'}, 'Location', 'SouthEast'); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('Average Reward'); 
title(['sigmaRW=', num2str(sigmaRW)]); 

% average optimal action rate plot 
figure; 
hold on; 
all_hnds = plot(1: nP, perOptAction);
legend(all_hnds, {'eps:0.1', 'fixed step'}, 'Location', 'SouthEast'); 
axis([0, nP, 0, 1]); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('% Optimal Action');
title(['sigmaRW=', num2str(sigmaRW)]); 