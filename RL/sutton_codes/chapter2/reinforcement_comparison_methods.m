% Demonstrate the technique of reinforcement comparison
% We compare three methods
% 
% M1) action-value method with constant update (alpha = 0.1) and epsilon=0.1 greedy
% M2) action-value method with 1/k update and epsilon=0.1 greedy
% M3) the reinforment comparison method
%
% Inputs: 
%   nB: number of bandits
%   nA: number of arms
%   nP: number of plays (times we will pull a arm)
%   sigmaReward: the standard deviation of the return from each of the arms

nB = 2000;
nA = 10;
nP = 2000;
sigmaReward = 1.0;

%-----
% Compare two methods: 
% M1) An epsion greedy (eps=0.1) algorithm with constant update (alpha=0.1)
% M2) An epsion greedy (eps=0.1) algorithm with 1/k action value update
% M3) the reinforment comparison method 
%-----
tEps  = 0.1;
alpha = 0.1; 
beta  = 0.1; 

avgReward = zeros(3, nP);
perOptAction = zeros(3, nP);

allRewards_M1 = zeros(nB, nP);
pickedMaxAction_M1 = zeros(nB, nP);
allRewards_M2 = zeros(nB, nP);
pickedMaxAction_M2 = zeros(nB, nP);
allRewards_M3 = zeros(nB, nP);
pickedMaxAction_M3 = zeros(nB, nP);

for bi = 1: nB % pick a bandit 
    fprintf('%d/%d bandit ...\n', bi, nB);
    % generate the TRUE reward Q^{\star}
    qStarMeans = randn(1, nA);
    
    % epsilon-greedy method 
    qT_M1 = zeros(1, nA); % initial action value estimations 
    qT_M2 = zeros(1, nA); 
    qN_M2 = zeros(1, nA); % initialize number of draws on this arm 
    
    % reinforcement comparison method 
    pT = zeros(1, nA); % initialize play reference 
    rT = 0; % initialize a reference reward (one for all received)
    
    for pi = 1: nP % make a play 
        % 1. First two epsilon-greedy methods
        % EXPLORITORY v.s. GREEDY MOVES 
        if (rand(1) < tEps) % pick a RANDOM arm 
            [~, arm_M1] = histc(rand(1), linspace(0, 1 + eps, nA + 1));
            arm_M2 = arm_M1;
        else % pick a GREEDY arm 
            [~, arm_M1] = max(qT_M1);
            [~, arm_M2] = max(qT_M2);
        end
        
        % 2. Reinforcement comparison method 
        % EXPLORITORY v.s. GREEDY MOVES 
        piT = exp(pT) ./ sum(exp(pT));
        % draw an arm from the distribution piT
        arm_M3 = sample_discrete(piT, 1, 1);
        
        % determine if the arm selected is the best possible 
        [~, bestArm] = max(qStarMeans);
        if (arm_M1 == bestArm)
            pickedMaxAction_M1(bi, pi) = 1;
        end
        if (arm_M2 == bestArm)
            pickedMaxAction_M2(bi, pi) = 1;
        end
        if (arm_M3 == bestArm)
            pickedMaxAction_M3(bi, pi) = 1;
        end
        
        % get the reward from drawing that arm 
        reward_M1 = qStarMeans(arm_M1) + sigmaReward * randn(1);
        reward_M2 = qStarMeans(arm_M2) + sigmaReward * randn(1);
        reward_M3 = qStarMeans(arm_M3) + sigmaReward * randn(1);
        allRewards_M1(bi, pi) = reward_M1;
        allRewards_M2(bi, pi) = reward_M2;
        allRewards_M3(bi, pi) = reward_M3;
        
        % update qT
        qT_M1(arm_M1) = qT_M1(arm_M1) + ...
            alpha * (reward_M1 - qT_M1(arm_M1));
        qT_M2(arm_M2) = qT_M2(arm_M2) + ...
            (reward_M2 - qT_M2(arm_M2)) / (qN_M2(arm_M2) + 1);
        qN_M2(arm_M2) = qN_M2(arm_M2) + 1;
        
        % the reinforcement comparison update 
        % play preference pT and average reward 
        pT(arm_M3) = pT(arm_M3) + beta * (reward_M3 - rT);
        rT = rT + alpha * (reward_M3 - rT); 
    end
end

avgReward(1, :) = mean(allRewards_M1, 1);
avgReward(2, :) = mean(allRewards_M2, 1);
avgReward(3, :) = mean(allRewards_M3, 1);

perOptAction(1, :) = mean(pickedMaxAction_M1, 1);
perOptAction(2, :) = mean(pickedMaxAction_M2, 1);
perOptAction(3, :) = mean(pickedMaxAction_M3, 1);

% average reward plot 
figure; 
hold on; 
all_hnds = plot(1: nP, avgReward);
legend(all_hnds, {'alpha=0.1', 'alpha=1/k', 'rein\_comp'}, 'Location', 'SouthEast'); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('Average Reward'); 

% optimal action rate plot 
figure; 
hold on; 
all_hnds = plot(1: nP, perOptAction);
legend(all_hnds, {'alpha=0.1', 'alpha=1/k', 'rein\_comp'}, 'Location', 'SouthEast'); 
axis([0, nP, 0, 1]); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('% Optimal Action');