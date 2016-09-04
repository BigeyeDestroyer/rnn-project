% Demonstrate the learning technique of persuit learning
% We compare three methods
% 
% M1) action-value method with 1/k update and epsilon=0.1 greedy
% M2) the reinforment comparison method (with zero initial reward) 
% M3) the persuit method 
% 
% Inputs: 
%   nB: the number of bandits
%   nA: the number of arms
%   nP: the number of plays (times we will pull a arm)
%   sigmaReward: the standard deviation of the return from each of the arms

nB = 2000;
nA = 10;
nP = 2000;
sigmaReward = 1.0;

%-----
% Compare three methods: 
% M1) action-value method with 1/k update and epsilon=0.1 greedy
% M2) the reinforment comparison method (with zero initial reward) 
% M3) the persuit method 
%-----
tEps = 0.1;
alpha = 0.1; 

avgReward = zeros(3, nP); 
perOptAction = zeros(3, nP);

allRewards_M1 = zeros(nB, nP);
pickedMaxAction_M1 = zeros(nB, nP);
allRewards_M2 = zeros(nB, nP);
pickedMaxAction_M2 = zeros(nB, nP);
allRewards_M3 = zeros(nB, nP);
pickedMaxAction_M3 = zeros(nB, nP);

for bi = 1: nB % pick a bandit 
    % generate the TRUE reward Q^{\star}
    qStarMeans = randn(1, nA);
    
    % for action-value method (\epsilon, 1/k) method: 
    qT_M1 = zeros(1, nA);
    qN_M1 = zeros(1, nA);
    
    % reinforcement comparison method: 
    pT = zeros(1, nA); % initialize play preference 
    rT = 0; % initialize a reference reward 
    
    % action persuit method: 
    qT_M3 = zeros(1, nA);
    qN_M3 = zeros(1, nA);
    piT_M3 = ones(1, nA) / nA;
    for pi = 1: nP % make a play 
        % Step 1: pick an arm 
        % 1. first method 
        if (rand(1) < tEps) % explore: random 
            [~, arm_M1] = histc(rand(1), linspace(0, 1 + eps, nA + 1));
        else % exploit: greedy 
            [~, arm_M1] = max(qT_M1);
        end
        
        % 2. reinforcement comparison 
        piT = exp(pT) ./ sum(exp(pT));
        % draw an arm from the distribution piT
        arm_M2 = sample_discrete(piT, 1, 1);
        
        % 3. action pursuit
        beta = 0.01;
        [~, arm_M3] = max(qT_M3); % pick the greedy choice 
        piT_M3(arm_M3) = piT_M3(arm_M3) + beta * (1 - piT_M3(arm_M3));
        for ar = 1: nA % decrement all the others 
            if (ar == arm_M3)
                continue;
            end
            piT_M3(ar) = piT_M3(ar) + beta * (0 - piT_M3(ar));
        end
        arm_M3 = sample_discrete(piT_M3, 1, 1); % sample from this distribution 
        
        % Step 2: determine the best arm and get the reward 
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
        
        % get the reward 
        reward_M1 = qStarMeans(arm_M1) + sigmaReward * randn(1);
        reward_M2 = qStarMeans(arm_M2) + sigmaReward * randn(1);
        reward_M3 = qStarMeans(arm_M3) + sigmaReward * randn(1);
        allRewards_M1(bi, pi) = reward_M1;
        allRewards_M2(bi, pi) = reward_M2;
        allRewards_M3(bi, pi) = reward_M3;
        
        % Step 3: update qT
        qT_M1(arm_M1) = qT_M1(arm_M1) + ...
            (reward_M1 - qT_M1(arm_M1)) / (qN_M1(arm_M1) + 1);
        qN_M1(arm_M1) = qN_M1(arm_M1) + 1;
        
        beta = 0.1;
        pT(arm_M2) = pT(arm_M2) + beta * (reward_M2 - rT);
        rT = rT + alpha * (reward_M2 - rT);
        
        qT_M3(arm_M3) = qT_M3(arm_M3) + ...
            (reward_M3 - qT_M3(arm_M3)) / (qN_M3(arm_M3) + 1);
        qN_M3(arm_M3) = qN_M3(arm_M3) + 1;
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
legend(all_hnds, {'act-val', 'rein\_comp', 'action\_persuit'}, 'Location', 'SouthEast'); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('Average Reward'); 

% optimal action rate plot 
figure; hold on; 
all_hnds = plot(1: nP, perOptAction);
legend(all_hnds, {'act-val', 'rein\_comp', 'action\_persuit'}, 'Location', 'SouthEast'); 
axis([0, nP, 0, 1]); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('% Optimal Action');