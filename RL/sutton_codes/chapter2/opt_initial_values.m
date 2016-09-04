nB = 2000;
nA = 10;
nP = 2000;
sigmaReward = 1.0;

%-----
% Compare two methods: 
% 1) Epsion greedy (eps=0.1) algorithm with Q_0 = +0 (less exploration):
% 2) Epsion greedy (eps=0.1) algorithm with Q_0 = +5 (allows early exploration): 
%    both use the alpha=0.1 action value update algorithm
%-----

tEps = 0.1;
alpha = 0.1;

avgReward = zeros(2, nP);
perOptAction = zeros(2, nP);

allRewards_M1 = zeros(nB, nP);
pickedMaxAction_M1 = zeros(nB, nP);
allRewards_M2 = zeros(nB, nP);
pickedMaxAction_M2 = zeros(nB, nP);
for bi = 1: nB
    fprintf('%d/%d bandit ...\n', bi, nB);
    qStarMeans = randn(1, nA);
    
    qT_M1 = zeros(1, nA); 
    qT_M2 = 5 * ones(1, nA);
    
    for pi = 1: nP
        if (rand(1) <= tEps) % pick a RANDOM arm ... explore
            [~, arm_M1] = histc(rand(1), linspace(0, 1 + eps, nA + 1));
            arm_M2 = arm_M1;
        else % pick the GREEDY arm ... exploit 
            [~, arm_M1] = max(qT_M1);
            [~, arm_M2] = max(qT_M2);
        end
        
        % determine if the arm selected is the best possible 
        [~, bestArm] = max(qStarMeans);
        if (arm_M1 == bestArm)
            pickedMaxAction_M1(bi, pi) = 1;
        end
        if (arm_M2 == bestArm)
            pickedMaxAction_M2(bi, pi) = 1;
        end
        
        % get the reward from drawing on that arm 
        reward_M1 = qStarMeans(arm_M1) + sigmaReward * randn(1);
        reward_M2 = qStarMeans(arm_M2) + sigmaReward * randn(1);
        allRewards_M1(bi, pi) = reward_M1;
        allRewards_M2(bi, pi) = reward_M2;
        
        % update qT using alpha incrementally 
        qT_M1(arm_M1) = qT_M1(arm_M1) + ...
            alpha * (reward_M1 - qT_M1(arm_M1));
        qT_M2(arm_M2) = qT_M2(arm_M2) + ...
            alpha * (reward_M2 - qT_M2(arm_M2));
    end
end
avgReward(1, :) = mean(allRewards_M1, 1);
avgReward(2, :) = mean(allRewards_M2, 1);

perOptAction(1, :) = mean(pickedMaxAction_M1, 1);
perOptAction(2, :) = mean(pickedMaxAction_M2, 1);

% average rewards plot
figure; 
hold on; 
all_hnds = plot(1: nP, avgReward);
legend(all_hnds, {'Q_0=0', 'Q_0=5'}, 'Location', 'SouthEast'); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('Average Reward');

% average optimal action rate plot 
figure; 
hold on; 
all_hnds = plot(1: nP, perOptAction);
legend(all_hnds, {'Q_0=0', 'Q_0=5'}, 'Location', 'SouthEast'); 
axis([0, nP, 0, 1]); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('% Optimal Action');