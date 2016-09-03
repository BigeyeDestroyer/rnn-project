nB = 2000;
nA = 10;
nP = 2000;
sigma = 1.0;

%% 1. Simulation 
% Generate the TRUE reward Q^{\star}
qStarMeans = mvnrnd(zeros(nB, nA), eye(nA));

% run an experiment for each epsilon  
% 0 => fully greedy 
% 1 => explore on each trial 
epsArray = [0, 0.01, 0.1];

% average reward for each play across all bandits 
avgReward = zeros(length(epsArray), nP);
% average optimal action rate for each play across all bandits
perOptAction = zeros(length(epsArray), nP);
% cumulative average reward 
cumReward = zeros(length(epsArray), nP);
% cumulative average optimal action rate
cumProb = zeros(length(epsArray), nP);

for ei = 1: length(epsArray)
    tEps = epsArray(ei);
    
    allRewards = zeros(nB, nP);
    pickedMaxAction = zeros(nB, nP);
    
    for bi = 1: nB % pick a bandit 
        qT = zeros(1, nA); % value function 
        qN = zeros(1, nA); % keep track of the draws on each arm 
        for pi = 1: nP % make a play 
            % determine if this move is exploritory or greedy 
            if (rand(1) < tEps) % explore
                [~, arm] = histc(rand(1), linspace(0, 1 + eps, nA + 1));
            else
                [~, arm] = max(qT); 
            end
            
            % determine if the arm selected is the best possible 
            [~, bestArm] = max(qStarMeans(bi, :));
            if (arm == bestArm)
                pickedMaxAction(bi, pi) = 1;
            end
            
            % get the reward from drawing on that arm 
            reward = qStarMeans(bi, arm) + sigma * randn(1);
            allRewards(bi, pi) = reward;
            
            % update qT, qN incrementally
            qT(arm) = qT(arm) + (reward - qT(arm)) / (qN(arm) + 1);
            qN(arm) = qN(arm) + 1;
        end
    end
    
    avgReward(ei, :) = mean(allRewards, 1);
    perOptAction(ei, :) = mean(pickedMaxAction, 1);
    cumReward(ei, :) = mean(cumsum(allRewards, 2), 1);
    cumProb(ei, :) = mean(cumsum(pickedMaxAction, 2) ./ ...
        cumsum(ones(size(pickedMaxAction)), 2), 1);  
end

%% 2. Plot figures 
% average reward plot 
figure; 
hold on; 
clrStr = 'brkc'; 
all_hnds = []; 
for ei = 1: length(epsArray)
    all_hnds(ei) = plot(1: nP, avgReward(ei, :), [clrStr(ei), '-']); 
end 
legend(all_hnds, {'0', '0.01', '0.1'}, 'Location', 'SouthEast'); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('Average Reward'); 

% average optimal action rate plot 
figure; 
hold on; 
clrStr = 'brkc'; 
all_hnds = []; 
for ei = 1: length(epsArray)
    all_hnds(ei) = plot(1: nP, perOptAction(ei, :), [clrStr(ei), '-']); 
end 
legend(all_hnds, {'0', '0.01', '0.1'}, 'Location', 'SouthEast'); 
axis([0, nP, 0, 1]); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('% Optimal Action');

% cummulative average rewards plot 
figure; 
hold on; 
clrStr = 'brkc'; 
all_hnds = []; 
for ei = 1: length(epsArray)
    all_hnds(ei) = plot(1: nP, cumReward(ei, :), [clrStr(ei), '-']); 
end 
legend(all_hnds, {'0', '0.01', '0.1'}, 'Location', 'SouthEast'); 
axis tight; 
grid on; 

% cummulative optimal action rate plot 
figure; 
hold on; 
clrStr = 'brkc'; 
all_hnds = []; 
for ei = 1: length(epsArray)
    all_hnds(ei) = plot(1: nP, cumProb(ei, :), [clrStr(ei), '-']); 
end 
legend(all_hnds, {'0', '0.01', '0.1'}, 'Location', 'SouthEast'); 
axis([0, nP, 0, 1]); 
axis tight; 
grid on; 
xlabel( 'plays' ); ylabel( 'Cummulative % Optimal Action' );
xlabel( 'plays' ); ylabel( 'Cummulative Average Reward' ); 