function [] = n_armed_testbed(nB,nA,nP,sigma)
% 
% Generates the 10-armed bandit testbed.
% 
% Inputs: 
%   nB: the number of bandits
%   nA: the number of arms
%   nP: the number of plays (times we will pull a arm)
%   sigma: the standard deviation of the return from each of the arms

%close all; 
%clc; 
%clear; 
if(nargin < 1) % the number of bandits
    nB = 2000;
end

if(nargin < 2) % the number of arms
    nA = 10;
end

if(nargin < 3) % the number of plays (times we will pull an arm)
    nP = 10000;
end

if(nargin < 4) % the standard deviation of the return from each of the arms
    sigma = 1.0;
end

randn('seed', 0); 

% generate the TRUE reward Q^{\star}: 
% For 'nB' bandits each with 'nA' arms
qStarMeans = mvnrnd(zeros(nB, nA), eye(nA)); 

% run an experiment for each epsilon:
% 0 => fully greedy
% 1 => explore on each trial
epsArray = [0, 0.01, 0.1];

% assume we have at least ONE draw from each "arm" (initialize with use the qStarMeans matrix):
% each row a bandit with 'nA' arms
qT0 = mvnrnd(qStarMeans, eye(nA));

% statistics across all the strategies and plays
avgReward    = zeros(length(epsArray), nP); 
perOptAction = zeros(length(epsArray), nP); 
cumReward    = zeros(length(epsArray), nP); 
cumProb      = zeros(length(epsArray), nP);

for ei = 1 : length(epsArray)
  tEps = epsArray(ei); % epsilon for current 'epsilon-greedy'

  
  % qT : nB x nA, estimation of mean for each arm in every bandit
  % qN : nB x nA, number of draws for each arm in every bandit 
  % qS : nB x nA, total rewards for each arm in every bandit 
  % 
  % qT = qT0;  % <- initialize to one draw per arm 
  qT = zeros(size(qT0));  % <- initialize to zero draws per arm (no knowledge)
  qN = ones(nB, nA); % keep track of the number draws on this arm 
  qS = qT;             % keep track of the SUM of the rewards (qT = qS./qN) 
  
  % allRewards      : nB x nP, all rewards for each bandit during every play
  % pickedMaxAction : nB x nP, picked action for each bandit in all plays
  allRewards      = zeros(nB,nP); 
  pickedMaxAction = zeros(nB,nP); 
  for bi = 1: nB % pick a bandit
    for pi = 1: nP % make a play
      % determine if this move is exploritory or greedy: 
      if(rand(1) <= tEps) % pick a RANDOM arm: 
        [~, arm] = histc(rand(1), linspace(0, 1 + eps, nA + 1));
      else                  % pick the GREEDY arm:
        [~, arm] = max(qT(bi, :));
      end
      % determine if the arm selected is the best possible: 
      [~, bestArm] = max(qStarMeans(bi, :)); 
      if(arm == bestArm)
          pickedMaxAction(bi,pi) = 1; 
      end
      % get the reward from drawing on that arm: 
      reward = qStarMeans(bi, arm) + sigma * randn(1); 
      allRewards(bi, pi) = reward; 
      % update qN,qS,qT: 
      qN(bi, arm) = qN(bi, arm) + 1;
      qS(bi, arm) = qS(bi, arm) + reward; 
      qT(bi, arm) = qS(bi, arm) / qN(bi, arm); 
    end
  end
  % 1. average rewards across among all bandits 
  avgRew = mean(allRewards, 1); % 1 x nP, average across all bandits
  avgReward(ei, :) = avgRew(:).'; % length(epsArray) x nP 
  
  % 2. average optimal action selected among all the bandits
  percentOptAction = mean(pickedMaxAction, 1); % 1 x nP
  perOptAction(ei, :) = percentOptAction(:).'; % length(epsArray) x nP
  
  % 3. cumsum rewards for all the plays
  csAR = cumsum(allRewards, 2); % do a cummulative sum across plays for each bandit
  csRew = mean(csAR, 1);
  cumReward(ei, :) = csRew(:).'; 
  
  % 4. cumsum optimal action selected for all plays
  csPA = cumsum(pickedMaxAction, 2) ./ cumsum(ones(size(pickedMaxAction)), 2);
  csProb = mean(csPA, 1);
  cumProb(ei, :) = csProb(:).';
end

% produce the average rewards plot: 
figure; 
hold on; 
clrStr = 'brkc'; 
all_hnds = []; % get all the handles of the curves
for ei = 1 : length(epsArray)
  all_hnds(ei) = plot(1: nP, avgReward(ei, :), [clrStr(ei), '-']); 
end 
legend(all_hnds, {'0', '0.01', '0.1' }, 'Location', 'SouthEast'); 
axis tight; grid on; 
xlabel('plays'); 
ylabel('Average Reward'); 

% produce the percent optimal action plot: 
figure; 
hold on; 
clrStr = 'brkc'; 
all_hnds = []; 
for ei = 1: length(epsArray)
  all_hnds(ei) = plot(1: nP, perOptAction(ei, :), [clrStr(ei), '-'] ); 
end 
legend(all_hnds, {'0', '0.01', '0.1'}, 'Location', 'SouthEast'); 
axis([0, nP, 0, 1]); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('% Optimal Action');

% produce the cummulative average rewards plot: 
figure; 
hold on; 
clrStr = 'brkc'; 
all_hnds = []; 
for ei = 1: length(epsArray)
  all_hnds(ei) = plot(1: nP, cumReward(ei, :), [clrStr(ei), '-'] ); 
end 
legend(all_hnds, {'0', '0.01', '0.1'}, 'Location', 'SouthEast'); 
axis tight; 
grid on; 
xlabel('plays'); 
ylabel('Cummulative Average Reward'); 

% produce the cummulative percent optimal action plot: 
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
xlabel('plays'); 
ylabel('Cummulative % Optimal Action');