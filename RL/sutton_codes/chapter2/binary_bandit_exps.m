nB = 200;  % number of bandits
nP = 1000; % number of plays 
p_win = [0.8, 0.9]; % p_win(i) is the probability we win when we pull arm i 


nA = length(p_win); % number of arms 
[~, bestArm] = max(p_win);

%% 1. Run the supervised experiment for two epsilon:
%     0   => fully greedy 
%     0.1 => inbetween 
%     1.0 => explore on each trial 
fprintf('Supervised method for 2-armed bandit ...\n');
pause(1);
epsArray = [0, 0.1];

% percentage of optimal actions among the 
% 'nB' bandits across all the 'nP' plays 
perOptAction = zeros(length(epsArray), nP);
for ei = 1: length(epsArray)
    tEps = epsArray(ei); % current prob for exploration 
    
    pickedMaxAction = zeros(nB, nP);
    for bi = 1: nB % pick a bandit 
        fprintf('%d/%d bandit for %.2f exploration ...\n', bi, nB, tEps);
        % randomly pick an arm to initial the play 
        [~, arm] = histc(rand(1), linspace(0, 1 + eps, nA + 1));
        for pi = 1: nP % make a play 
            % determine if this move is exploritory or greedy 
            if (rand(1) <= tEps)  % in this case, we explore
                [~, arm] = histc(rand(1), linspace(0, 1 + eps, nA + 1));
            end
            
            otherArm = 3 - arm; 
            % determine if the arm selected is the best possible 
            if (arm == bestArm)
                pickedMaxAction(bi, pi) = 1;
            end
            
            if (rand(1) <= p_win(arm))
                % do nothing and consider this arm to be the best
            else % otherwise we think the best arm is the other one
                arm = otherArm; 
            end
        end
    end
    percentOptAction = mean(pickedMaxAction, 1);
    perOptAction(ei, :) = percentOptAction;
end

%% 2. Run the L_{R-P} (linear reward penalty) algorithm 
fprintf('Linear reward penalty for 2-armed bandit ...\n');
pause(1);
alpha = 0.1;
perOptActionRP = zeros(1, nP);

qT = 0.5 * ones(nB, nP); % my estimation of the prob that 
                         % this arm makes a success
pickedMaxAction = zeros(nB, nP); 
for bi = 1: nB % pick a bandit 
    fprintf('%d/%d bandit ...\n', bi, nB);
    for pi = 1: nP % make a play 
        if (rand(1) < qT(bi, 1))
            arm = 1;
        else
            arm = 2;
        end
        
        otherArm = 3 - arm;
        % determine if the arm selected is the best possible 
        if (bestArm == arm)
            pickedMaxAction(bi, pi) = 1;
        end

        % determine whether the current arm leads to success
        if (rand(1) <= p_win(arm)) % success
            addTo = arm;
        else % failure 
            addTo = otherArm;
        end

        otherArm = 3 - addTo; % update 'otherArm' for prob updating 
        qT(bi, addTo) = qT(bi, addTo) + alpha * (1 - qT(bi, addTo));
        qT(bi, otherArm) = 1 - qT(bi, addTo);
    end
end
perOptActionRP(1, :) = mean(pickedMaxAction, 1);
perOptAction = [perOptAction; perOptActionRP];

%% 3. Run the L_{R-I} (linear reward inaction) algorithm 
fprintf('Linear reward inaction method for 2-armed bandit ...\n');
pause(1);
alpha = 0.1;
perOptActionRI = zeros(1, nP);
qT = 0.5 * ones(nB, nA); % my estimation of the prob that 
                    % this arm makes a success
pickedMaxAction = zeros(nB, nP);
for bi = 1: nB % pick a bandit 
    fprintf('%d/%d bandit ...\n', bi, nB);
    for pi = 1: nP
        % pick an arm based in the distribution in qT
        if (rand(1) < qT(bi, 1))
            arm = 1;
        else
            arm = 2;
        end
        otherArm = 3 - arm;
        
        % determine whether the selected arm is the best arm 
        if (arm == bestArm)
            pickedMaxAction(bi, pi) = 1;
        end
        
        % determine whether the current arm leads to success
        % if not, we won't update its prob of success
        if (~(rand(1) < p_win(arm)))
            continue;
        end
        addTo = arm;
        otherArm = 3 - addTo;
        
        qT(bi, addTo) = qT(bi, addTo) + alpha * (1 - qT(bi, addTo));
        qT(bi, otherArm) = 1 - qT(bi, addTo);
    end
end
perOptActionRI(1, :) = mean(pickedMaxAction, 1);
perOptAction = [perOptAction; perOptActionRI];

%% 4. plot all the learning curves
figure;
hold on;
clrStr = 'brkc';
all_hnds = [];
for ei = 1: size(perOptAction, 1)
    all_hnds(ei) = plot(1: nP, perOptAction(ei, :), [clrStr(ei), '-']);
end
legend(all_hnds, {'0', '0.1', 'L_{RP}', 'L_{RI}' }, 'Location', 'Best'); 
axis([0, nP, 0, 1]); 
axis tight; 
grid on; 
xlabel('plays');
ylabel('% Optimal Action');