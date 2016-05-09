function [q, steps_per_episode] = qlearning_backup(episodes)

% set up parameters and initialize q values
alpha = 0.05;
gamma = 0.99;
num_states = 100;
num_actions = 2;
actions = [-1, 1];
q = zeros(num_states, num_actions);

steps_per_episode = zeros(1, episodes);
for i = 1 : episodes
    % the start point is [0.0, -pi / 6]
    [x, s, absorb] =  mountain_car([0.0 -pi/6], 0);
    if absorb % suppose we get to top within only one step
        steps_per_episode(i) = steps_per_episode(i) + 1;
        continue;
    end
    %%% YOUR CODE HERE
    while(~absorb)
        a = (rand > 0.5) + 1;
        [x_prime, s_prime, absorb] = mountain_car(x, actions(a)); % carry on for one step 
        q(s, a) = (1 - alpha) * q(s, a) + ...
            gamma * (-1 * (x_prime(1) <= 0.5) + gamma * max(q(s_prime, :)));
        
        x = x_prime;
        s = s_prime;
        
        steps_per_episode(i) = steps_per_episode(i) + 1;
    end
    fprintf('%d-th episode\n', i);
end
end