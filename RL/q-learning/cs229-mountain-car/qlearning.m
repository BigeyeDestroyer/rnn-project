function [q, steps_per_episode] = qlearning(episodes)

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
    plot_mountain_car(x);
    pause(0.1);
    
    % maxq : the maximum action for current state
    % a    : the corresponding action index 
    [~, a] = max(q(s, :));
    if (q(s, 1) == q(s, 2))
        a = ceil(rand * num_actions);
    end
    steps = 0;
    %%% YOUR CODE HERE
    while(~absorb)
        % execute the best action 
        [x, s_prime, absorb] = mountain_car(x, actions(a)); 
        reward = -double(absorb == 0);
        
        % find the best action for next state and update q value 
        [maxq, a_prime] = max(q(s_prime, :));
        if (q(s_prime, 1) == q(s_prime, 2))
            a_prime = ceil(rand * num_actions);
        end
        q(s, a) = (1 - alpha) * q(s, a) + ...
            alpha * (reward + gamma * maxq);
        
        s = s_prime;
        a = a_prime;
        steps = steps + 1;
        fprintf('%d-th iter in %d-th episode ...\n', steps, i);
        plot_mountain_car(x);
        pause(0.1);
    end
    steps_per_episode(i) = steps;
    fprintf('%d-th episode\n', i);
end
end