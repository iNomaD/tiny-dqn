from __future__ import division, print_function, unicode_literals

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(
    description="Train a DQN net to play MsMacman.")
parser.add_argument("-n", "--number-steps", type=int, default=500000,
    help="number of training steps for epsilon decay")
parser.add_argument("-l", "--learn-iterations", type=int, default=4,
    help="number of game iterations between each training step")
parser.add_argument("-s", "--save-steps", type=int, default=1000,
    help="number of training steps between saving checkpoints")
parser.add_argument("-c", "--copy-steps", type=int, default=10000,
    help="number of training steps between copies of online DQN to target DQN")
parser.add_argument("-r", "--render", action="store_true", default=False,
    help="render the game during training or testing")
parser.add_argument("-p", "--path", default="",
    help="path of the checkpoint file")
parser.add_argument("-k", "--keep-intervals", nargs='+', type=int,
    help="a list of intervals in minutes to keep checkpoints (e.g. -k 30 60 120 240 480")
parser.add_argument("-t", "--test", action="store_true", default=False,
    help="test (no learning and minimal epsilon)")
parser.add_argument("-v", "--verbosity", action="count", default=0,
    help="increase output verbosity")
parser.add_argument("-g", "--game", default="MsPacman-v0",
    help="game id in gym")
args = parser.parse_args()

from collections import deque
import gym
import numpy as np
import os
import tensorflow as tf
import time
from threading import Timer

env = gym.make(args.game)
done = True  # env needs to be reset
model_save_path = os.path.join(os.getcwd(), args.game) if not args.path else args.path

# Define actions for games (gym-0.9.4)
if args.game == "Pong-v0":
    action_space = [0, 2, 5] # [NONE, UP, DOWN]
elif args.game == "Breakout-v0":
    action_space = [1, 2, 3] # [FIRE, RIGHT, LEFT]
else:
    # 9 discrete actions are available
    action_space = [i for i in range(0, env.action_space.n)]
n_outputs = len(action_space)

def pick_action(action_number):
    return action_space[action_number]

# First let's build the two DQNs (online & target)
input_height = 80
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 10 * 10  # conv3 has 64 maps of 10x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
initializer = tf.contrib.layers.variance_scaling_initializer()

def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

# We need an operation to copy the online DQN to the target DQN
copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# Now for the training operations
learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_episode = tf.Variable(0, trainable=False, name='global_episode')
    global_time = tf.Variable(0, trainable=False, name='global_time')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Let's implement a simple replay memory
replay_memory_size = 20000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
           cols[4].reshape(-1, 1))

# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.1
eps_max = 1.0 if not args.test else eps_min
eps_decay_steps = args.number_steps
eps_decay_rate = (eps_max - eps_min) / eps_decay_steps  # constant

def epsilon_calc(step):
    return max(eps_min, eps_max - eps_decay_rate * step )

# keep saved model in separate folder
keep_flag = False
if(args.keep_intervals):
    ki = args.keep_intervals
    print("keep-intervals: ", ki)
    delays = [ki[i] - ki[i - 1] for i in range(1, len(ki))]
    print("delays ", delays)
    def set_flag():
        print("keep time")
        global keep_flag
        keep_flag = True
        if delays:
            Timer(delays.pop(0) * 60, set_flag).start()
    Timer(ki[0] * 60, set_flag).start()

# We need to preprocess the images to speed up training
mspacman_color = np.array([210, 164, 74]).mean()
pong_bg_color = np.array([144, 72, 17]).mean()
breakout_wall_color = np.array([142, 142, 142]).mean()

def preprocess_observation(obs):
    # crop and downsize (from 210x160 to 80x80x3)
    # to greyscale (to 80x80)
    # Improve contrast
    if args.game == "MsPacman-v0":
        img = obs[7:167:2, ::2]
        img = img.mean(axis=2)
        img[img == mspacman_color] = 0
    elif args.game == "Pong-v0":
        img = obs[35:195:2, ::2]
        img = img.mean(axis=2)
        img[img == pong_bg_color] = 0
    elif args.game == "Breakout-v0":
        img = obs[35:195:2, ::2]
        img = img.mean(axis=2)
        img[img == breakout_wall_color] = 25
    else:  # others
        img = obs[35:195:2, ::2]  # guess?
        img = img.mean(axis=2)
    #from scipy.misc import toimage
    #toimage(img).show()
    img = (img - 128) / 128  # normalize from -1. to 1.
    return img.reshape(80, 80, 1) # to 80x80x1

# TensorFlow - Execution phase
training_start = 10000  # start training after 10,000 game iterations
discount_rate = 0.99
skip_start = 30  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
state = None

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

reward_sum = 0
stat_episodes = 0
stat_reward = 0
stat_iterations = 0
stat_prev_time = time.clock()

with tf.Session() as sess:
    ckpt_path = os.path.join(model_save_path, 'my_dqn.ckpt')
    if os.path.isfile(ckpt_path + ".index"):
        print("Restoring model %s." % ckpt_path)
        saver.restore(sess, ckpt_path)
    else:
        init.run()
        copy_online_to_target.run()

    # prepare variables from model
    step = global_step.eval()
    episode = global_episode.eval()
    total_time = global_time.eval()
    print("Starting with step=%d, episode=%d, total_time=%d" % (step, episode, total_time/60))

    while True:
        iteration += 1
        if args.verbosity > 0:
            print("\rIteration {}   Training step {}/{} ({:.1f})%   "
                  "Loss {:5f}    Mean Max-Q {:5f}   ".format(
            iteration, step, args.number_steps, step * 100 / args.number_steps,
            loss_val, mean_max_q), end="")
        if done: # game over, start again
            obs = env.reset()
            for skip in range(skip_start): # skip the start of each game
                obs, reward, done, info = env.step(pick_action(0))
            state = preprocess_observation(obs)

        if args.render:
            env.render()

        # Epsilon greedy strategy
        epsilon = epsilon_calc(step)
        if np.random.rand() >= epsilon:
            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = np.argmax(q_values)  # optimal action

            # Compute statistics for tracking progress (not shown in the book)
            total_max_q += q_values.max()
            game_length += 1
        else:
            action = np.random.randint(n_outputs)  # random action

        # Online DQN plays
        obs, reward, done, _ = env.step(pick_action(action))
        next_state = preprocess_observation(obs)

        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        # antistuck if game is too long
        if game_length == args.save_steps:
            done = True
            reward = -1

        # Compute further statistics for tracking progress (not shown in the book)
        reward_sum += reward
        stat_iterations += 1
        if done:
            episode += 1
            mean_max_q = total_max_q / game_length if game_length != 0 else 0
            total_max_q = 0.0
            game_length = 0
            stat_episodes += 1
            stat_reward += reward_sum
            print('resetting env. episode%d reward %f.' % (episode, reward_sum))
            reward_sum = 0

        if args.test:
            continue

        if iteration < training_start or iteration % args.learn_iterations != 0:
            continue # only train after warmup period and at regular intervals

        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN and increase step counter
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})
        step += 1

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            print("Copy online DQN to target DQN.")
            copy_online_to_target.run()

        # And print statistics regularly (with save frequency)
        if step % args.save_steps == 0:
            stat_time = time.clock() - stat_prev_time
            stat_prev_time = time.clock()
            total_time += stat_time
            iter_per_second = stat_iterations / stat_time
            running_reward = 1.0 * stat_reward / stat_episodes if stat_episodes != 0 else float('nan')
            print("iterations per second: %d. running reward mean: %f." % (iter_per_second, running_reward))
            print("total train steps: %d. total time (min.): %d. epsilon: %f" % (step, total_time//60, epsilon_calc(step)))
            stat_episodes = 0
            stat_reward = 0
            stat_iterations = 0

        # And save regularly
        if step % args.save_steps == 0:
            # prepare variables first
            sess.run(tf.assign(global_episode, episode))
            sess.run(tf.assign(global_step, step))
            sess.run(tf.assign(global_time, total_time))
            print("Saving model.")
            if keep_flag:
                saver.save(sess, os.path.join(model_save_path, str(total_time//60)+'m', 'my_dqn.ckpt'))
                keep_flag = False
            else:
                saver.save(sess, os.path.join(model_save_path, 'my_dqn.ckpt'))
