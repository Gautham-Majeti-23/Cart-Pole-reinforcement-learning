# CartPole with REINFORCE (PyTorch)

A from-scratch implementation of the **REINFORCE** policy gradient algorithm that learns to balance a pole on a cart in Gymnasium's `CartPole-v1`. The policy network is built with PyTorch, but the reinforcement learning machinery (episode collection, discounted returns, advantage normalisation, the policy gradient loss and the update step) is written by hand, with no RL libraries.

This project was built as a test bed for the learning code before applying it to a harder problem: training a car to drive an obstacle course from sensor readings.

## How it works

The agent interacts with the environment in a loop:

1. The environment gives a **state** (cart position, cart velocity, pole angle, pole angular velocity).
2. The policy network outputs two logits; a softmax turns them into probabilities for pushing left or right.
3. An action is **sampled** from those probabilities (not the argmax), which gives the agent exploration.
4. The environment returns the next state and a reward (+1 for every step the pole stays up).
5. After a batch of episodes, the network is updated with the policy gradient, and the process repeats with fresh episodes.

### The algorithm

**Discounted returns.** For each step, the return is the reward at that step plus the discounted return of the next step, computed backwards from the end of each episode:

```
G_t = r_t + gamma * G_(t+1)
```

**Advantages.** Returns from every episode in a batch are pooled, then normalised across the whole batch (subtract the mean, divide by the standard deviation). Steps that did better than average get a positive advantage, and steps that did worse get a negative one. Subtracting the mean does not change the expected gradient, but it greatly reduces its variance.

**Loss.** The policy gradient loss is the negative mean of the log-probability of each action taken, weighted by its advantage:

```
loss = -mean( log pi(a_t | s_t) * advantage_t )
```

Minimising this loss makes actions that were followed by above-average outcomes more likely, and below-average ones less likely. The advantages are plain numbers computed from rewards, so gradients flow only through the log-probabilities.

## Setup

| Component | Choice |
|---|---|
| Environment | Gymnasium `CartPole-v1` |
| Policy network | `4 -> 16 -> 10 -> 2`, ReLU activations, no activation on the output |
| Optimiser | Adam, learning rate `1e-3` |
| Discount factor (gamma) | `0.99` |
| Batch size | 10 episodes per update |
| Training length | 300 updates |
| Device | CPU |

## Project structure

Everything lives in a single notebook. The main pieces are:

| Function | What it does |
|---|---|
| `passing_loop()` | Plays one episode with the current policy and returns the states, actions and rewards, with the final state dropped so all three lists line up. |
| `calculate_returns(rews, gamma)` | Turns one episode's rewards into a discounted return for every step. |
| `collect_batch(n_episodes)` | Collects several episodes, computes returns per episode, normalises them across the batch, and returns stacked tensors of states, actions and advantages, plus the average episode length. |
| `update(states, actions, advantages)` | One batched forward pass, the policy gradient loss, and the optimiser step. |
| `plot_training(history)` | Plots average episode length per iteration with a moving average. |
| `watch(policy)` / `show_frames(frames)` | Runs one episode on a rendering environment and shows it as an animation. |

## Installation

```
pip install torch gymnasium "gymnasium[classic-control]" matplotlib numpy
```

The `classic-control` extra is needed for rendering the animation.

## Usage

1. Create the model and optimiser, then the environment.
2. (Optional) Run `watch(model)` before training to see the untrained policy.
3. Run the training loop, which collects a batch, calls `update`, and records the average episode length each iteration.
4. Call `plot_training(history)` to see the learning curve.
5. Run `watch(model, greedy=True)` to see the trained policy, which takes the highest-probability action at each step.

Note that the model lives outside the training loop, so rerunning the loop continues from the current weights. Rebuild the model and optimiser to start from scratch.

## Results

The trained policy balances the pole for the full 500-step limit of `CartPole-v1`. During training, the average episode length per batch sits close to the 500-step cap, with a moving average of roughly 470 to 495 and regular dips in individual batches.

## Notes and limitations

- **Plain REINFORCE is noisy.** Even after the task is learned, performance wobbles from batch to batch, because the gradient estimate has high variance. Using a learned value baseline (actor-critic) or PPO would reduce this.
- **The batch mean is used as the baseline.** It depends very slightly on the sampled returns, which adds a small bias that shrinks with larger batches.
- **Time-limit truncation is treated as a normal end.** Episodes cut off at 500 steps are handled the same way as failures when computing returns. This is fine for CartPole but is a refinement worth making with a value function.
- **On-policy only.** Each batch is used for one update and then discarded, since the data goes stale once the policy changes.

## What's next

Implement DQN Algorithm.
The same training loop will be reused for a car driving an obstacle course on a 2D grid: sensor readings as the state, three steering actions (left, straight, right), and a progress-based reward.
