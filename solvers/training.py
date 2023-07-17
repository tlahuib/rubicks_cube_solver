import models
import numpy as np
import torch
import utils
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import concurrent.futures as cf
from time import time
from copy import deepcopy


class PPOTrainer():
    def __init__(
        self,
        actor_critic,
        ppo_clip=0.2,
        kl_div=0.01,
        max_iters=80,
        value_iters=80,
        lr=3e-4,
        value_lr = 1e-2
    ) -> None:
        self.ppo_clip = ppo_clip
        self.kl_div = kl_div
        self.max_iters = max_iters
        self.value_iters = value_iters
        self.ac = actor_critic

        policy_params = list(self.ac.shared_layers.parameters()) + \
            list(self.ac.policy_layers.parameters())
        self.optimizer = torch.optim.Adam(policy_params, lr=lr)

        value_params = list(self.ac.shared_layers.parameters()) + \
            list(self.ac.value_layers.parameters())
        self.value_optimizer = torch.optim.Adam(value_params, lr=value_lr)

    def train_policy(self, locs, colors, acts, old_log_probs, adv):

        for _ in range(self.max_iters):
            self.optimizer.zero_grad()

            new_logits, _ = self.ac(locs, colors)
            new_logits = torch.distributions.Categorical(logits=new_logits)
            new_log_probs = new_logits.log_prob(acts)

            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip, 1 + self.ppo_clip)

            clipped_loss = clipped_ratio * adv
            full_loss = policy_ratio * adv
            loss = -torch.min(full_loss, clipped_loss).mean()

            loss.backward()
            self.optimizer.step()

            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.kl_div: break


    def train_values(self, locs, colors, returns):

        losses = []
        for _ in range(self.value_iters):
            self.value_optimizer.zero_grad()

            _, values = self.ac(locs, colors)
            loss = torch.nn.functional.mse_loss(torch.squeeze(values), returns)
            losses.append(loss.item())

            loss.backward()
            self.value_optimizer.step()

        losses = np.array(losses)
        print(f"Value Loss Mean: {losses.mean():.2f}  Value Loss Std: {losses.std():.2f}")


def initialize_model(n_embed: int = 3, n_heads: int = 6, n_layers: int = 6, dropout: float = 1):
    
    _, loc_embed, color_embed, _ = utils.receiveRandomCubes(np.array([1]))

    model = models.Transformer(len(loc_embed[0]), len(color_embed[0]), n_embed, n_heads, n_layers, dropout)

    return model


def discount_rewards(rewards, gamma=0.99):
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return new_rewards[::-1]


def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):

    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return gaes[::-1]


def rollout(orig_cubes, orig_loc_embed, orig_color_embed, orig_is_solved, max_steps=50):

    # Copy originals to avoid changing them
    cubes = deepcopy(orig_cubes)
    loc_embed = deepcopy(orig_loc_embed)
    color_embed = deepcopy(orig_color_embed)
    is_solved = deepcopy(orig_is_solved)

    # Perform rollout of the different states
    args = [[], [], [], [], []] # loc, color, acts, log_probs, vals

    solved_flag = is_solved
    solved_moves = np.array([max_steps] * len(cubes))
    for step in range(max_steps):

        logits, values = model(loc_embed, color_embed)
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        log_prob = dist.log_prob(act)

        for i, item in enumerate([loc_embed, color_embed, act, log_prob, values]):
            args[i].append(item.cpu().detach().numpy())

        solved_flag = np.where(is_solved, True, solved_flag)
        solved_moves = np.where((is_solved) & (solved_moves > step), step, solved_moves)

        if all(solved_flag): break

        cubes, loc_embed, color_embed, is_solved = utils.moveCubes(cubes, act, device)
    
    args = [np.array(item) for item in args]

    # Calculate rewards and gaes
    rewards = []
    gaes = []
    for i, move in enumerate(solved_moves):
        rew = [-1] * move
        if not solved_flag[i]:
            rew[-1] = -max_steps

        rewards += discount_rewards(rew)
        gaes += calculate_gaes(rew, args[4].reshape(max_steps, -1).T[i][:move])

    # Unify the results into flat arrays
    flat_args = [[], [], [], [], []] # loc, color, acts, log_probs, vals

    t_list = [1, 0, 2]
    types = [torch.long, torch.long, torch.long, torch.float32, torch.float32]
    for i, variable in enumerate(args):
        for j, item in enumerate(np.transpose(variable, t_list[:variable.ndim])):
            flat_args[i] += item[:solved_moves[j]].tolist()

        flat_args[i] = torch.tensor(flat_args[i], dtype=types[i], device=device)

    flat_args.append(torch.tensor(rewards, dtype=torch.float32, device=device))
    flat_args.append(torch.tensor(gaes, dtype=torch.float32, device=device))
    
    return flat_args, solved_moves # loc, color, acts, log_probs, vals, rewards, gaes
    



def train(epochs, batch_size, u_bound: int):

    for epoch in range(epochs):
        print(f"Epoch ({epoch+1} / {epochs})")

        if u_bound == 1:
            scramble_moves = np.array([1] * batch_size)
        else:
            scramble_moves = np.random.randint(1, u_bound, batch_size)
        cube, loc_embed, color_embed, is_solved = utils.receiveRandomCubes(scramble_moves, device)

        args, solved_moves = rollout(cube, loc_embed, color_embed, is_solved)
        
        # Shuffle
        permute_idxs = np.random.permutation(len(args[0]))

        # Policy data
        locs = args[0][permute_idxs]
        colors = args[1][permute_idxs]
        acts = args[2][permute_idxs]
        log_probs = args[3][permute_idxs]
        gaes = args[6][permute_idxs]

        # Value data
        returns = args[5][permute_idxs]

        # Training
        ppo.train_policy(locs, colors, acts, log_probs, gaes)
        ppo.train_values(locs, colors, returns)

        # Print results
        move_diff = solved_moves - scramble_moves
        print(f"Mean Solve: {solved_moves.mean():.1f}  Solve Std: {solved_moves.std():.2f}  Mean Diff: {move_diff.mean():.1f}  Diff Std: {move_diff.std():.2f}")
        # print(solved_moves)
        # print(scramble_moves)
        # print(move_diff)
        print("----------------------------------------------------------\n\n")
    




if __name__ ==  '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model().to(device)

    ppo = PPOTrainer(
        actor_critic=model,
        ppo_clip=0.1,
        kl_div=0.005,
        max_iters=20,
        value_iters=20,
        lr=7e-5,
        value_lr=3e-3
    )

    train(100, 50, 1)


