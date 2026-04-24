import numpy as np
from mushroom_rl.core import Core


class SafeCore(Core):
    """
    Extended Core that injects cost into agent.fit, adapted from cremini_rl for mushroom-rl 2.0.

    The original cremini _step returns (state, action, reward, next_state, cost, absorbing, last)
    directly from the MDP. In mushroom-rl 2.0, cost is instead extracted from step_info and we
    build the same 7-tuple for agent.fit via a parallel safe_dataset list.
    """

    def __init__(self, agent, env, callbacks_fit=None, callback_step=None, record_dictionary=None):
        super(SafeCore, self).__init__(agent, env, callbacks_fit, callback_step, record_dictionary)
        self._cost = 0

    def _reset(self, initial_states):
        super(SafeCore, self)._reset(initial_states)
        self._cost = 0

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None):
        self._core_logic.initialize_run(n_steps, n_episodes, initial_states, quiet)

        safe_dataset = []
        last = True

        while self._core_logic.move_required():
            if last:
                self._reset(initial_states)
                if self.agent.info.is_episodic:
                    dataset.append_theta(self._current_theta)

            # Standard mushroom-rl 2.0 _step returns 8-tuple:
            # (state, action, reward, next_state, absorbing, last, policy_state, policy_next_state)
            sample, step_info = self._step(render, record)
            cost = float(step_info.get("cost", 0.0))
            self._cost = cost

            self.callback_step(sample)
            self._core_logic.after_step(sample[5])  # sample[5] = last

            dataset.append(sample, step_info)

            # Build 7-tuple matching the original cremini format
            state, action, reward, next_state, absorbing, last_flag = sample[:6]
            safe_dataset.append((state, action, reward, next_state, cost, absorbing, last_flag))

            if self._core_logic.fit_required():
                self.agent.fit(safe_dataset)
                self._core_logic.after_fit()
                for c in self.callbacks_fit:
                    c(safe_dataset)
                safe_dataset.clear()
                dataset.clear()

            last = sample[5]

        self.agent.stop()
        self.env.stop()
        self._end(record)

        dataset.info.parse()
        dataset.episode_info.parse()

        dataset._safe_dataset = safe_dataset
        return dataset

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=True, record=False):
        dataset = super().evaluate(
            initial_states=initial_states, n_steps=n_steps, n_episodes=n_episodes,
            render=render, quiet=quiet, record=record,
        )

        safe_data = getattr(dataset, "_safe_dataset", [])

        if not safe_data:
            return {"J": 0.0, "mean_cost": 0.0}

        gamma = self.env.info.gamma
        episode_J, episode_costs = [], []
        current_J, current_cost, discount = 0.0, 0.0, 1.0

        for _, _, reward, _, cost, _, last in safe_data:
            current_J += discount * float(reward)
            current_cost += float(cost)
            discount *= gamma
            if last:
                episode_J.append(current_J)
                episode_costs.append(current_cost)
                current_J, current_cost, discount = 0.0, 0.0, 1.0

        return {
            "J": float(np.mean(episode_J)) if episode_J else 0.0,
            "mean_cost": float(np.mean(episode_costs)) if episode_costs else 0.0,
        }
