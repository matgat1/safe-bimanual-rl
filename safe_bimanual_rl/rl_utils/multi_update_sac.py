from mushroom_rl.algorithms.actor_critic import SAC


class MultiUpdateSAC(SAC):
    """
    SAC performing several gradient steps per fit.

    With a vectorized environment, n_steps_per_fit must be >= n_envs, so plain SAC
    would do a single gradient step every n_envs transitions. This keeps the
    update-to-data ratio at n_updates_per_fit / n_steps_per_fit.
    """

    def __init__(self, *args, n_updates_per_fit: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_updates_per_fit = n_updates_per_fit
        self._add_save_attr(_n_updates_per_fit="primitive")

    def fit(self, dataset):
        self._history_manager.update_preprocessors(dataset)
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            for _ in range(self._n_updates_per_fit):
                self._gradient_step()

    def _gradient_step(self):
        state, action, reward, next_state, absorbing, *_ = self._replay_memory.get(
            self._batch_size()
        )
        state, next_state = self._augment(state), self._augment(next_state)

        if self._replay_memory.size > self._warmup_transitions():
            action_new, log_prob = self.policy.draw_with_log_prob(state)
            loss = self._loss(state, action_new, log_prob)
            self._optimize_actor_parameters(loss)
            self._update_alpha(log_prob.detach())

        q_next = self._next_q(next_state, absorbing)
        q = reward + self.mdp_info.gamma * q_next

        self._critic_approximator.fit(state, action, q.detach(), **self._critic_fit_params)

        self._update_target(self._critic_approximator, self._target_critic_approximator)
