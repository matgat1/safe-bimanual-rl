from mushroom_rl.core._impl.core import VectorizedCore


class PersistentVectorizedCore(VectorizedCore):
    """
    VectorizedCore whose ``learn`` continues the running episodes from one call to the next.

    MushroomRL's VectorizedCore resets every environment at the beginning of each
    ``learn`` call. With ``n_steps / n_envs`` smaller than the horizon (e.g. 240 steps
    per env for a horizon of 600), training episodes are then always cut early and
    the policy never sees the end of the task. Here the observations, episode step
    counters and end-of-episode flags are kept between calls, so episodes only end
    on an absorbing state or at the horizon, like in rsl_rl / Isaac Lab.

    Episodes cut at the end of a rollout are not absorbing: GAE bootstraps them with
    the value of the next state, so continuing them in the next call is unbiased.

    Restrictions: ``learn`` must be called with ``n_steps`` and ``n_steps_per_fit``,
    both multiples of the number of environments, and the environment must not be
    used by anything else between two calls (evaluate on a separate environment).
    """

    def __init__(self, agent, env, *args, **kwargs):
        super().__init__(agent, env, *args, **kwargs)
        self._carry = None

    def reset_episodes(self):
        """Start new episodes in every environment at the next ``learn`` call."""
        self._carry = None

    def evaluate(self, *args, **kwargs):
        # Evaluating resets the environment: the running episodes cannot be continued
        self.reset_episodes()
        return super().evaluate(*args, **kwargs)

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None, n_episodes_per_fit=None,
              render=False, record=False, quiet=False):
        n_envs = self.env.number
        assert n_steps is not None and n_episodes is None and n_episodes_per_fit is None, \
            "PersistentVectorizedCore only supports learn(n_steps=..., n_steps_per_fit=...)"
        assert n_steps % n_envs == 0 and n_steps_per_fit % n_envs == 0, \
            "n_steps and n_steps_per_fit must be multiples of the number of environments"
        self._persistent_learn = True
        try:
            super().learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, render=render,
                          record=record, quiet=quiet)
        finally:
            self._persistent_learn = False

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None,
             greedy=False):
        if not getattr(self, "_persistent_learn", False):
            return super()._run(
                dataset, n_steps, n_episodes, render, quiet, record, initial_states, greedy
            )

        logic = self._core_logic
        logic.initialize_run(n_steps, None, None, quiet)
        converter = logic.converter
        device = self.env.info.device
        n_envs = self.env.number

        # Every environment steps at every step: n_steps is a multiple of n_envs
        mask = converter.ones(n_envs, dtype=bool, device=device)
        logic._running_envs = converter.copy(mask)
        logic._n_active_envs = n_envs

        if self._carry is None:
            last = converter.ones(n_envs, dtype=bool, device=device)
        else:
            self._state, self._policy_state, self._episode_steps, last = self._carry

        need_reset = bool(last.any())
        while logic.move_required():
            if need_reset:
                logic._n_reset_envs = int(last.sum())
                self._reset(dataset, None, last, mask)

            samples, step_infos = self._step(self.agent.draw_action, render, record, mask)

            self.callback_step(samples)
            completed = logic.after_step(samples[5])

            dataset.append_vectorized(samples, step_infos, mask)

            last = samples[5]
            need_reset = completed > 0

            if logic.fit_required():
                consumed = dataset.consume(logic.n_steps_per_fit)
                self.agent.fit(consumed.flatten())

                for c in self.callbacks_fit:
                    c(consumed)

                n_carry_forward_steps = dataset.clear(
                    keep_leftovers=True,
                    history_context=self.agent.history_manager.history_context(),
                )
                last = logic.after_fit_vectorized(last, n_carry_forward_steps)

        # Keep the running episodes for the next call
        self._carry = (self._state, self._policy_state, self._episode_steps, last)

        self.agent.stop()
        self.env.stop()
        self._end(record)

        return dataset.flatten()
