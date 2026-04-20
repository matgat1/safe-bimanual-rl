from safe_bimanual_rl.reach_point_experiment_sac import experiment


def test_experiment_1_epoch():
    """verify that experiment() runs for 1 epoch without errors."""
    experiment(
        n_epochs=1,
        n_steps=100,
        n_episodes_test=1,
        initial_replay_size=500,
        use_cluster=True,
        save_model=False,
        use_wandb=False
    )
