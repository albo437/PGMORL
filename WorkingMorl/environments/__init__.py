from gym.envs.registration import register

# Register dummy multi-objective environment with different objective numbers
register(
    id='MO-Dummy-v0',
    entry_point='environments.dummy_mo_env:DummyMOEnv',
    max_episode_steps=500,
    kwargs={'obj_num': 2}
)

register(
    id='MO-Dummy3-v0',
    entry_point='environments.dummy_mo_env:DummyMOEnv',
    max_episode_steps=500,
    kwargs={'obj_num': 3}
)

register(
    id='MO-Dummy4-v0',
    entry_point='environments.dummy_mo_env:DummyMOEnv',
    max_episode_steps=500,
    kwargs={'obj_num': 4}
)

register(
    id='MO-Dummy5-v0',
    entry_point='environments.dummy_mo_env:DummyMOEnv',
    max_episode_steps=500,
    kwargs={'obj_num': 5}
)

print("Dummy multi-objective environment registered successfully")
print("Available environments: MO-Dummy-v0")

