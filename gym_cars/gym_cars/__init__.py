from gym.envs.registration import register


register(
    id="cars-v0",
    entry_point="gym_cars.envs:carsEnv",
    timestep_limit=50)
