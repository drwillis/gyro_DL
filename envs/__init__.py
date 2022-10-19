from gym.envs.registration import register

register(
    id='MyPendulum-v1',
    entry_point='envs.pendulum:PendulumEnvV1',
)
register(
    id = 'MyGyro-v1',
    entry_point = 'envs.gyro:GyroEnvV1',
)