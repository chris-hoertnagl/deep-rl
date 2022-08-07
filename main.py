import browser_environment
import snake_dqn_pixel_agent as agent
import tensorflow as tf;


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    env = browser_environment.ChromeEnv()
    agent = agent.DQNAGENT(env=env)
    agent.train()
    