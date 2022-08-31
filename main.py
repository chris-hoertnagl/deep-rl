import python_snake_env
import snake_python_agent as agent


if __name__ == '__main__':
    env = python_snake_env.SnakeEnv(tiles=5)
    agent = agent.DQNAGENT(env=env)
    agent.train()
    