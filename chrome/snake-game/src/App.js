import React, { Component } from 'react';
import Snake from './Snake';
import Food from './Food';
import mqtt from "mqtt";

const URL = 'ws://127.0.0.1:8883'
const STATE_TOPIC = "rl/state"
const ACTION_TOPIC = "rl/action"
const RESET_TOPIC = "rl/reset"

const percentSize = 10

const getRandomCoordinates = () => {
  let min = 0;
  let max = 98;
  let x = Math.floor((Math.random() * (max - min + 1) + min) / percentSize) * percentSize;
  let y = Math.floor((Math.random() * (max - min + 1) + min) / percentSize) * percentSize;
  return [x, y]
}

const initialState = {
  food: getRandomCoordinates(),
  snakeDots: [
    [0, 0],
    [percentSize, 0]
  ]
}

class App extends Component {

  constructor(props) {
    super(props);

    this.client = mqtt.connect(URL)
    this.direction = 'RIGHT'
    this.done = false
    this.didAction = false
    this.didReset = false
    this.episode = 0
    this.highScore = 0
    this.oldState = null
  }

  state = initialState;

  componentDidMount() {
    if (this.client) {
      this.client.on('connect', () => {
        console.log('Connected');
      });
      this.client.on('error', (err) => {
        console.error('Connection error: ', err);
        this.client.end();
      });
      this.client.on('reconnect', () => {
        console.log('Reconnecting');
      });
      this.client.on('message', (topic, message) => {
        this.onMessage(topic, message)
      });
      this.client.subscribe(ACTION_TOPIC, (error) => {
        if (error) {
          console.log('Subscribe to topics error', error)
          return
        }
      });
      this.client.subscribe(RESET_TOPIC, (error) => {
        if (error) {
          console.log('Subscribe to topics error', error)
          return
        }
      });
    }
  }

  componentDidUpdate() {
    this.checkIfOutOfBorders();
    this.checkIfCollapsed();
    this.checkIfEat();
    if (this.didAction || this.didReset) {
      if (this.didAction) {
        this.didAction = false
      }
      if (this.didReset) {
        this.didReset = false
        this.didAction = true
      }
      if (this.client) {
        let payload = null
        if (!this.done) {
          payload = {
            rewardIndicators: {
              length: this.state.snakeDots.length - 2
            },
            observation: {
              field_size: Math.floor(100 / percentSize),
              snake_dots: this.state.snakeDots,
              food_x: Math.floor(this.state.food[0] / percentSize),
              food_y: Math.floor(this.state.food[1] / percentSize),
              head_x: Math.floor(this.state.snakeDots[this.state.snakeDots.length - 1][0] / percentSize),
              head_y: Math.floor(this.state.snakeDots[this.state.snakeDots.length - 1][1] / percentSize),
              direction: this.direction
            },
            done: false
          }
        } else {
          payload = {
            rewardIndicators: {
              length: this.oldState.snakeDots.length - 2
            },
            observation: {
              field_size: Math.floor(100 / percentSize),
              snake_dots: this.oldState.snakeDots,
              food_x: Math.floor(this.oldState.food[0] / percentSize),
              food_y: Math.floor(this.oldState.food[1] / percentSize),
              head_x: Math.floor(this.oldState.snakeDots[this.oldState.snakeDots.length - 1][0] / percentSize),
              head_y: Math.floor(this.oldState.snakeDots[this.oldState.snakeDots.length - 1][1] / percentSize),
              direction: this.direction
            },
            done: true
          }
        }
        const json = JSON.stringify(payload);
        this.client.publish(STATE_TOPIC, json, error => {
          if (error) {
            console.log('Publish error: ', error);
          }
        });
      }
    }
  }

  onMessage = (topic, message) => {
    if (topic === RESET_TOPIC) {
      this.onReset()
    }
    if (topic === ACTION_TOPIC) {
      this.onAction(message.toString().toUpperCase())
    }
  }

  onAction = (action) => {
    if (action !== this.direction) {
      switch (action) {
        case 'UP':
          if (this.direction !== 'DOWN') {
            this.direction = 'UP'
          }
          break;
        case 'DOWN':
          if (this.direction !== 'UP') {
            this.direction = 'DOWN'
          }
          break;
        case 'LEFT':
          if (this.direction !== 'RIGHT') {
            this.direction = 'LEFT'
          }
          break;
        case 'RIGHT':
          if (this.direction !== 'LEFT') {
            this.direction = 'RIGHT'
          }
          break;
        default:
          break;
      }
    }
    this.oldState = this.state
    this.moveSnake()
    this.didAction = true
  }

  moveSnake = () => {
    let dots = [...this.state.snakeDots];
    let head = dots[dots.length - 1];

    switch (this.direction) {
      case 'RIGHT':
        head = [head[0] + percentSize, head[1]];
        break;
      case 'LEFT':
        head = [head[0] - percentSize, head[1]];
        break;
      case 'DOWN':
        head = [head[0], head[1] + percentSize];
        break;
      case 'UP':
        head = [head[0], head[1] - percentSize];
        break;
      default:
        break;
    }
    dots.push(head);
    dots.shift();
    this.setState({
      snakeDots: dots
    })
  }

  checkIfOutOfBorders() {
    let head = this.state.snakeDots[this.state.snakeDots.length - 1];
    if (head[0] >= 100 || head[1] >= 100 || head[0] < 0 || head[1] < 0) {
      this.done = true
    }
  }

  checkIfCollapsed() {
    let snake = [...this.state.snakeDots];
    let head = snake[snake.length - 1];
    snake.pop();
    snake.forEach(dot => {
      if (head[0] === dot[0] && head[1] === dot[1]) {
        this.done = true
      }
    })
  }

  checkIfEat() {
    let head = this.state.snakeDots[this.state.snakeDots.length - 1];
    let food = this.state.food;
    if (head[0] === food[0] && head[1] === food[1]) {
      this.setState({
        food: getRandomCoordinates()
      })
      this.enlargeSnake();
    }
  }

  enlargeSnake() {
    let newSnake = [...this.state.snakeDots];
    newSnake.unshift([])
    this.setState({
      snakeDots: newSnake
    })
  }

  onReset() {
    this.done = false
    this.didReset = true
    this.episode += 1
    let score = this.state.snakeDots.length - 2
    if (this.highScore < score) {this.highScore = score}
    this.setState(initialState)
  }

  render() {
    return (
      <>
        <div>
          <h2>
            Food: x:{Math.floor(this.state.food[0] / percentSize)} y:{Math.floor(this.state.food[1] / percentSize)}  
            Head: x:{Math.floor(this.state.snakeDots[this.state.snakeDots.length - 1][0] / percentSize)} y:{Math.floor(this.state.snakeDots[this.state.snakeDots.length - 1][1] / percentSize)}  
          </h2>
          <h2>
          Episode: {this.episode}  HighScore: {this.highScore}  Score: {this.state.snakeDots.length - 2}  
          </h2>
        </div>
        <div className="game-area">
          <Snake snakeDots={this.state.snakeDots} size={percentSize}/>
          <Food dot={this.state.food} size={percentSize}/>
        </div>
      </>
    );
  }
}

export default App;
