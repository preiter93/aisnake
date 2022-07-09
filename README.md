# Game of Snake played by Neural Nets

In progress

## Architecture

- Neural net with input, 2xhidden and output layer

## Algorithm

- Weights are optimized by a genetic algorithm
- Coming: Q-learning

## States

- Obstacle detection: Vision in a radius of N pixels
- Food detection: 4 states, i.e. Ahead / Right / Behind / Left of Snake

Additionally, the Snake is given a *hunger* state, so it must eat within
*max_hunger* steps or it dies. 

![Alt text](screenshot.jpg?raw=true "States")

Here: Vision-Radius = 2

## Example Games

![Alt text](anim.gif?raw=true "Play")

*Hunger death*

## Todo
- Design
- Q-Learning
- Documentation
- Show multiple generations

## Dependencies
- pygame
