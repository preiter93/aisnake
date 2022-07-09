from aisnake.ai_game import AIGame
from aisnake.ai_snake import AISnake

snake = AISnake.new_random()
snake.neural_net.load("model_gen_0250_best.pth")
game = AIGame(snake)
game.play_ai(visible=True, save_game=True)
