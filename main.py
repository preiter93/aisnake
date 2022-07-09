# from snakey.ai.nnet import EvolvingNeuralNet
from aisnake.ai_game import AIGame
from aisnake.ai_snake import AISnake
from aisnake.evolpy.evolution import Evolution
import matplotlib.pyplot as plt
import numpy as np

# Save Neural net of best model at Gen X
SAVE_GEN = [0, 10, 20, 50, 100, 150, 200, 250, 260, 270, 280, 290, 300]

# Play best model ever N steps
WATCH_EVERY = 10

# # nnet = NeuralNet4Layer([5,4,4,4])
# snake_ai = AISnake.new_random()
# # snake_ai.neural_net.save()
# snake_ai.neural_net.load()
# game = AIGame()
# game.play_ai(snake_ai.neural_net, visible=True)

avg_fit_history = []
max_fit_history = []
evolution = Evolution(AISnake)


def my_callback(pop, fit, gen):
    """
    Callback function.

    :pop: list of population
    :fit: list of populations fitness
    :gen: int, number of generation
    """
    max_fit = max(fit)
    avg_fit = sum(fit) / len(fit)
    print(
        "Max Fit: {:4.2f}".format(max_fit),
        "Avg Fit: {:4.2f}".format(avg_fit),
        "in Gen:",
        gen,
    )
    # Plot fitness
    if len(avg_fit_history) % WATCH_EVERY == 0:
        max_index = fit.index(max(fit))
        best_model = pop[max_index]
        game = AIGame(best_model)
        game.play_ai(visible=True)
        # Plot
        x = [i for i in range(len(avg_fit_history))]
        plt.scatter(x, avg_fit_history)
        plt.scatter(x, max_fit_history)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    # Append
    avg_fit_history.append(avg_fit)
    max_fit_history.append(max_fit)
    # Save neural nets
    if len(avg_fit_history) in SAVE_GEN:
        fname = "model_gen_{}_best.pth".format(str(len(avg_fit_history)).zfill(4))
        print("Save {}".format(fname))
        best_model.neural_net.save(file_name=fname)
        for i, model in enumerate(pop):
            fname = "model_gen_{}_{}.pth".format(
                str(len(avg_fit_history)).zfill(4), str(i).zfill(4)
            )
            model.neural_net.save(file_name=fname)
        np.savetxt("fitness_avg", np.array(avg_fit_history))
        np.savetxt("fitness_max", np.array(max_fit_history))
    return


# Start from old population that has been saved
old_pop = []
for i in range(1000):
    snake_ai = AISnake.new_random()
    fname = "model_gen_{}_{}.pth".format(str(250).zfill(4), str(i).zfill(4))
    snake_ai.neural_net.load(file_name=fname)
    old_pop.append(snake_ai)
avg_fit_history = list(np.loadtxt("fitness_avg"))
max_fit_history = list(np.loadtxt("fitness_max"))

pop = evolution.optimize(
    population_size=1000,
    max_generations=200,
    preservation_rate=0.1,
    callback=my_callback,
    mutation_rate=0.1,
    pop=old_pop,
    # add_individuum=snake_ai,
)

# Postprocess
fit = [ind.get_fitness() for ind in pop]
max_index = fit.index(max(fit))
best_snake = pop[max_index]

# Save Model
best_snake.neural_net.save()


np.savetxt("fitness_avg", np.array(avg_fit_history))
np.savetxt("fitness_max", np.array(max_fit_history))
# import numpy as np
# w = 3
# h = 3
# a = [False] * w * h
# x = 2
# y = 1

# ind = y + x * h
# print(ind, y, x * h)
# a[ind] = True

# a = np.array(a).reshape((w,h))
# print(a.transpose())
# print(a[0,1])
