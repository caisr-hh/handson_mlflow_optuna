# import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import requests
import json
import torch


def plot_res(model, data):
    # Plots the result of the predictions.

    steps = 100
    grid = torch.meshgrid(
        torch.linspace(-3, 3, steps=steps),
        torch.linspace(-3, 3, steps=steps),
        indexing="xy",
    )
    x = grid[0].ravel()
    y = grid[1].ravel()
    grid = torch.stack([x, y], dim=-1)
    grid_res = model(grid)

    x = x.reshape(steps, steps)
    y = y.reshape(steps, steps)
    grid_res = grid_res.detach().reshape(steps, steps)
    plt.contourf(x, y, grid_res, cmap="plasma", alpha=0.4)

    inputs = data.test_loader.dataset[:][0]
    labels = data.test_loader.dataset[:][1]

    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap="plasma", s=5)

    fig = plt.gcf()

    return fig


def plot_server_res():
    # Creates a grid of points and calls a local server, before printingg the model output.

    steps = 100
    grid = torch.meshgrid(
        torch.linspace(-1.5, 1.5, steps=steps),
        torch.linspace(-1.5, 1.5, steps=steps),
        indexing="xy",
    )
    x = grid[0].ravel()
    y = grid[1].ravel()
    grid = torch.stack([x, y], dim=-1)

    grid_list = grid.tolist()
    json_dump = json.dumps({"inputs": grid_list})
    response = requests.post(
        "http://127.0.0.1:5001/invocations", data=json_dump, headers={"Content-Type": "application/json"}
    )
    values = response.json()["predictions"]
    plt.scatter(x, y, c=values, cmap="plasma", s=5)

    plt.show()