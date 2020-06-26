import matplotlib.pyplot


def state_to_image(state, identifier):
    i = 0
    for channel in state[0]:
        matplotlib.pyplot.imsave("state-" + identifier + "-" + str(i) + ".png", channel.cpu())
        i += 1
