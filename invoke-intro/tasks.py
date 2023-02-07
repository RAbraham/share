from invoke import task
from activity import train

@task
def say_hi(context, name):
    # function names are underscore, cli invocations are hyphen
    # context object
    context.run(f"echo 'Hi {name}'")


@task(iterable=['learning_rate'])
def do(context, activity, epochs=20, debug=False, profile=True, learning_rate=None, path=None):
    """
    Train or Predict Activity

    :param context: Invoke context object
    :param activity: train or predict
    :param epochs: number of epochs
    :param debug: print debug statements
    :param profile: profile memory
    :param learning_rate: model learning rate
    :param path: output folder
    """

    context.run("echo 'Starting Training'")
    if activity == 'train':
        train(epochs=epochs, debug=debug, profile=profile, learning_rates=learning_rate, path=path)


