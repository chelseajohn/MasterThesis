import numpy as np


def hamiltonian(model, sample_size, leapfrog_steps, stepsize, surrogate=None, start=None, save_gradient=False):
    """
    Hamiltonian Monte Carlo sampler.

    Args:
        model: model must have get_dim(), energy() and gradient() methods
        sample_size: number of HMC draws
        leapfrog_steps: number of leapfrog steps for a single trajectory
        stepsize: leapfrog step size
        surrogate: surrogate for gradient approximation

    Note:
       traj_length and leapfrog_steps plays a crucial role for  satisfactory acceptance probability.

    Returns:
        HMC draws and training data.
    """

    dim = model.get_dim()

    samples = np.zeros((sample_size, dim))
    energy = np.zeros(sample_size)

    if save_gradient:
        training = np.zeros((sample_size * leapfrog_steps, dim))
        gradient = np.zeros((sample_size * leapfrog_steps, dim))

    if start is None:
        current_position = np.random.normal(0, 1, dim)
    else:
        current_position = start
    current_U = model.energy(current_position)

    proposed = 0
    accepted = 0
    total = 0
    total_count = 0
    total_acc = 0
    

    for i in range(sample_size):
        proposed_position = current_position
        proposed_momentum = np.random.normal(0, 1, dim)
        current_momentum = proposed_momentum

        L = int(np.random.uniform(0, 1) * leapfrog_steps)
        for l in range(L):
            if surrogate is None:
                proposed_position, proposed_momentum = leapfrog(
                    proposed_position, proposed_momentum, stepsize, model)
            else:
                proposed_position, proposed_momentum = leapfrog(
                    proposed_position, proposed_momentum, stepsize, surrogate)
            if save_gradient:
                training[total, :] = proposed_position
                gradient[total, :] = model.gradient(proposed_position)
                total += 1

        current_H = current_U + 0.5 * np.sum(np.square(current_momentum))
        proposed_U = model.energy(proposed_position)
        proposed_H = proposed_U + 0.5 * np.sum(np.square(proposed_momentum))

        if transition(np.exp(current_H - proposed_H)):
            current_position = proposed_position
            current_U = proposed_U
            accepted += 1

        samples[i, :] = current_position
        energy[i] = current_U
        proposed += 1
        total_count+=1

        if (i + 1) % (sample_size / 10) == 0:
            accepted = accepted * 1.0
            proposed = proposed * 1.0
            total_acc += accepted
            print('{n} iterations with acceptance probability {prob}'.format(
                n=i + 1, prob=accepted / proposed))
            accepted = 0
            proposed = 0
    print("acceptance prob = ",total_acc/total_count)

    if save_gradient:
        training = training[0:total, :]
        gradient = gradient[0:total, :]
        return(samples, energy, training, gradient)
    else:
        return(samples, energy)


def leapfrog(position, momentum, stepsize, system):
    """
    Leapfrog numerical integrator.
    """
    momentum = momentum + 0.5 * stepsize * system.gradient(position)
    position = position + stepsize * momentum
    momentum = momentum + 0.5 * stepsize * system.gradient(position)
    return(position, momentum)


def transition(ratio):
    """
    Metropolis transition.
    """
    if np.isnan(ratio):
        return(False)
    return (np.random.uniform(0.0, 1.0) < np.minimum(1.0, ratio))
