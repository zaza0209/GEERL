# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:54:38 2024

@author: Lenovo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from datetime import datetime
from sklearn import metrics
import pickle, copy


# %% state function mean
class StateModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def predict(self, state, action):
        raise NotImplementedError("This method should be overridden by the subclass")


class NeuralNetworkStateModel(nn.Module, StateModel):
    def __init__(
        self,
        state_dim,
        action_dim,
        layer_sizes=[1, 32, 1],
        action_type="discrete",
        state_type="discrete",
    ):
        nn.Module.__init__(self)
        StateModel.__init__(self, state_dim, action_dim)
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.layer_sizes = layer_sizes
        self.action_type = action_type
        self.state_type = state_type  # add state type (discrete or continuous)

    def to_one_hot(self, tensor, dim):
        one_hot = torch.zeros(tensor.size(0), dim).to(tensor.device)
        one_hot.scatter_(1, tensor.unsqueeze(1), 1.0)
        return one_hot

    def forward(self, s, a):
        if type(s) == int:
            s = torch.tensor([s])
        if type(a) == int:
            a = torch.tensor([a])
        if (
            self.state_type == "discrete"
        ):  # convert state to one-hot vector if it's discrete
            s = self.to_one_hot(s, self.state_dim)
        if self.action_type == "discrete":  # same for action
            a = self.to_one_hot(a, self.action_dim)

        x = torch.cat([s.float(), a.float()], dim=1)
        for layer in self.layers[:-1]:  # Apply all layers but last with ReLU activation
            x = nn.ReLU()(layer(x))

        if (
            self.state_type == "discrete"
        ):  # use softmax for discrete states, output probability over states
            x = nn.functional.softmax(self.layers[-1](x), dim=1)
        else:  # continuous states
            x = self.layers[-1](x)

        return x

    def predict_mean(self, s, a):
        if type(s) == int:
            s = torch.tensor([s])
        if type(a) == int:
            a = torch.tensor([a])
        if len(s.shape) == 1:
            s = s.reshape(1, -1)
        if type(s) is not torch.Tensor:
            s = torch.tensor(s, dtype=torch.float32)
        if type(a) is not torch.Tensor:
            if self.action_type == "discrete":
                a = torch.tensor(a, dtype=torch.int32)
            else:
                a = torch.tensor(a, dtype=torch.float32)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        with torch.no_grad():
            next_state_probs = self.forward(s, a).detach().numpy()

        # If the state type is discrete, sample the next state from the output probabilities
        if self.state_type == "discrete":
            next_state = np.random.choice(self.state_dim, p=next_state_probs[0])

        # For the continuous case, the output is a deterministic next state
        else:
            next_state = next_state_probs

        return next_state

    def predict(self, state, action):
        means = self.predict_mean(state, action)

        # Ensure means is a 2D array for consistent looping
        if means.ndim == 1:
            means = means.reshape(1, -1)

        samples = []
        for mean in means:
            sample = np.random.multivariate_normal(
                mean, self.cov * np.identity(means.shape[1])
            )
            samples.append(sample)

        return np.array(samples)

    def fit(
        self,
        states,
        actions,
        transformed_next_states,
        validation_data,
        lr=0.001,
        batch_size=32,
        num_epochs=50,
        patience=5,
        estimate_constant_covariance=1,
        param_convergence_patience=5,
        convergence_threshold=1e-5,
    ):
        """

        Parameters
        ----------
        validation_data : a List of data for validation to implement early stopping = [state, action, next_state]
        patience : TYPE, optional
            DESCRIPTION. The default is 5.
        estimate_constant_covariance : TYPE, optional
            DESCRIPTION. The default is 1.

        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        if self.state_type == "discrete":  # use cross-entropy for discrete states
            loss_fn = nn.CrossEntropyLoss()
        else:  # continuous states
            loss_fn = nn.MSELoss()

        # Parameter convergence initialization
        param_patience_count = 0
        prev_state_dict = {
            name: param.clone() for name, param in self.state_dict().items()
        }

        best_loss = np.inf
        patience_count = 0
        best_model = None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(states), batch_size):
                batch_states = torch.from_numpy(states[i : i + batch_size])
                batch_actions = torch.from_numpy(
                    actions[i : i + batch_size].reshape(-1, self.action_dim)
                )
                batch_transformed_next_states = torch.from_numpy(
                    transformed_next_states[i : i + batch_size]
                ).float()
                optimizer.zero_grad()
                predicted_next_states = self.forward(batch_states, batch_actions)
                if (
                    self.state_type == "discrete"
                ):  # use argmax to get the index of the predicted state for discrete states
                    predicted_next_states = predicted_next_states.argmax(dim=1)
                loss = loss_fn(
                    predicted_next_states.float(), batch_transformed_next_states
                )
                loss.backward()
                optimizer.step()

                # Parameter convergence check
                max_param_change = max(
                    (prev_state_dict[name] - param).abs().max().item()
                    for name, param in self.state_dict().items()
                )
                prev_state_dict = {
                    name: param.clone() for name, param in self.state_dict().items()
                }

                if max_param_change < convergence_threshold:
                    param_patience_count += 1
                    if param_patience_count >= param_convergence_patience:
                        print("Stopping training due to parameter convergence.")
                        break
                else:
                    param_patience_count = 0

                epoch_loss += loss.item() * len(batch_states)
            epoch_loss /= len(states)

            # if epoch % 10 == 0:
            #     print("Epoch %d, Loss: %.4f" % (epoch + 1, epoch_loss))

            if epoch == num_epochs - 1:
                print("Epoch %d, Loss: %.4f" % (epoch + 1, epoch_loss))

            if param_patience_count >= param_convergence_patience:
                break

            if validation_data != None:
                val_states, val_actions, _, val_transformed_next_states = (
                    validation_data
                )
                # val_actions = torch.eye(self.action_dim)[val_actions.squeeze()].float() if self.action_type == 'discrete' else torch.from_numpy(val_actions).float()
                predicted_val_next_states = self.forward(
                    torch.from_numpy(val_states),
                    torch.from_numpy(val_actions.reshape(-1, self.action_dim)),
                )
                val_loss = loss_fn(
                    predicted_val_next_states,
                    torch.from_numpy(val_transformed_next_states).float(),
                ).item()

                # Early stopping condition
                # print('val_loss', val_loss, 'best_loss', best_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_count = 0
                    best_model = copy.deepcopy(
                        self.state_dict()
                    )  # save model state with the best validation loss
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        self.validation_loss = best_loss
                        # print('Early stopping. Best validation loss: {:.4f}'.format(best_loss))
                        self.load_state_dict(
                            best_model
                        )  # load best model state before exiting
                        break

        # if estimate_constant_covariance:
        #     # if self.action_type == 'discrete':
        #     #     transformed_actions = torch.eye(self.action_dim)[actions.squeeze()].float()
        #     # else:
        #     #     transformed_actions = torch.from_numpy(actions).float()
        #     predicted_next_states = self.predict_mean(states, actions.reshape(-1,self.action_dim))

        #     prediction_errors = predicted_next_states - next_states
        #     self.cov = np.mean((prediction_errors)**2, axis = 0)

        # else:
        self.cov = 0

    def save(self, filepath):
        """
        Save the model to the specified path.
        """
        # Create a dictionary to save
        model_data = {
            "state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "layer_sizes": self.layer_sizes,
            "action_type": self.action_type,
            "state_type": self.state_type,
            "cov": self.cov,
        }

        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load the model from the specified path.
        """
        model_data = torch.load(filepath)
        # Reconstruct the model
        model = cls(
            state_dim=model_data["state_dim"],
            action_dim=model_data["action_dim"],
            layer_sizes=model_data["layer_sizes"],
            action_type=model_data["action_type"],
            state_type=model_data["state_type"],
        )
        model.load_state_dict(model_data["state_dict"])
        model.cov = model_data["cov"]

        return model

    def evaluate_dis(self, states, actions, next_states):
        predicted_next_states = self.predict(
            states, actions.reshape(-1, self.action_dim)
        )
        prediction_errors = predicted_next_states - next_states
        return np.linalg.norm(prediction_errors, 2) / np.linalg.norm(next_states, 2)

    def evaluate(self, states, actions, next_states):
        predicted_next_states = self.predict_mean(
            states, actions.reshape(-1, self.action_dim)
        )
        prediction_errors = predicted_next_states - next_states
        return np.linalg.norm(prediction_errors, 2) / np.linalg.norm(next_states, 2)


# %%
class TabularStateModel(StateModel):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        # self.transition_matrix = transition_matrix

    def fit(self, states, actions, next_states):
        # Reset the transition probabilities
        self.transition_probabilities = np.zeros(
            (self.state_dim, self.state_dim, self.state_dim)
        )
        for s, a, ns in zip(states, actions, next_states):
            self.transition_probabilities[int(s), int(a), int(ns)] += 1
        # Normalize the transition probabilities
        self.transition_probabilities /= self.transition_probabilities.sum(
            axis=-1, keepdims=True
        )

    # def predict_probability(self, state, action):
    #     return self.transition_probabilities[state, action]

    def predict(self, state, action):
        p = self.transition_probabilities[state, action]
        next_state = np.random.choice(self.state_dim, p=p.flatten())
        return next_state

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.transition_probabilities, f)
        print("Model saved at location:", filepath)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.transition_probabilities = pickle.load(f)
        print("Model loaded from location:", filepath)


# %% state function cov


class SigmaModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(SigmaModel, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim * (state_dim + 1) // 2)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        sigma = torch.zeros((s.shape[0], self.state_dim, self.state_dim)).to(s.device)
        triu_indices = torch.triu_indices(self.state_dim, self.state_dim)
        sigma[:, triu_indices[0], triu_indices[1]] = x
        sigma += torch.transpose(sigma, 1, 2) - torch.diag_embed(
            torch.diagonal(sigma, dim1=1, dim2=2)
        )
        sigma[:, torch.arange(self.state_dim), torch.arange(self.state_dim)] = (
            torch.exp(
                sigma[:, torch.arange(self.state_dim), torch.arange(self.state_dim)]
            )
        )
        return sigma

    def predict(self, s, a):
        if len(s.shape) == 1:
            s = s.reshape(1, -1)
        if len(a.shape) == 1:
            a = a.reshape(1, -1)
        if type(s) is not torch.Tensor:
            s = torch.tensor(s, dtype=torch.float32)
        if type(a) is not torch.Tensor:
            a = torch.tensor(a, dtype=torch.float32)
        cov = self.forward(s, a).detach().numpy()
        return cov

    def fit(self, states, actions, residuals, lr=1e-3, batch_size=32, num_epochs=50):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(residuals), batch_size):
                batch_states = states[i : i + batch_size]
                batch_actions = actions[i : i + batch_size]
                batch_residuals = residuals[i : i + batch_size]
                optimizer.zero_grad()
                sigma_pred = self.forward(
                    torch.from_numpy(batch_states).float(),
                    torch.from_numpy(batch_actions).float(),
                )
                loss = loss_fn(
                    torch.from_numpy(
                        batch_residuals[:, :, None] * batch_residuals[:, None, :]
                    ).float(),
                    sigma_pred,
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_residuals)
            epoch_loss /= len(states)
            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss)
            )

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))


# %% reward function
class RewardModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def predict(self, state_dim, action_dim):
        raise NotImplementedError("This method should be overridden by the subclass")


class NeuralNetworkRewardModel(nn.Module, RewardModel):
    def __init__(
        self,
        state_dim,
        action_dim,
        layer_sizes=[1, 32, 1],
        action_type="discrete",
        loss_type="mse",
        lr=0.001,
    ):
        nn.Module.__init__(self)
        RewardModel.__init__(self, state_dim, action_dim)

        self.action_type = action_type

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.epoch = 0
        self.loss_type = loss_type
        if loss_type == "mse":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def to_one_hot(self, tensor, dim):
        one_hot = torch.zeros(tensor.size(0), dim).to(tensor.device)
        one_hot.scatter_(1, tensor.unsqueeze(1), 1.0)
        return one_hot

    def forward(self, s, a):
        if self.action_type == "discrete":  # same for action
            a = self.to_one_hot(a, self.action_dim)
        x = torch.cat([s.float(), a.float()], dim=1)
        # x = nn.functional.relu(self.fc1(x))
        # x = self.fc2(x)
        for layer in self.layers[:-1]:  # Apply all layers but last with ReLU activation
            x = nn.ReLU()(layer(x))
        r = self.layers[-1](x)
        return r.squeeze()

    def predict(self, s, a):
        # if type(s) == int:
        #     s = torch.tensor([s])
        # if type(a) == int:
        #     a = torch.tensor([a])
        if len(s.shape) < 2:
            raise ValueError(
                "Reshape your state either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
            )
        if len(a.shape) < 2:
            raise ValueError(
                "Reshape your action either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
            )
        if type(s) is not torch.Tensor:
            s = torch.tensor(s, dtype=torch.float32)
        if type(a) is not torch.Tensor:
            if self.action_type == "discrete":
                a = torch.tensor(a, dtype=torch.int32)
            else:
                # if isinstance(a, (int, np.int32, np.int64)):
                #     a = np.array([a])
                a = torch.tensor(a, dtype=torch.float32)
        # if len(a.shape) == 1:
        #     a = a.unsqueeze(0)
        # if len(s.shape) != len(a.shape):
        #     a=a.reshape(s.shape)
        with torch.no_grad():
            reward = self.forward(s, a).detach().numpy()
        return reward.flatten()  # .item()

    def fit(
        self,
        states,
        actions,
        rewards,
        validation_data,
        batch_size=32,
        num_epochs=50,
        patience=5,
        param_convergence_patience=5,
        convergence_threshold=1e-5,
    ):
        """

        Parameters
        ----------
        validation_data : A list of data for validation to realize early stopping =[state, action, rewards]
        patience : TYPE, optional
            DESCRIPTION. The default is 5.

        """
        # optimizer = optim.Adam(self.parameters(), lr=lr)
        # # loss_fn = nn.MSELoss()
        # loss_fn = nn.BCEWithLogitsLoss()

        # Early stopping initialization
        best_loss = np.inf
        patience_count = 0

        # Parameter convergence initialization
        param_patience_count = 0
        prev_state_dict = {
            name: param.clone() for name, param in self.state_dict().items()
        }

        # Store the best model
        best_model = None

        for epoch in range(self.epoch, self.epoch + num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(states), batch_size):
                batch_states = torch.from_numpy(states[i : i + batch_size])
                batch_actions = torch.from_numpy(
                    actions[i : i + batch_size].reshape(-1, self.action_dim)
                )
                batch_rewards = torch.from_numpy(rewards[i : i + batch_size]).float()
                self.optimizer.zero_grad()
                predicted_rewards = self.forward(batch_states, batch_actions)
                loss = self.loss_fn(predicted_rewards, batch_rewards)
                loss.backward()
                self.optimizer.step()

                # Parameter convergence check
                max_param_change = max(
                    (prev_state_dict[name] - param).abs().max().item()
                    for name, param in self.state_dict().items()
                )
                prev_state_dict = {
                    name: param.clone() for name, param in self.state_dict().items()
                }

                if max_param_change < convergence_threshold:
                    param_patience_count += 1
                    if param_patience_count >= param_convergence_patience:
                        print("Stopping training due to parameter convergence.")
                        break
                else:
                    param_patience_count = 0

                epoch_loss += loss.item() * len(batch_states)
            epoch_loss /= len(states)
            print("Epoch %d, Loss: %f" % (epoch + 1, epoch_loss))

            if param_patience_count >= param_convergence_patience:
                break

            if validation_data != None:
                # Validation Loss calculation
                val_states, val_actions, val_rewards, _ = validation_data
                predicted_val_rewards = self.forward(
                    torch.from_numpy(val_states),
                    torch.from_numpy(val_actions.reshape(-1, self.action_dim)),
                )
                val_loss = self.loss_fn(
                    predicted_val_rewards, torch.from_numpy(val_rewards).float()
                ).item()

                # Early stopping condition
                print("val_loss", val_loss, "best_loss", best_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_count = 0
                    best_model = copy.deepcopy(
                        self.state_dict()
                    )  # save model state with the best validation loss
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        self.validation_loss = best_loss
                        print(
                            "Early stopping. Best validation loss: {:.4f}".format(
                                best_loss
                            )
                        )
                        self.load_state_dict(
                            best_model
                        )  # load best model state before exiting
                        # optimizer = optim.Adam(self.parameters(), lr=0.01*lr)
                        break

            self.epoch = epoch

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

    def evaluate(self, states, actions, rewards):
        if self.loss_type == "mse":
            predicted_rewards = self.predict(states, actions)
            return np.linalg.norm(rewards - predicted_rewards, 2) / np.linalg.norm(
                rewards, 2
            )
        else:
            # use model to predict probability that given y value is 1
            y_pred_proba = self.predict(states, actions)

            # calculate AUC of model
            auc = metrics.roc_auc_score(rewards, y_pred_proba)

            # print AUC score
            return auc
