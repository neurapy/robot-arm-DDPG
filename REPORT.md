## DRL - DQN - Navigation Control


#### Agent

A DDPG (Deep Deterministic Policy Gradient) Agent was used. Actor & Critic both have 2 hidden layers with 128 units.


### Model Hyperparameters
- Gamma: 0.99
- TAU: 1e-3
- LR-Actor: 2e-4
- LR-Critic: 2e-4
- mu: 0
- theta: 0.15
- sigma: 0.1


## Results
The current configuration solved the task in just 257 episodes!

<table>
  <tr>
    <td><img src="13-avg.png" width="400" height="260" /></td>
    <td><img src="13-total.png" width="400" height="260" /></td>
  </tr>
</table>

## Possible Improvements

There are many ways this could be improved. for example a Prioritized Experience Replay Buffer could be added. In the future i would also like to try true parallel algorithms like A3C.
