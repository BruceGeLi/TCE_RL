<p align="center">
  <img src='./web_assets/Metaworld.gif' width="200" />
  <img src='./web_assets/Box_Pushing.gif' width="200" />
  <img src='./web_assets/Table_Tennis.gif' width="200" />
</p>
<br><br>

## Abstract

Current advancements in reinforcement learning (RL) have predominantly focused
on learning step-based policies that generate actions for each perceived state.
While these methods efficiently leverage step information from environmental
interaction, they often ignore the temporal correlation between actions,
resulting in inefficient exploration and unsmooth trajectories that are
challenging to implement on real hardware. Episodic RL (ERL) seeks to overcome
these challenges by exploring in parameters space that capture the correlation
of actions. However, these approaches typically compromise data efficiency, as
they treat trajectories as opaque 'black boxes.' In this work, we introduce a
novel ERL algorithm, Temporally-Correlated Episodic RL (TCE), which effectively
utilizes step information in episodic policy updates, opening the 'black box' in
existing ERL methods while retaining the smooth and consistent exploration in
parameter space. TCE synergistically combines the advantages of step-based and
episodic RL, achieving comparable performance to recent ERL methods while
maintaining data efficiency akin to state-of-the-art (SoTA) step-based RL.

This work has been accepted by ICLR 2024.

## Framework

<br><br>
![TCE](./web_assets/Framework.png)
<!--- -->

## Code Base and Installation
TBD soon.


## Experiment Results in WandB
[Metaworld (TCE, BBRL)](https://api.wandb.ai/links/gelikit/ypzc58q1)

## Citation
Please consider citing our work if you find it useful:
```
@inproceedings{li2023open,
  title={Open the Black Box: Step-based Policy Updates for Temporally-Correlated Episodic Reinforcement Learning},
  author={Li, Ge and Zhou, Hongyi and Roth, Dominik and Thilges, Serge and Otto, Fabian and Lioutikov, Rudolf and Neumann, Gerhard},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

Previous work for ProDMP:
```
@article{li2023prodmp,
  title={Prodmp: A unified perspective on dynamic and probabilistic movement primitives},
  author={Li, Ge and Jin, Zeqi and Volpp, Michael and Otto, Fabian and Lioutikov, Rudolf and Neumann, Gerhard},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={4},
  pages={2325--2332},
  year={2023},
  publisher={IEEE}
}
```