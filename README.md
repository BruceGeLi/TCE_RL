<p align="center">
  <img src='./web_assets/Metaworld.gif' width="260" />
  <img src='./web_assets/Box_Pushing.gif' width="260" />
  <img src='./web_assets/Table_Tennis.gif' width="260" />
</p>

## Abstract

Current advancements in reinforcement learning (RL) have predominantly focused on learning step-based policies that generate actions for each perceived state. While these methods efficiently leverage step information from environmental interaction, they often ignore the temporal correlation between actions, resulting in inefficient exploration and unsmooth trajectories that are challenging to implement on real hardware. Episodic RL (ERL) seeks to overcome these challenges by exploring in parameters space that capture the correlation of actions. However, these approaches typically compromise data efficiency, as they treat trajectories as opaque 'black boxes.' In this work, we introduce a novel ERL algorithm, Temporally-Correlated Episodic RL (TCE), which effectively utilizes step information in episodic policy updates, opening the 'black box' in existing ERL methods while retaining the smooth and consistent exploration in parameter space. TCE synergistically combines the advantages of step-based and episodic RL, achieving comparable performance to recent ERL methods while maintaining data efficiency akin to state-of-the-art (SoTA) step-based RL. 

**This work has been accepted by ICLR 2024.** 


<br><br>
![TCE](./web_assets/Framework.png)
<!--- -->

## Citation
todo

<div align="center">
  <br><br>
    <a href='https://github.com/BruceGeLi/Temporally_Correlated_Exploration_RL'><img src='./web_assets/CodeOnGithub.png' width="300px"></a>
</div>

## Installation
todo