# GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation

Code for our paper [GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation](as). 

Mukul Khanna*, Ram Ramrakhya*, Gunjan Chhablani, Sriram Yenamandra, Theophile Gervet, Matthew Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi

<p align="center">
    <a href="https://mukulkhanna.github.io/goat-bench/">Project Page</a>
</p>

## GOAT-Bench

<p align="center">
  <img src="imgs/teaser.jpg" width="700">
  <p align="center">Sample episode from GOAT-Bench</p>  
</p>


GOAT-Bench is a benchmark for the Go to Any Thing (GOAT) task where an agent is spawned randomly in an unseen indoor environment and tasked with sequentially navigating to a variable number (in 5-10) of goal objects, described via the category name of the object (e.g. `couch`), a language description (e.g. `a black leather couch next to coffee table`), or an image of the object uniquely identifying the goal instance in the environment. We refer to finding each goal in a GOAT episode as a subtask. Each GOAT episode comprises 5 to 10 subtasks. We set up the GOAT task in an open-vocabulary setting; unlike many prior works, we are not restricted to navigating to a predetermined, closed set of object categories. The agent is expected to reach the goal object $g^k$ for the $k^{th}$ subtask as efficiently as possible within an allocated time budget. Once the agent completes the $k^{th}$ subtask by reaching the goal object or exhausts the allocated time budget, the agent receives next goal $g^{k+1}$ to navigate to. We use HelloRobot's Stretch robot embodiment for the GOAt agent. The agent has a height of 1.41m and base radius of 17cm. At each timestep, the agent has access to an 360 x 640 resolution RGB image $I_t$, depth image $D_t$, relative pose sensor with GPS+Compass information $P_t = (\delta x, \delta y, \delta z)$ from onboard sensors, as well as the current subtask goal $g^{k}_t$, $k$  $\forall$ $\{1, 2,...,5-10\}$. The agent's action space comprises move forward (by 0.25m), turn left and right (by 30º), look up and down (by 30º), and stop actions. A sub-task in a GOAT episode is deemed successful when the agent calls stop action within 1 meter euclidean distance from the current goal object instance – within a budget of 500 agent actions (per sub task).


## :hammer: Installation

Create the conda environment and install all of the dependencies. Mamba is recommended for faster installation:
```bash
# Record of how the environment was set up
# Create conda environment. Mamba is recommended for faster installation.
conda_env_name=goat
mamba create -n $conda_env_name python=3.7 cmake=3.14.0 -y
mamba install -n $conda_env_name \
  habitat-sim=0.2.3 headless pytorch cudatoolkit=11.3 \
  -c pytorch -c nvidia -c conda-forge -c aihabitat -y

# Install this repo as a package
mamba activate $conda_env_name
pip install -e .

# Install habitat-lab
git clone --branch v0.2.3 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install ftfy regex tqdm GPUtil trimesh seaborn timm scikit-learn einops transformers
```


## :floppy_disk: Dataset

- Download the HM3D dataset using the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) (download the full HM3D dataset for use with habitat)

- Move the HM3D scene dataset or create a symlink at `data/scene_datasets/hm3d`.

- Download the GOAT-Bench episode dataset from [here]().

### Dataset Folder Structure

The code requires the datasets in `data` folder in the following format:

  ```bash
  ├── goat-bench/
  │  ├── data
  │  │  ├── scene_datasets/
  │  │  │  ├── hm3d/
  │  │  │  │  ├── JeFG25nYj2p.glb
  │  │  │  │  └── JeFG25nYj2p.navmesh
  │  │  ├── datasets
  │  │  │  ├── goat_bench/
  │  │  │  │  ├── hm3d/
  │  │  │  │  │  ├── train/
  │  │  │  │  │  ├── val_seen/
  │  │  │  │  │  ├── val_seen_synonyms/
  │  │  │  │  │  ├── val_unseen/
  ```

## :bar_chart: Training

Run the following to train:
```bash
sbatch scripts/train/2-goat-ver.sh
```

### Evaluation

Run the following to evaluate:
```bash
sbatch scripts/eval/2-goat-eval.sh
```