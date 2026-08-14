# IPPS: Integrated Process Planning and Scheduling

## Project Overview

Research project focusing on optimizing manufacturing processes through Integrated Process Planning and Scheduling (IPPS) using meta-heuristic algorithms and deep learning models. This research explores solutions to complex combinatorial optimization problems in manufacturing environments.

**Organization**: Production Process Scheduling Lab, Hongik University  
**Project Duration**: 2017-01 to 2017-12  
**Your Role**: Undergraduate Researcher  
**Research Advisor**: Prof. [Advisor Name]  
**Status**: Completed research project

---

## Problem Statement

Manufacturing process optimization involves coordinating:
- **Process Planning**: Determining how to manufacture a product
- **Job Scheduling**: Assigning jobs to machines with time constraints
- **Resource Allocation**: Managing limited machine and tool resources

The IPPS problem is NP-hard, requiring heuristic and advanced optimization techniques.

---

## Research Objectives

1. **Evaluate meta-heuristic algorithms** for IPPS optimization
   - Genetic Algorithm (GA)
   - Ant Colony Optimization (ACO)
   - Comparison with baseline approaches

2. **Explore reinforcement learning (RL)** for scheduling
   - Deep learning models for dynamic scheduling decisions
   - Joint models combining ACO + RL

3. **Benchmark against TSP (Traveling Salesman Problem)**
   - Use TSP as a simplified test environment
   - Validate algorithm effectiveness before IPPS deployment

---

## Research Methodology

### Phase 1: Meta-Heuristic Algorithms Testing

#### Genetic Algorithm (GA)
- **Implementation**: `ipps/GA.py` - Population-based evolutionary approach
- **Key Components**:
  - `construct_schedule()`: Generates initial population with random job-machine-tool assignments
  - Crossover operators for solution recombination
  - Mutation operators to introduce diversity
  - Fitness evaluation based on makespan minimization
- **Testing Environments**: TSP (50 cities) and IPPS (manufacturing scheduling)

#### Ant Colony Optimization (ACO)
- **Implementation**: Multiple versions across directories
  - `tsp_aco/ACO_v1.py` & `ACO_v2.py`: Classic ACO with pheromone updates
  - `tsp_aco_rl/aco_linear.py`: Linear pheromone encoding
  - `tsp_aco_rl/aco_nn.py`: Neural network-enhanced ACO
- **Key Features**:
  - Pheromone-based distributed search
  - Probabilistic node selection using pheromone and heuristic information
  - Local and global pheromone update strategies
  - Parameter tuning: α (pheromone weight), β (heuristic weight), evaporation rate

### Phase 2: Reinforcement Learning Exploration

#### Standalone RL Models
- **Q-Learning** (`tsp_rl/RL_v1.py`):
  - Q-value table: `q_value[current_node][next_node]`
  - Exploration rate: 0.1 (exploit 90% of the time)
  - Discount rate: 0.9
  - Learns state-action values through trial and error

- **Policy Gradient Methods** (`tsp_rl/RL_v3_policy_gradient.py`):
  - Neural network outputs action probabilities
  - Direct policy optimization using gradient ascent
  - Better for continuous action spaces

- **DQN** (`tsp_rl/RL_v5_dqn.py`):
  - Deep Q-Network with experience replay
  - Target network for stable training
  - Handles larger state spaces than tabular Q-learning

- **A3C** (`tsp_rl/RL_v4_a3c.py`):
  - Asynchronous Advantage Actor-Critic
  - Parallel agents for faster learning
  - Advantage function reduces variance in policy gradient

#### Joint ACO + RL Model
- **Implementation**: `tsp_aco_rl/aco_nn.py`, `aco_linear_backprop.py`
- **Architecture**:
  - ACO generates candidate tour sequences
  - Neural network (RL) evaluates and refines solutions
  - Pheromone updates based on RL feedback
  - Creates feedback loop between exploration (ACO) and exploitation (RL)
- **Benefits**:
  - ACO's population-based search avoids local optima
  - RL's learning improves pheromone guidance over time
  - Synergistic combination for complex IPPS scenarios

---

## Project Structure

```
IPPS/
│
├── ipps/                           # Core IPPS (Manufacturing Scheduling) Implementation
│   ├── Main.py                    # Entry point - runs GA on IPPS problem
│   ├── IPPS.py                    # Core algorithm - Operation & Schedule classes
│   ├── GA.py                      # Genetic Algorithm implementation
│   ├── ACO.py                     # Ant Colony Optimization for IPPS
│   ├── Data.py                    # Problem data structures & constants
│   ├── Params.py                  # Configuration parameters
│   └── *.txt                      # Problem instances (machines, jobs, operations)
│
├── tsp_nn/                         # TSP - Neural Network Baseline
│   ├── NeuralNet.py               # Multi-layer perceptron for TSP
│   ├── LinearRegression.py        # Linear baseline comparison
│   └── _data*.txt                 # TSP instances (city coordinates)
│
├── tsp_rl/                         # TSP - Reinforcement Learning Approaches
│   ├── RL_v1.py                   # Basic Q-Learning with q-value table
│   ├── RL_v2.py                   # Enhanced Q-Learning variant
│   ├── RL_v3_policy_gradient.py   # Policy Gradient method
│   ├── RL_v4_a3c.py               # Asynchronous Advantage Actor-Critic
│   ├── RL_v5_dqn.py               # Deep Q-Network
│   ├── data_model.py              # State representation & environment
│   └── _data*.txt                 # TSP problem instances
│
├── tsp_aco/                        # TSP - Ant Colony Optimization
│   ├── ACO_v1.py                  # Classic ACO with pheromone trails
│   ├── ACO_v2.py                  # Improved ACO variant
│   └── _data*.txt                 # TSP instances
│
└── tsp_aco_rl/                     # TSP - Hybrid ACO + RL Approach
    ├── aco_linear.py              # ACO with linear pheromone encoding
    ├── aco_linear_bias.py         # ACO with bias term
    ├── aco_linear_backprop.py     # ACO with backpropagation learning
    ├── aco_nn.py                  # ACO with neural network integration
    └── _data*.txt                 # TSP instances

Key Files:
- ipps/IPPS.py: Schedule class with {seq, st, ct, slot_used, tool_used, makespan}
- tsp_rl/RL_v1.py: Q-learning with graph[n_cities][2], q_value[n][n], visited[n]
- tsp_aco/ACO_v1.py: Pheromone matrix tau[n][n], heuristic η[n][n]
```

---

## Key Technologies

| Category | Technology |
|----------|-----------|
| **Language** | Python |
| **ML Framework** | Keras, TensorFlow |
| **Algorithms** | Genetic Algorithm, ACO, Reinforcement Learning |
| **Environment** | OpenAI Gym-style interfaces |
| **Optimization** | NumPy, SciPy |

---

## Experimental Results

### TSP Environment (N=50 cities)
- **ACO**: Competitive solutions with fast convergence
- **RL Model**: Improved performance with training
- **ACO+RL Hybrid**: Best performance combining both approaches

### IPPS Environment
- Successfully applied meta-heuristic algorithms to manufacturing scenarios
- RL models showed promise in learning job-machine affinity
- Hybrid approach provided balanced exploration-exploitation trade-off

---

## Key Insights

1. **Meta-heuristic effectiveness**: ACO outperforms GA for routing/scheduling problems
2. **RL potential**: Neural networks can learn scheduling patterns with sufficient training
3. **Hybrid approach benefits**: Combining ACO + RL leverages strengths of both methods
4. **Problem complexity**: IPPS is significantly harder than TSP due to resource constraints
5. **Scalability considerations**: Algorithm performance degrades with problem size

---

## Lessons Learned

- **Algorithm Design**: Understanding strengths/weaknesses of different optimization approaches
- **Research Methodology**: Designing fair experiments and controlled environments
- **Deep Learning Application**: Applying RL to combinatorial optimization problems
- **Performance Analysis**: Benchmarking and comparing different solution approaches

---

## Research Publications

Results from this research were presented at:
- Hongik University Research Symposium
- Production Process Scheduling Lab seminars

---

## Future Research Directions

1. **Scalability**: Test algorithms on larger IPPS instances
2. **Real-world deployment**: Validate on actual manufacturing data
3. **Advanced RL techniques**: Use policy gradient methods (A3C, PPO)
4. **Hybrid methods**: Explore more sophisticated ACO+RL integration
5. **Machine learning for parameter tuning**: Auto-tune algorithm hyperparameters

---

## Code and Implementation Notes

- **Python 3.6+** required
- **Keras backend**: TensorFlow
- **Development environment**: Jupyter notebooks and command-line scripts
- **Reproducibility**: Random seeds set for deterministic results

### Running the Experiments

```bash
# TSP with ACO
python tsp_aco/aco_solver.py

# TSP with RL
python tsp_rl/train.py

# IPPS with Joint ACO+RL
python tsp_aco_rl/hybrid_solver.py
```

---

## Project Takeaways

This research provided foundational knowledge in:
- Meta-heuristic algorithm implementation and comparison
- Reinforcement learning application to optimization problems
- Research methodology and experimental design
- Python scientific computing and machine learning frameworks

The hybrid ACO+RL approach showed that combining domain-specific algorithms with learned policies can be effective for complex manufacturing optimization problems.

---

*Project completed: December 2017*  
*Repository**: [GitHub - IPPS](https://github.com/hslee1064/IPPS)
