# Glossary for *Artificial Intelligence: A Modern Approach*

[A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [J](#j) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w) | [X](#x) | [Y](#y) | [Z](#z)

## 8-puzzle

**8-puzzle** consists of a 3x3 grid containing 8 numbered tiles and a blank space. A tile adjacent to the blank space can slide into that space. The object is to reach a specified **goal state** from a given **initial state**.

# A

## absolute error 

Magnitude of the difference between the theoretical value (expected value) and the actual value of a physical quantity. 

## abstraction
## abstraction hierarchy

It hides the complexity of the system and allows individuals to work on different modules of the hierarchy at the same time. 

## accessibility relations
## action monitoring
## action schema
## action-utility function
## actions

The things that an agent can do. We model this with a function, **Actions(s)**, that returns a collection of actions
that the agent can execute in state *s*.

## activation
## activation function
A mathematical function that transforms the input or set of inputs received at a neuron to produce an output. 
Popular examples include the Sigmoid function, Rectificied Linear Units (ReLU) and Hyperbolic Tangent (Tanh)

## active learning

An active learning agent decided which actions to take in order to guide its learning: it values leearning new things
as well as reaping immediate rewards from the environment.
This is in contrast to a passive learning agent, which learns from its observations, but the actions the agent takes are not influenced by the learning process.

## actor
## adaptive dynamic programming

Also known as Approximate Dynamic Programming; it is a type of Reinforcement Learning where local rewards and transitions depend on unknown parameters - we set an initial control policy and update it until it converges to an optimal control policy. 

## add list
## admissible heuristic

A **heuristic** is a function that scores alternatives at each branching in a search algorithm. An **admissible heuristic** is one that *never overestimates* the cost to reach the goal. Admissible heuristics are **optimistic** in nature as they believe the cost of reaching the goal is less than it actually is. 

## adversarial search
## adversary argument
## agent

An **agent** is anything that can be viewed as perceiving its **environment** through **sensors** and acting upon that environment through **actuators**.

## agent function

An agent's behavior is described by the **agent function** that maps any given percept sequence to an action.

## agent program

_Internally,_ the agent function for an artificial agent will be implemented by an **agent program**.

## agglomerative clustering
## aggregation
## algorithm

An **algorithm** is a sequence of **unambiguous finite steps** that when carried out on a given problem produce the expected outcome and terminate in **finite time**.

## alignment method
## alpha-beta
**alpha** (**&alpha;**) is the value of the best (i.e., highest-value) choice we have found so far at any choice point along the path for MAX and **beta**(**&beta;**) is the value of the best (i.e., lowest-value) choice we have found so far at any choice point along the path for MIN in a standard minimax tree.
## alpha-beta pruning
**alpha—beta pruning** is applied to a standard minimax tree to prune away branches that cannot possibly influence the final minimax decision.
## ambient illumination
## ambiguity
## ambiguity aversion
## analogical reasoning
## anchoring effect
## And-Elimination
## AND-parallelism
## angelic nondeterminism
## angelic semantics
## answer literal
## answer set programming
## answer sets
## aortic coarctation
## appearance
## appearance model
## applicable
## apprenticeship learning
## architecture
## arity
## artificial life
## ascending-bid
## Asilomar Principles
## assignment
## associative memory
## asymptotic analysis
## asymptotic bounded optimality
## ATMS
## atom
## atomic representation
## atomic sentence
## attribute-based extraction
## augmented grammar
## authority
## automatic assembly sequencing
## autonomy
## average reward
## axiom

# B

## back-propagation

**back-propagation** is an algorithm used for *supervised learning* of **artificial neural networks** using gradient descent.
The method calculates the gradient of a given error function with respect to the weights of the network. The "backward" terminology stems because the gradient calculation requires backward propagation through the newtork.

## backed-up value
## backgammon
## background subtraction
## backjumping
## backmarking
## backoff model
## backpropagation

To minimize the cost function, we need to know how the changes in weights and biases affect the cost function i.e. partial derivatives of the cost function w.r.t every weight and bias in the network; back propagation is a method that allows us to quickly compute all these partial derivatives. 

## Backus-Naur form (BNF)

It’s a mathematical notation used to describe the syntax of a programming language.  

## backward-chaining
## bag of words
## bagging
## bang-bang control
## baseline
## batch gradient descent
## Bayes' rule

**Bayes' rule** describes the probabilty of an event(lets say A) in the light of that a given event B has already occured.
Mathematically Bayes' rule can be described as :-  **P(A|B) = P(A)P(B|A)/P(B)**

## Bayes-Nash equilibrium
## Bayesian learning

It’s a Machine Learning method which enables us to encode our initial perception of what a model should look like, regardless of what the data tells us. It proves to be very useful when there’s a sparse amount of data to train our model properly. 

## Bayesian network
## beam search
## behaviorism
## belief function
## belief propagation
## belief revision
## belief state
## Bellman equation
## Bellman update
## benchmarking
## best-first search
## biconditional
## binary constraint
## binary resolution
## binding list
## binocular stereopsis
## biological naturalism
## blocks world
## bluff
## body
## boid
## boosting
## boundary set
## bounded optimality
## bounded PlanSAT
## bounded rationality
## bounds consistent
## bounds propagation
## branching factor
## bridge
## bunch

# C

## calculative rationality
## canonical distribution
## cart-pole
## cascaded finite-state transducers
## case agreement
## causal
## causal link
## causal network
## center
## central limit theorem
## certainty effect
## certainty equivalent
## CFG
## chain rule
## characters
## chart
## checkers
## chess
## Chomsky Normal Form

A grammar is in **Chomsky Normal Form (usually found as CNF)** if all its production rules are in one of the following forms:

```
A -> BC
A -> a
S -> ε
```

where `S` is the starting symbol and `ε` the symbol for the empty string.

## circuit verification
## circumscription
## Clark Normal Form
## classification

Sorting or dividing data into two or more categories on the basis of a distinct feature. 

## clause
## closed-loop
## clustering 

In terms of Data Science, clustering is the grouping of data instances or objects with similar features and characteristics.   

## clutter
## CMAC
## co-NP
## co-NP-complete
## coarse-to-fine
## coarticulation
## coastal navigation
## cognitive psychology
## collusion
## color constancy
## communication
## commutativity
## competitive
## competitive ratio
## complementary literals
## complete assignment
## complete data
## completeness
## completing the square
## completion
## compliant motion
## composition
## compositional semantics
## compositionality
## computable
## computational linguistics
## computational neuroscience
## conclusion
## concurrent action list
## conditional effect
## conditional Gaussian
## conditional probability table
## conditional random field
## conditioning
## confirmation
## conflict set
## conflict-directed backjumping
## conformant
## conjugate gradient
## conjunct ordering
## conjunction
## conjunctive normal form
## connectionist
## consciousness
## consequentialism
## consistency
## consistent
## consistent plan

A plan in which there are no cycles in the *ordering constraints* and no conflicts with the **causal links**.  

## constraint language
## constraint learning
## constraint logic programming
## constraint optimization problem
## constraint propagation
## constraint satisfaction problem
## constraint weighting
## consumable
## context-free grammar
## context-specific independence
## contingency plan
## continuous
## contraction
## contradiction
## control theory
## controller
## convention
## convex set
## convolution
## cooperative
## coordination
## corpus
## Cournot competition
## covariance
## covariance matrix
## critic
## critical path
## critical path  method
## cross-correlation
## crossover point
## cryptarithmetic
## cumulative distribution
## cumulative probability density function
## current-best-hypothesis
## cycle cutset
## cyclic solution
## CYK algorithm

# D
## DARPA Grand Challenge

The DARPA Grand Challenge is a prize competition for American Autonomous Vehicles,funded by the **Defense Advanced Research Projects Agency**,the most prominent research organization of the United States Department of Defense.   

## data association
## data complexity
## data compression
## data matrix
## data mining
## data-driven
## database semantics
## Datalog
## Davis-Putnam algorithm
## decayed MCMC
## decentralized planning
## decision analysis
## decision boundary
## decision maker
## decision network
## decision theory
## decision theory
## decision tree

A decision tree is a construct that uses a tree like graph or model of decisions and their possible consequences,including chance event outcomes,resource costs and utility. 

## declarative
## declarative bias
## decomposition
## deduction theorem
## deductive learning

Going from a known
general rule to a new rule that is logically entailed (and thus nothing new), but is nevertheless useful
because it allows more efficient processing.  

## deep belief networks
## deep learning

It is a subfield of Machine Learning that tries to map the working of the human brain in processing data and creating patterns to use in decision making. 

## default logic
## definite clause
## definite clause grammar
## definition of a rational agent
## deformable template
## degree of belief
## degree of freedom
## delete list
## deliberative layer
## demonic nondeterminism
## Dempster-Shafer theory
## depth
## depth of field
## depth-first search

It is an algorithm which allows us to traverse a graph or a tree data structure; it starts from the root node and traverses as far as possible for each branch before backtracking. (In case of graph data structure, the root node would be any arbitrary node that you select). 

## depth-limited search
## detailed balance
## Deterministic
## diachronic
## diagnostic
## diameter
## Differential GPS
## diffuse albedo
## Diophantine equations
## direct utility estimation
## Dirichlet process
## disambiguation
## discount factor
## discrete
## discretization
## disjoint
## disjunction
## disjunctive constraint
## disparity
## distant point light source
## distortion
## distributed constraint satisfaction
## DL
## domain
## domain closure
## dominant strategy equilibrium
## downward refinement property
## dropping conditions
## DT
## dual graph
## dualism
## duration
## dynamic
## dynamic backtracking
## dynamic Bayesian network
## dynamic programming

It’s a method of solving complex problems by breaking them down to sub-problems that can be solved by back tracking from the last stage. It’s popular real-world problems include traveling salesman problem, Fibonacci sequence, knapsack problem, etc. 

## dynamic state

# E

## early stopping
## economy
## effect
## effective branching factor
## efficient
## electric motor
## eliminative materialism
## elitism
## embodied cognition
## emergent behavior
## empirical gradient
## empirical loss
## empiricism
## English auction
## entailment
## entropy
## environment
## environment generator
## episodic
## epsilon-ball
## equality symbol
## equilibrium
## ergodic
## error rate
## event
## evidence
## evidence reversal
## evolutionary algorithms
## evolutionary psychology
## evolutionary strategies
## exact cell decomposition
## execution
## execution monitoring
## executive layer
## exhaustive decomposition
## existence uncertainty
## Existential Instantiation
## expand
## expectation
## expectation-maximization
## expected value
## expectiminimax value
## explanation-based learning
## explanatory gap
## exploitation
## exploration
## exploration problem
## expressiveness
## extended Kalman filter (EKF)
## extension
## extensive form
## externalities
## extrinsic

# F
## fact
## factor
## factored frontier
## factored representation
## factorial HMM
## factoring
## false negative
## false positive
## feature extraction
## feature selection
## feed-forward network
## FIFO queue
## filtering
## finite horizon
## first-choice hill climbing
## first-order Markov process
## fixate
## fixed point
## fixed-lag smoothing
## flaw
## fluent
## focal plane
## foreshortening
## forward-backward algorithm
## forward-chaining
## frame problem
## frames
## framing effect
## free space
## frequentist
## friendly AI
## frontier
## full joint probability distribution
## fully observable

If an agent's sensors give it access to the complete state of the environment at each point in time,then we say that the task environment is fully observable.

## functionalism
## futility pruning
## fuzzy control
## fuzzy logic
## fuzzy set theory

# G

## G-set
## gain parameter
## gain ratio
## gait
## game theory
## game tree
## Gaussian distribution
## Gaussian error model
## Gaussian filter
## Gaussian process
## generalization
## generalization hierarchy
## generalization loss
## generalized modus ponens
## generating
## generator
## genetic algorithms
## genetic programming
## Gibbs sampling
## GLIE
## global constraint
## global minimum
## Go
## goal
## goal clauses
## goal formulation
## goal monitoring
## goal test
## goal-directed reasoning
## gold standard
## gorilla problem
## gradient
## gradient descent
## grammar
## graph
## graph coloring
## grasping
## greedy agent
## greedy best-first search
## grid world
## ground term
## grounding

# H

## Hamming distance
## Hansard
## haptic feedback
## head
## heavy-tailed distribution
## Hebbian learning
## Hessian
## heuristic function
## heuristic search
## hidden Markov model

A hidden Markov model (or HMM) is a temporal probabilistic model in which the state of the process is described by a *single discrete*
random variable.

## hierarchical lookahead
## hierarchical reinforcement learning
## high-level action
## Hinton diagrams
## holdout cross-validation
## homeostatic
## homophones
## horizon effect
## Horn clause
## hub
## human-level AI
## Hungarian algorithm
## hybrid A*
## hybrid agent
## hybrid architecture
## hybrid Bayesian network
## hydraulic actuation
## hypothesis
## hypothesis prior
## hypothesis space

# I

## i.i.d.
## identification in the limit
## identity matrix
## identity uncertainty
## ignore delete lists
## ignore preconditions heuristic
## image
## imperfect information
## implementation
## implementation level
## implication
## importance sampling
## inclusion-exclusion principle
## incompleteness theorem
## incremental belief-state search
## independence
## independent subproblems
## index
## indexed random variable
## indexing
## individuation
## induction
## inductive learning

Going from a set of specific input-output pairs
to a (possibly incorrect) general rule  is called **inductive learning**.

## inductive logic
## inductive logic programming
## inference
## inference rules
## inferential frame problem
## infinite
## infinite horizon
## infix
## information extraction
## information gain
## information gathering
## information retrieval
## information sets
## informed search
## inheritance
## initial state
## input resolution
## inside-outside algorithm
## insurance premium
## intelligence
## interleaving
## interlingua
## internal state
## interpretation
## interreflections
## intrinsic
## intuition pump
## inverse
## inverse entailment
## inverse kinematics
## inverse reinforcement learning
## inverted pendulum
## inverted spectrum
## IR
## irreversible
## iterative deepening search
## iterative expansion
## iterative-deepening A*

# J

## join tree
## joint action
## joint plan
## JTMS
## justification

# K

## k-d tree
## K-means clustering
## Kalman filtering
## Kalman gain matrix
## kernel
## kernel function
## kernel trick
## kinematic state
## kinematics
## King Midas problem
## knowledge acquisition
## knowledge base
## knowledge engineering
## knowledge-based agents
## Known
## Kriegspiel
## Kullback-Leibler divergence

# L

## label
## Lambert's cosine law
## landmarks
## language
## language generation
## language identification
## large-scale learning
## layers
## leak node
## learning

An agent is **learning** if it improves its
performance after making observations about the world.

## learning curve
## learning element
## learning rate
## least commitment
## least-constraining-value
## leave-one-out cross-validation
## lens
## level cost
## level of abstraction
## level sum
## leveled off
## lexical category
## lexicon
## LIFO queue
## lifting lemma
## likelihood
## likelihood weighting
## line search
## linear Gaussian
## linear interpolation smoothing
## linear programming
## linear regression
## linear resolution
## linear separator
## linkage constraints
## links
## liquid event
## Lisp
## literal
## local consistency
## local search
## locality
## locality-sensitive hash
## localization
## locally structured
## locally weighted regression
## location sensors
## locking
## log likelihood
## logic
## logical equivalence
## logical minimization
## logical omniscience
## logicist
## logistic function
## logistic regression
## long-distance dependencies
## LOOCV
## loopy path
## loosely coupled
## loss function
## lottery
## low-dimensional embedding

# M

## machine reading
## macrops
## magic set
## Mahalanobis distance
## Maintaining Arc Consistency (MAC)
## makespan
## margin
## marginalization
## Markov blanket
## Markov chain
## Markov decision process
## Markov localization
## Markov network
## Markov property
## material value
## materialism
## matrix
## max norm
## max-level
## maximin
## maximin equilibrium
## maximum a posteriori
## maximum expected utility
## maximum-likelihood
## mechanism
## mechanism design
## mel frequency cepstral coefficient (MFCC)
## memoization
## memoized
## mental states
## mereology
## metadata
## metalevel learning
## metalevel state space
## metaphor
## metareasoning
## metonymy
## micromort
## min-conflicts
## mind-body problem
## minimax
## minimax decision
**minimax decision** is the optimal choice which leads MAX to the state with the highest minimax value and leads MIN to lowest minimax value.
## minimax search
## minimax value
The **minimax value** of a node in a game tree is the utility (for MAX) of being in the corresponding state, assuming that both players play optimally from there to the end of the game.
## minimum description length
## minimum slack
## minimum-remaining-values
## Minkowski distance
## missing precondition
## missing state variable
## mixture distribution
## mixture of Gaussians
## mobile manipulator
## modal logic
## model
## model checking
## model selection
## modus ponens
## monitoring
## monotonic preference
## monotonicity
## Monte Carlo
## Monte Carlo localization
## Monte Carlo simulation
## Monte Carlo tree search
## motion blur
## motion model
## multiactor
## multiagent
## multiagent planning problem
## multiagent systems
## multiplexer
## multiplicative utility function
## multiply connected
## multivariate Gaussian
## multivariate linear regression
## mutation
## mutex
## mutual preferential independence
## mutually utility independent
## myopic

# N

## n-armed bandit
## n-gram model
## natural kind
## natural numbers
## nearest-neighbor filter

The nearest-neighbour filter, which repeatedly chooses the closest pairing of predicted position and observation and adds that pairing to the assignment.

## nearest-neighbors regression
## negation
## negative
## neuroscience
## Newton-Raphson
## no-good
## no-regret learning
## noise
## noisy channel model
## noisy-OR
## nondeterministic
## nonholonomic
## nonlinear
## nonlinear regression
## nonmonotonicity
## nonparametric
## nonparametric density estimation
## nonparametric model
## normalization
## normalized form
## normative theory
## NP-complete
## NP-completeness
## null hypothesis

# O

## object model
## objective function
## objectivist
## occupancy grid
## occupied space
## occur check
## Ockham's razor
## odometry
## off-policy
## omniscience
## on-policy
## online replanning
## online search
## ontological commitment
## ontological engineering
## ontology
## open list
## open-code
## open-loop
## operationality
## operations research
## optimal brain damage
## optimal controllers
## optimally efficient
## optimization
## optimizer's curse
## optogenetics
## ordering constraints
## OR-parallelism
## orientation
## origin function
## Othello
## out of vocabulary
## outcome
## overall intensity
## overfitting

# P

## PAC learning
## PageRank
## parameter independence
## parameter learning
## parametric model
## Pareto dominated
## parse tree
## parsing
## partial assignment
## partial information
## partial program
## partially observable
## particle filtering
## partition
## passive learning

A passive learning agent learns from its observations, but the actions the agent takes are not influenced by the learning process.
This is in contrast to an active learning agent, which chooses actions that will facilitate its own learning.

## path
## path planning
## paths
## pattern matching
## payoff function
## PD controller
## PDDL
## Peano axioms
## PEAS
## peeking
## percept

The term **percept** refers to the agent's perceptual inputs at any given instant.

## percept schema
## percept sequence

An agent's **percept sequence** is the complete history of everything the agent has ever perceived.

## perception
## perception layer
## perceptron
## perceptron network
## perfect rationality
## performance element
## perplexity
## persistence arc
## persistent failure model
## perspective projection
## phone model
## phoneme
## phrase structure
## physical symbol system
## physicalism
## piano movers
## pictorial structure model
## PID controller
## plan monitoring
## plan recognition
## planning graph
## PlanSAT
## playout
## ply
## pneumatic actuation
## point-to-point motion
## poker
## policy
## policy evaluation
## policy gradient
## policy improvement
## policy iteration
## policy loss
## policy search
## policy value
## polynomial kernel
## pose
## positive
## possibility axiom
## possibility theory
## possible world
## post-decision disappointment
## pragmatics
## precedence constraints
## precision
## precondition
## prediction
## preference elicitation
## preference independence
## prefix
## premise
## presentation
## principle of indifference
## principle of insufficient reason
## principle of trichromacy
## prioritized sweeping
## priority queue
## prisoner's dilemma
## probabilistic checkmate
## probabilistic Horn abduction
## probabilistic inference
## probability
## probability density function
## probability distribution
## probability model
## probit distribution
## problem
## problem formulation
## problem-solving agent
## procedural attachment
## process
## product rule
## progression planning
## Prolog
## pronunciation model
## proof
## proof-checker
## proposition symbol
## propositionalize
## protein design
## provably beneficial
## pruning
## psychological reasoning
## PUMA
## pure strategy
## pure symbol

# Q

## Q-learning
## QALY
## quadratic programming
## qualia
## qualification problem
## qualitative physics
## quantification
## quantization factor
## quasi-logical form
## question answering
## queue
## quiescence search

# R

## radial basis function
## radiometry
## random surfer model
## random-restart hill climbing
## randomized weighted majority algorithm
## rational agent

A rational agent selects an action that is expected to maximize its performance measure,given the evidence provided by the
*percept sequence* and whatever built-in knowledge the agent has. 

## rationalism
## rationality
## reachable set
## reactive control
## reactive layer
## real-time AI
## realizable
## reasoning
## recall
## reciprocal rank
## recognition
## recombine
## reconstruction
## record linkage
## rectangular grid
## recurrent network
## recursive
## recursive best-first search
## reduct
## reference class
## reference controller
## reference path
## reflect
## reflective architecture
## refutation
## regions
## regression
## regression planning
## regression to the mean
## regret
## regular expression
## regularization
## reinforcement
## reinforcement learning

In **reinforcement learning** the agent learns from a series of
reinforcements-rewards or punishments.

## rejection sampling
## relational extraction
## relational uncertainty
## relative error
## relative likelihood
## relaxed problem
## relevance
## relevance feedback
## relevant
## relevant-states
## renaming
## rendering
## rendering model
## repeated state
## resolution
## resolvent
## result set
## Rete algorithm
## retrograde
## reusable
## revelation principle
## revenue equivalence theorem
## reward
## reward shaping
## reward-to-go
## risk-averse
## risk-neutral
## risk-seeking
## Robocup
## robot navigation
## robotic soccer
## robust control theory
## ROC curve
## rollout
## Roomba
## root mean square
## rules

# S

## S-set
## sample complexity
## sample space
## sampling rate
## SARSA
## SAT
## satisfiability
## satisfiability threshold conjecture
## satisficing
## scaled orthographic projection
## scanning lidars
## scene
## schedule
## schedulers
## schema
## Scrabble
## sealed-bid second-price auction
## search
## search cost
## search tree
## segmentation
## selection
## semantic ambiguity
## semantics
## semi-supervised learning
## semidynamic
## semiotics
## sensitivity analysis
## sensor interface layer
## sensor Markov assumption
## sensorless
## sequence form
## sequential
## sequential Monte Carlo
## set of support
## set semantics
## set-cover problem
## set-level
## shading
## shadow
## shape
## shaving
## shortcuts
## shoulder
## sibyl attack
## sideways move
## sigmoid perceptron
## significance test
## similarity networks
## simulated annealing
## simultaneous localization and mapping (SLAM)
## single agent
## singly connected
## singular
## singularity
## situation
## situation calculus
## skeletonization
## Skolemization
## slack
## slant
## sliding window
## small-scale learning
## smoothing

Smoothing is the process of computing the distribution over past states given evidence up to the present.

## soccer
## social laws
## Socratic reasoner
## soft margin
## softmax function
## software architecture
## sokoban
## solution
## sonar sensors
## sound
## spam detection
## sparse
## sparse model
## spatial reasoning
## specialization
## specular reflection
## specularities
## speech act
## speech recognition
## split point
## stable
## stable model
## standard normal distribution
## standardizing apart
## Starcraft
## start symbol
## state abstraction
## state estimation
## state space
## state-space landscape
## static
## stationarity assumption
## stationary distribution
## stationary process
## stemming
## step cost
## step size
## stereo vision
## stochastic
## stochastic beam search
## stochastic games
## stochastic hill climbing
## stochastic policy
## straight-line distance
## strategic form
## strategy
## strategy profile
## strategy-proof
## strong AI
## structural EM
## structured representation
## stuff
## subcategory
## subgoal independence
## subject-verb agreement
## subjectivist
## subproblem
## substitution
## subsumption
## subsumption architecture
## subsumption lattice
## successor
## successor-state axiom
## Sudoku
## sum of squared differences
## superpixels
## supervised learning

In **supervised learning** the agent observes some example
input-output pairs and learns a function that maps from input to
output.

## support vector machine
## symmetry-breaking constraint
## synchro drive
## synchronic
## synchronization
## syntactic ambiguity
## syntactic theory
## syntax
## synthesis

# T

## table lookup
## tabu search
## tactile sensors
## taxonomy
## Taylor expansion
## technological singularity
## template
## temporal logic
## temporal-difference
## temporal-projection
## term
## terminal states
## terminal test
## test set
## text classification
## texture
## theorem proving
## thrashing
## tiling
## tilt
## time and tense
## time line
## time of flight camera
## time to answer
## tit-for-tat
## topological sort
## total Turing Test
## toy problem
## trace
## tractability
## tragedy of the commons
## trail
## training curve
## training set
## transfer model
## transhumanism
## transition model
## transition probability
## transpose
## transposition table
## traveling salesperson problem
## tree decomposition
## tree width
## treebank
## truth
## truth value
## truth-preserving
## truth-revealing
## turbo decoding
## Turing Test
  The **Turing Test** is a test proposed by Alan Turing in 1950, which is used to determine whether a computer is intelligent by evaluating the "human-ness" of its responses.
## type A strategy
## type B strategy
## type constraint
## type signature

# U

## ultraintelligent machine
## unary constraint
## unbiased
## uncertainty
## underconstrained
## understanding
## unification
## unifier
## uniform-cost search
## Unimate
## uninformed search
## unique action axioms
## unique string axiom
## unit clause
## unit preference
## unit propagation
## unit resolution
## units function
## universal grammar
## Universal Instantiation
## unknown
## unobservable
## unrolling
## unsupervised clustering
## unsupervised learning

In **unsupervised learning**
the agent learns patterns in the input without any explicit feedback. 

## upper confidence bounds on trees
## upper ontology
## Urban Challenge
## utility
## utility independence

# V

## vague
## validation set
## validity
## value
## value alignment
## vanishing point
## variable
## variational approximation
## variational parameters
## VCG
## vector
## vector field histograms
## vehicle interface layer
## verification
## version space
## Vickrey-Clarke-Groves
## virtual counts
## virtual support vector machine
## VLSI layout
## vocabulary
## Voronoi graph

# W

## weak AI
## weak learning
## weight
## weight space
## weighted A* search
## weighted training set
## wide content
## Widrow-Hoff rule
## Winnow algorithm
## workspace representation
## wrapper
## wumpus world

# Z

## zero-sum games
