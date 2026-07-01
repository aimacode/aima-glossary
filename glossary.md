# Glossary for *Artificial Intelligence: A Modern Approach*

[A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [J](#j) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w) | [X](#x) | [Y](#y) | [Z](#z)

## 8-puzzle

**8-puzzle** consists of a 3x3 grid containing 8 numbered tiles and a blank space. A tile adjacent to the blank space can slide into that space. The object is to reach a specified **goal state** from a given **initial state**.

# A

## absolute error

Magnitude of the difference between the theoretical value (expected value) and the actual value of a physical quantity.

## abstraction

Abstraction is selecting data from a larger pool to show only the relevant details to the object.

## abstraction hierarchy

It hides the complexity of the system and allows individuals to work on different modules of the hierarchy at the same time.

## accessibility relations

In modal logic, an accessibility relation R is a binary relation such that R⊆ W×W where W is a set of possible worlds. The accessibility relation determines for each world w ∈ W which worlds ẃ are accessible from w.

## action monitoring

Checking the preconditions of each action as it is executed, rather than checking the preconditions of the entire remaining plan.

## action schema

In classical planning, a lifted representation of a family of related actions using variables. An action schema specifies the action's name and parameters, its **precondition** (a conjunction of literals that must hold for the action to apply), and its **effect** (literals describing how the state changes). Concrete ground actions are obtained by substituting values for the variables.

## action-utility function

A function learned or used by an agent that gives the expected utility (or value) of taking a particular action in a given state, written **Q(s, a)**. Because it maps state-action pairs directly to values, an agent can select the best action without needing a model of the environment's transitions.

## actions

The things that an agent can do. We model this with a function, **Actions(s)**, that returns a collection of actions
that the agent can execute in state *s*.

## activation

In a neural network, the output value produced by a unit (neuron) after applying its activation function to the weighted sum of its inputs. It represents the strength of the signal that the unit passes on to units in the next layer.

## activation function

A mathematical function that transforms the input or set of inputs received at a neuron to produce an output.
Popular examples include the Sigmoid function, Rectificied Linear Units (ReLU) and Hyperbolic Tangent (Tanh)

## active learning

An active learning agent decided which actions to take in order to guide its learning: it values leearning new things
as well as reaping immediate rewards from the environment.
This is in contrast to a passive learning agent, which learns from its observations, but the actions the agent takes are not influenced by the learning process.

## actor

In an actor-critic reinforcement learning architecture, the component that selects actions according to the current policy. The actor updates the policy based on feedback (a signal about how good the chosen actions were) supplied by the critic, which estimates the value function.

## adaptive dynamic programming

Also known as Approximate Dynamic Programming; it is a type of Reinforcement Learning where local rewards and transitions depend on unknown parameters - we set an initial control policy and update it until it converges to an optimal control policy.

## add list

In the STRIPS representation of a planning action's effect, the set of positive (added) literals that become true in the state after the action is applied. It is contrasted with the delete list, which contains the literals that become false.

## admissible heuristic

A **heuristic** is a function that scores alternatives at each branching in a search algorithm. An **admissible heuristic** is one that *never overestimates* the cost to reach the goal. Admissible heuristics are **optimistic** in nature as they believe the cost of reaching the goal is less than it actually is.

## adversarial search

Traversing a tree data structure to find all possible moves. It is usually used in a two-player game; each available move is represented using gain and loss for an individual player. An important application of it is in zero sum games, as in those games, one players' loss is the other players' gain.

## adversary argument

A proof technique for establishing lower bounds on the difficulty of a problem by imagining an adversary that answers the algorithm's queries in the worst possible (but consistent) way. It is used, for example, to show that an online search agent can be forced by the environment into arbitrarily costly behavior.

## agent

An **agent** is anything that can be viewed as perceiving its **environment** through **sensors** and acting upon that environment through **actuators**.

## agent function

An agent's behavior is described by the **agent function** that maps any given percept sequence to an action.

## agent program

_Internally,_ the agent function for an artificial agent will be implemented by an **agent program**.

## agglomerative clustering

It is a category of hierarchical clustering which uses a bottom-up approach. All observations start in their own cluster and different pairs of clusters are merged as you move up levels in the hierarchy. Its results are represented using a dendrogram.

## aggregation

The process of combining multiple individual elements into a single summary or group. In probabilistic and decision-theoretic settings this often refers to combining preferences, votes, or probability estimates from several sources into one collective result.

## algorithm

An **algorithm** is a sequence of **unambiguous finite steps** that when carried out on a given problem produce the expected outcome and terminate in **finite time**.

## alignment method

An approach for ensuring that the objectives an AI system actually pursues match the objectives its designers intend. Alignment methods aim to prevent the value alignment problem, in which an agent optimizes a misspecified objective and produces harmful or unintended behavior.

## alpha-beta

**alpha** (**&alpha;**) is the value of the best (i.e., highest-value) choice we have found so far at any choice point along the path for MAX and **beta**(**&beta;**) is the value of the best (i.e., lowest-value) choice we have found so far at any choice point along the path for MIN in a standard minimax tree.

## alpha-beta pruning

**alpha—beta pruning** is applied to a standard minimax tree to prune away branches that cannot possibly influence the final minimax decision.

## ambient illumination

Light that is already present in a scene, before any additional lighting is added. It usually refers to natural light.

## ambiguity

It is the state of being uncertain or doubtful.

## ambiguity aversion

Ambiguity aversion (also known as uncertainty aversion) is a preference for known risks over unknown risks.

## analogical reasoning

Analogical reasoning is any type of thinking that relies upon an analogy. An analogical argument is an explicit representation of a form of analogical reasoning that cites accepted similarities between two systems to support the conclusion that some further similarity exists.

## anchoring effect

It is a type of cognitive bias which makes people focus on the first piece of information (the "anchor") that was given to them, to make decisions. To explain this with an example; when buying a product if you're told a high price by the seller, your mind estimates the worth of that product as that initial/anchor price you're told, and then when you're offered a discount on it, you are more inclined to buy it thinking that you're getting it for cheap.

## And-Elimination

In propositional logic, conjunction elimination (also called and elimination, ∧ elimination, or simplification0 is a valid immediate inference, argument form and rule of inference which makes the inference that, if the conjunction A and B is true, then A is true, and B is true. The rule makes it possible to shorten longer proofs by deriving one of the conjuncts of a conjunction on a line by itself.

## AND-parallelism

A form of parallel execution in logic programming in which the several conjuncts (subgoals) of a single goal are solved simultaneously. It contrasts with OR-parallelism, where alternative clauses that could satisfy a goal are explored in parallel.

## angelic nondeterminism

A notional ability always to choose the most favorable option, in constant time. With angelic non-determinism, any problem in NP would be solvable in polynomial time.

## angelic semantics

In hierarchical planning, an interpretation of a high-level action in which the agent (the "angel") gets to choose which of the action's possible outcomes will occur. The reachable set of a plan under angelic semantics is the set of states the agent can guarantee reaching, which allows an agent to commit to a high-level plan knowing it can later refine it into a successful concrete plan.

## answer literal

An extra literal added to a query when using resolution theorem proving to extract answers. By attaching a special predicate containing the variables of interest to the negated goal, the bindings that satisfy the query can be read off when the empty (or answer) clause is derived.

## answer set programming

Answer set programming (ASP) is a form of declarative programming oriented towards difficult (primarily NP-hard) search problems. It is based on the stable model (answer set) semantics of logic programming.

## answer sets

The stable models that constitute the solutions to a program in answer set programming. Each answer set is a consistent set of ground literals that satisfies the program's rules under the stable-model semantics, capturing one acceptable way the world could be given the program's constraints.

## aortic coarctation

It is a Medical term which means the narrowing of aorta - the largest artery in the body which starts from the heart.

## apprenticeship learning

It is the process of learning by observing the demonstration of an expert. In a way, it is a form of supervised learning where the training data would be the tasks performed by the expert.

## architecture

It is a very broad term; however, it is generally used to refer to the structure of the buildings and other constructions.

## arity

In logic, mathematics, and computer science, the arity of a function or operation is the number of arguments or operands that the function takes.

## artificial life

Artificial life (often abbreviated ALife or A-Life) is a field of study wherein researchers examine systems related to natural life, its processes, and its evolution, through the use of simulations with computer models, robotics, and biochemistry.

## ascending-bid

Bidders place bids of progressively higher amounts, aiming to outbid each other. The bidder who places the highest bid by the end of the auction wins.

## Asilomar Principles
The Asilomar Conference on Beneficial AI was a conference organized by the Future of Life Institute, held January 5-8, 2017, at the Asilomar Conference Grounds in California. More than 100 thought leaders and researches in economics, law, ethics, and philosophy met at the conference, to address and formulate principles of beneficial AI. Its outcome was the creation of a set of guidelines for AI research – the 23 Asilomar AI Principles.

## assignment

In a constraint satisfaction problem, a choice of values for some or all of the variables. A complete assignment gives a value to every variable, a partial assignment gives values to only some, and a consistent (or legal) assignment is one that violates no constraints; a solution is a complete, consistent assignment.

## associative memory

In terms of Psychology, it is the type of memory which allows us to remember things by finding links between, apparently, unrelated things. To explain with an example; remembering someones' name by the dress they wore the first time you met them. Clearly, these two things seem completely unrelated.

## asymptotic analysis

It is a Mathematical method of describing limiting behavior by using an input bound function, which means that the algorithm would run in constant time if no output was given, as the rest of the factors contributing to the computation are constant.

## asymptotic bounded optimality

A relaxed notion of bounded optimality for agent programs: a program is asymptotically bounded optimal if, given a machine that is a constant factor faster (or larger), it performs at least as well as any other program on the original machine. This makes the ideal of doing the best possible with limited computational resources achievable in practice.

## ATMS

Assumption-Based Truth Maintenance System (ATMS) allows to maintain and reason with a number of simultaneous, possibly incompatible, current sets of assumption.

## atom

An atom is the smallest constituent unit of ordinary matter that has the properties of a chemical element.

## atomic representation

A way of representing a state as an indivisible whole with no internal structure. Each state is simply a distinct entity, and the only relation the algorithm can test is whether two states are identical. Search and game-playing algorithms often work over atomic representations. It contrasts with factored and structured representations.

## atomic sentence

In logic, the simplest form of sentence, containing no logical connectives. In propositional logic it is a single proposition symbol; in first-order logic it is a predicate applied to terms (possibly none), which asserts a fact and is either true or false.

## attribute-based extraction

An information-extraction task in which the goal is to pull out the values of specific attributes from text, such as reading the price, size, or manufacturer of a product from a description. It typically treats the text as describing a single object with a set of attribute-value pairs.

## augmented grammar

Any grammar whose productions are augmented with conditions expressed using features.

## authority

In web search and information retrieval, a measure of how trustworthy or definitive a page is as a source on a topic. In the HITS algorithm, an authority page is one that is linked to by many good hub pages, and its authority score is computed jointly with the hub scores of the pages pointing to it.

## automatic assembly sequencing

A robotics planning task of determining a feasible order in which the parts of a product can be put together. It requires reasoning about geometry and reachability so that each part can be added without collisions and without disturbing parts already placed.

## autonomy

It is the character of being independent and self-governing in vital as well as non-vital situaltions.

## average reward

In reinforcement learning and Markov decision processes, an optimality criterion that evaluates a policy by the long-run average reward received per time step rather than by a discounted sum. It is well suited to continuing (non-terminating) tasks where the agent operates indefinitely.

## axiom

In Mathematics or logics, an axiom is a statement or a proposition which is assumed to be true to serve as a starting point for further arguments and reasoning. Example of an axiom: “Nothing can both be and not be at the same time and in the same respect”.

# B

## back-propagation

**back-propagation** is an algorithm used for *supervised learning* of **artificial neural networks** using gradient descent.
The method calculates the gradient of a given error function with respect to the weights of the network. The "backward" terminology stems because the gradient calculation requires backward propagation through the newtork.

## backed-up value

In game-tree and search algorithms, the backed-up value of a node is the value propagated to it from its descendants. For example, in minimax the value of a node is backed up from the best of its children according to whose turn it is to move.

## backgammon

Backgammon is one of the oldest known board games. It is a two player game where each player has fifteen pieces (checkers) which move between twenty-four triangles (points) according to the roll of two dice. The objective of the game is to be first to bear off, i.e. move all fifteen checkers off the board.

## background subtraction

Foreground detection is one of the major tasks in the field of computer vision and image processing whose aim is to detect changes in image sequences. Background subtraction is any technique which allows an image's foreground to be extracted for further processing (object recognition etc.).

## backjumping

An improvement to backtracking search for constraint satisfaction problems in which, upon reaching a dead end, the search backs up not to the immediately preceding variable but to the most recent variable in the conflict set—the set of previously assigned variables responsible for the failure. This avoids revisiting assignments that could not resolve the conflict.

## backmarking

A constraint satisfaction technique that remembers, for each variable, which values have already been tested against which earlier assignments, so that redundant consistency checks can be skipped during backtracking search.

## backoff model

A language model smoothing technique in which, when there is insufficient data to estimate the probability of an n-gram, the model "backs off" to a shorter (n-1)-gram estimate, combining evidence across n-gram orders to handle unseen sequences.

## backpropagation

To minimize the cost function, we need to know how the changes in weights and biases affect the cost function i.e. partial derivatives of the cost function w.r.t every weight and bias in the network; back propagation is a method that allows us to quickly compute all these partial derivatives.

## Backus-Naur form (BNF)

A mathematical notation used to describe the syntax of a programming language.  

## backward-chaining

A goal-directed inference method that starts from a query (goal) and works backward, seeking rules whose conclusions match the goal and then recursively trying to prove the premises of those rules. It is commonly used in logic programming systems such as Prolog.

## bag of words

The **bag of words** model is a simplifying representation used in natural language processing and information retrieval. Also known as the vector space model. In this model, a text is represented as the bag of its words, disregarding grammar and even word order but keeping multiplicity.

## bagging

Bagging (stands for **B**ootstrap **Agg**regat**ing**) is a way to decrease the variance of your prediction by generating additional data for training from your original dataset using combinations with repetitions to produce multisets of the same cardinality/size as your original data. By increasing the size of your training set you can't improve the model predictive force, but just decrease the variance, narrowly tuning the prediction to expected outcome.

## bang-bang control

A control scheme in which the control signal switches abruptly between extreme values (such as fully on and fully off) rather than taking intermediate values, often arising as the optimal policy for certain time-optimal control problems.

## baseline

In stereo vision, the distance between the two camera positions (or the two eyes) from which images are taken; a larger baseline improves depth estimation. More generally, a baseline is a reference method or level of performance against which a new system is compared.

## batch gradient descent

A form of gradient descent in which the gradient of the loss function is computed over the entire training set before each update to the parameters, in contrast to stochastic or minibatch gradient descent, which use one example or a small subset per update.

## Bayes' rule

**Bayes' rule** describes the probabilty of an event(lets say A) in the light of that a given event B has already occured.
Mathematically Bayes' rule can be described as :-  **P(A|B) = P(A)P(B|A)/P(B)**

## Bayes-Nash equilibrium

A solution concept for games of incomplete information in which each player, given their private information (type) and beliefs about the types of others, chooses a strategy that maximizes their expected payoff, and no player can do better by deviating. It generalizes the Nash equilibrium to Bayesian games.

## Bayesian learning

A Machine Learning method which enables us to encode our initial perception of what a model should look like, regardless of what the data tells us. It proves to be very useful when there’s a sparse amount of data to train our model properly.

## Bayesian network

A probabilistic graphical model representing a group of variables along with their conditional dependencies through a direct acyclic graph; it is also used to compute the probability distribution for a subset of network variables, provided the distributions or values of any subset of the remaining variables.

## beam search

Beam search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. Beam search is an optimization of best-first search that reduces its memory requirements. Best-first search is a graph search which orders all partial solutions (states) according to some heuristic. But in beam search, only a predetermined number of best partial solutions are kept as candidates.[1] It is thus a greedy algorithm.

## behaviorism

A movement in psychology, associated with figures such as Watson and Skinner, that held that scientific study of the mind should focus only on observable stimuli and responses (behavior) while rejecting appeals to internal mental states.

## belief function

In Dempster-Shafer theory, a function that assigns to each proposition a lower bound on the probability of that proposition, representing the total belief committed to it while allowing for explicit ignorance about how remaining belief mass is distributed.

## belief propagation

A message-passing algorithm for performing inference in graphical models, in which nodes exchange messages summarizing local evidence to compute marginal distributions. It is exact on trees (as in the polytree algorithm) and used as an approximate method (loopy belief propagation) on general graphs.

## belief revision

The process of updating an agent's set of beliefs when new information arrives that may contradict what was previously held, changing the agent's model of the world as opposed to merely updating a probability distribution.

## belief state

A representation of an agent's knowledge about the current situation, given as the set of possible physical states the agent might be in (in nondeterministic or partially observable settings) or, in probabilistic settings, a probability distribution over those states.

## Bellman equation

A recursive equation, central to dynamic programming and reinforcement learning, that expresses the utility (value) of a state in terms of the immediate reward plus the discounted expected utility of successor states, thereby characterizing an optimal policy.

## Bellman update

The iterative computation that applies the Bellman equation to improve estimates of state utilities; repeatedly applying this update, as in value iteration, converges to the true utilities under a Markov decision process.

## benchmarking

Benchmarking is to measure the quality of something for the purposes of comparison or evaluation.

## best-first search

A general search strategy in which the node to expand next is chosen as the one with the lowest value of an evaluation function f(n). Different choices of f yield different algorithms, such as greedy best-first search and A* search.

## biconditional

A logical connective, written as an equivalence (P ⇔ Q), that is true exactly when both operands have the same truth value; it is logically equivalent to the conjunction of "P implies Q" and "Q implies P."

## binary constraint

In a constraint satisfaction problem, a constraint that relates exactly two variables, restricting the pairs of values they may jointly take.

## binary resolution

An inference rule for propositional and first-order logic that takes two clauses containing complementary literals and derives a new clause (the resolvent) consisting of the remaining literals; repeated application forms the basis of resolution theorem proving.

## binding list

The set of variable-to-term substitutions produced by unification, recording which value each logical variable is bound to when two expressions are matched.

## binocular stereopsis

The perception of depth obtained by combining the two slightly different images captured from two eyes or cameras; the disparity between corresponding points in the images is used to infer distance.

## biological naturalism

A position, associated with John Searle, holding that mental states are high-level features of the brain arising from low-level neurobiological processes, and that consciousness cannot be reduced to mere computation or functional organization.

## blocks world

A classic, simplified planning domain consisting of a set of cube-shaped blocks that can be stacked on one another or placed on a table, used to illustrate planning and reasoning about actions.

## bluff

In game theory and games such as poker, an action taken to deceive opponents about one's true situation—for example, betting strongly with a weak hand to induce opponents to fold.

## body

In a definite clause or rule of the form "head ← body," the body is the conjunction of premises (the antecedent) that must hold for the head (conclusion) to be inferred.

## boid

Boids is an artificial life program, developed by Craig Reynolds in 1986, which simulates the flocking behaviour of birds. His paper on this topic was published in 1987 in the proceedings of the ACM SIGGRAPH conference. The name "boid" corresponds to a shortened version of "bird-oid object", which refers to a bird-like object.

## boosting

Boosting is a two-step approach, where one first uses subsets of the original data to produce a series of averagely performing models and then "boosts" their performance by combining them together using a particular cost function (=majority vote). Unlike bagging, in the classical boosting the subset creation is not random and depends upon the performance of the previous models: every new subsets contains the elements that were (likely to be) misclassified by previous models.

## boundary set

In version-space learning, the compact representation of all hypotheses consistent with the training examples, given by the most general boundary (G-set) and the most specific boundary (S-set); every consistent hypothesis lies between these two sets.

## bounded optimality

A criterion for rational agents proposing that an agent should behave as well as possible given the computational resources (such as time and memory) available to it, rather than being perfectly rational in an unconstrained sense.

## bounded PlanSAT

A decision version of the planning problem that asks whether a plan of length at most k exists to achieve the goal; this bounded variant is NP-complete, whereas the unbounded PlanSAT is PSPACE-complete.

## bounded rationality

The idea, introduced by Herbert Simon, that real agents have limited computational resources and therefore cannot always compute the fully optimal decision; instead they use heuristics and often "satisfice," settling for a good-enough choice.

## bounds consistent

A relaxed form of consistency for constraint problems with numeric variables in which, for each variable, its lower and upper bounds are consistent with the constraints, even if some intermediate values in the interval are not.

## bounds propagation

A constraint-propagation technique for variables with ordered numeric domains that repeatedly tightens the upper and lower bounds of each variable's domain to enforce consistency with the constraints.

## branching factor

The number of successors (children) a node has in a search tree or graph; the average branching factor is a key determinant of the size of the search space and the cost of search.

## bridge

A card game for four players in two partnerships, notable in AI as an example of a partially observable, multiplayer game requiring bidding (communication) and card play under uncertainty.

## bunch

In the situation-calculus / knowledge representation of substances and objects, a bunch is a composite object formed by grouping together a collection of parts, treated as a single entity.

# C

## calculative rationality

The property of an agent that eventually computes the rational action to take, but perhaps too slowly to be useful. In practice, decisions must often be made under time pressure, so pure calculative rationality is not always a sufficient goal.

## canonical distribution

A conditional probability distribution that fits a fixed, parameterized pattern rather than requiring a full table of independent values. The noisy-OR distribution is a common example, allowing a node with many parents to be described with few parameters.

## cart-pole

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force to the cart in the left or right direction. The pendulum starts upright, and the goal is to prevent it from falling over. A reward is provided for every timestep that the pole remains upright.

## cascaded finite-state transducers

A sequence of finite-state transducers applied in series for information extraction, where each stage handles one level of processing (such as tokenizing, recognizing groups, then combining groups) and passes its output to the next stage.

## case agreement

The grammatical requirement that a word take a form matching its syntactic role, such as the difference between subjective case (e.g., "I") and objective case (e.g., "me") for pronouns in English.

## causal

Relating to the direction from causes to effects. In a Bayesian network, causal links point from cause to effect, which typically yields sparser and more natural models than the reverse (diagnostic) direction.

## causal link

In partial-order planning, a structure recording that one action achieves a precondition of another. Written as a protected connection between a producer step and a consumer step for a specific literal, it flags any step that might delete that literal as a threat to be resolved.

## causal network

A model expressing cause-and-effect relationships among variables, in which the directed edges represent direct causal influence. Such networks support reasoning about the effects of interventions, not just observations.

## center

In the context of clustering, the representative point (such as the mean) of a cluster; k-means, for example, iteratively assigns points to the nearest center and recomputes centers.

## central limit theorem

The central limit theorem (CLT) establishes that, in some situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution (informally a "bell curve") even if the original variables themselves are not normally distributed.

## certainty effect

A pattern of human decision making in which people place disproportionately high value on outcomes that are certain compared to those that are merely very probable, contributing to violations of expected utility theory.

## certainty equivalent

The guaranteed payoff that a decision maker regards as equally desirable to a given uncertain lottery. For risk-averse agents this is less than the lottery's expected monetary value.

## CFG

See context-free grammar. A grammar whose production rules each rewrite a single nonterminal symbol independently of context.

## chain rule

It is a mathematical formula used to compute the derivatives of a composition of two or more functions.

## characters

A character is any letter, number, space, punctuation mark, or a symbol.

## chart

A data structure used by chart parsers to store partial and completed constituents found while parsing a string, avoiding recomputation by recording which substrings can be derived as which categories.

## checkers

A two-player board game (also called draughts) played on an 8x8 board with pieces that move and capture diagonally. It was an early landmark for game-playing programs; Samuel's checkers program was a pioneering machine-learning system, and the game was later proven to be a draw with perfect play.

## chess

Chess is a two-player strategy board game played on a chessboard, a checkered gameboard with 64 squares arranged in an 8×8 grid. Play does not involve hidden information. Each player begins with 16 pieces: one king, one queen, two rooks, two knights, two bishops, and eight pawns. Each of the six piece types moves differently, with the most powerful being the queen and the least powerful the pawn. The objective is to checkmate the opponent's king by placing it under an inescapable threat of capture. To this end, a player's pieces are used to attack and capture the opponent's pieces, while supporting each other.

## Chomsky Normal Form

A grammar is in **Chomsky Normal Form (usually found as CNF)** if all its production rules are in one of the following forms:

```
A -> BC
A -> a
S -> ε
```

where `S` is the starting symbol and `ε` the symbol for the empty string.

## circuit verification

The task of proving that a digital circuit meets its specification, which can be cast as logical inference: encode the circuit's structure and gate behaviors as sentences and check that the desired input/output properties follow.

## circumscription

A nonmonotonic logic that formalizes default reasoning by assuming that the objects known to have a certain property are the only ones that do (minimizing the extension of that property). It provides a way to express assumptions like "abnormal cases are as few as possible."

## Clark Normal Form

A standardized form for logic-program definitions in which each predicate is defined by a single completed equivalence, used as a preliminary step in constructing Clark's completion for negation-as-failure semantics.

## classification

Sorting or dividing data into two or more categories on the basis of a distinct feature.

## clause

A disjunction of literals. Clauses are the building blocks of conjunctive normal form and are the units on which the resolution inference rule operates.

## closed-loop

Describing control or execution that uses feedback from perception of the environment to adjust behavior, in contrast to open-loop execution that carries out a fixed plan regardless of what is observed.

## clustering

In terms of Data Science, clustering is the grouping of data instances or objects with similar features and characteristics.   

## clutter

Extraneous objects or measurements in a sensing environment that do not correspond to the targets of interest, complicating tasks such as data association in tracking.

## CMAC

The Cerebellar Model Articulation Controller, a form of function approximator that combines multiple coarse, overlapping tilings of the input space to represent a value function, allowing efficient local generalization.

## co-NP

The complexity class consisting of decision problems whose complement is in NP; that is, problems for which a "no" answer has a short, efficiently checkable certificate. It is widely believed but not proven that NP and co-NP differ.

## co-NP-complete

The hardest problems in co-NP, to which every co-NP problem can be reduced in polynomial time. Deciding whether a propositional formula is valid (a tautology) is a canonical example.

## coarse-to-fine

A strategy that first solves a problem at a coarse level of resolution or abstraction and then progressively refines the solution, used in areas such as image processing and search to save computation.

## coarticulation

In speech, the phenomenon whereby the acoustic realization of a phoneme is influenced by neighboring sounds, so that the same phoneme is pronounced differently depending on its context.

## coastal navigation

A robot navigation heuristic in which the robot stays near features (such as walls) that provide reliable localization information, reducing positional uncertainty even at the cost of a longer path.

## cognitive psychology

It is a branch of Psychology which deals with the mental processes involved in obtaining and comprehending new information. Some of the prominent processes include judging, problem solving and remembering.

## collusion

An agreement among agents (for example, bidders in an auction) to coordinate their strategies for mutual benefit at the expense of others, such as keeping prices artificially low or high. Auction and mechanism design often aim to discourage collusion.

## color constancy

The perceptual ability to judge the intrinsic color of a surface as roughly constant despite changes in the color and intensity of the illuminating light.

## communication

The intentional exchange of information among agents through the production and perception of signs, such as words in a language, typically for the purpose of influencing one another's beliefs or actions.

## commutativity

A property of a search or planning problem in which the order of applying a set of independent actions does not affect the outcome, allowing the search to consider the actions in any single fixed order without loss.

## competitive

Describing a multiagent environment in which agents have opposing goals, so that one agent's gain tends to be another's loss, as in a zero-sum game.

## competitive ratio

For an online algorithm operating without full knowledge of the environment, the worst-case ratio between the cost it incurs and the cost of the best solution achievable with full knowledge; a bounded ratio indicates good performance despite uncertainty.

## complementary literals

A pair of literals in which one is the negation of the other. Resolution combines two clauses containing complementary literals by canceling that pair.

## complete assignment

In a constraint satisfaction problem, an assignment that gives a value to every variable, as opposed to a partial assignment. A complete assignment is a solution when it also satisfies all constraints.

## complete data

In learning, a data set in which the value of every relevant variable is observed for every example, so that no values are missing or hidden. Complete data simplifies parameter estimation.

## completeness

A property of a search algorithm or inference procedure guaranteeing that it will find a solution (or derive any entailed sentence) whenever one exists.

## completing the square

An algebraic technique for rewriting a quadratic expression in a form that reveals its extremum, used for instance in manipulating Gaussian distributions during probabilistic inference.

## completion

For a logic program, a logical theory (Clark's completion) that turns the program's "if" rules into "if and only if" definitions, giving a semantics that supports negation as failure.

## compliant motion

Robot motion carried out while maintaining contact with a surface or object and yielding to constraints imposed by that contact, such as sliding a peg along a hole to insert it.

## composition

The combination of functions or relations in which the output of one becomes the input of the next; more generally, building complex structures or meanings from parts.

## compositional semantics

The principle and practice of computing the meaning of a phrase as a function of the meanings of its parts and the way they are combined syntactically.

## compositionality

The property of a representation whereby the meaning of a whole is determined systematically by the meanings of its components and their arrangement.

## computable

Describing a function that can be computed by some algorithm (equivalently, by a Turing machine); functions that no algorithm can compute are noncomputable.

## computational linguistics

The subfield concerned with computational methods for analyzing, understanding, and generating human language, overlapping heavily with natural language processing.

## computational neuroscience

The study of how information is processed in nervous systems using computational and mathematical models of neurons and neural circuits.

## conclusion

In an implication of the form "if premise then conclusion," the consequent part that is asserted to hold when the premise holds; in logical rules, the head or right-hand side.

## concurrent action list

In planning with concurrent actions, a representation that specifies which actions occur simultaneously at a given time step, together with constraints on which actions may or may not co-occur.

## conditional effect

An effect of a planning action that occurs only when an associated condition holds in the state where the action is applied, allowing a single action schema to have context-dependent results.

## conditional Gaussian

A distribution in a hybrid Bayesian network for a continuous variable whose value is Gaussian with mean and variance that depend on the values of continuous parents and possibly on discrete parents.

## conditional probability table

A table associated with a node in a Bayesian network that specifies the probability distribution over that node's values for each combination of values of its parent nodes.

## conditional random field

An undirected discriminative model that defines the conditional distribution of a set of output labels given observed inputs, commonly used for sequence-labeling tasks such as part-of-speech tagging.

## conditioning

The operation of updating or computing a probability distribution given the observed values of some variables, as in computing a conditional probability from a joint distribution.

## confirmation

The support that evidence lends to a hypothesis; in probabilistic terms, evidence confirms a hypothesis when it raises the hypothesis's probability.

## conflict set

In a constraint satisfaction search, the set of previously assigned variables that are responsible for a dead-end at the current variable, used to guide intelligent backtracking.

## conflict-directed backjumping

A backtracking method for constraint satisfaction that, upon reaching a dead end, jumps back to the most recent variable in the conflict set and propagates conflict information, avoiding unproductive reassignments.

## conformant

Describing planning (also called sensorless planning) that must achieve a goal without any observations, so the plan must work for every possible initial state and outcome in a belief state.

## conjugate gradient

An iterative optimization method that finds successive search directions that are conjugate with respect to the problem's curvature, converging faster than plain gradient descent on quadratic-like objectives.

## conjunct ordering

In constraint satisfaction or query answering, choosing the order in which to evaluate the conjoined conditions (conjuncts) so as to reduce the amount of search or the size of intermediate results.

## conjunction

A logical sentence formed by connecting sentences with "and" (∧), which is true exactly when all of its conjuncts are true.

## conjunctive normal form

A representation of a logical sentence as a conjunction of clauses, where each clause is a disjunction of literals. Every propositional sentence can be converted to an equivalent CNF sentence, which is required for resolution.

## connectionist

Describing an approach to AI that models intelligence using networks of simple interconnected units (as in neural networks), where knowledge is represented by the pattern of connection strengths rather than explicit symbolic rules.

## consciousness

The subjective, first-person experience of awareness and sensation; a phenomenon whose relationship to physical computation remains a deeply contested question in the philosophy of AI.

## consequentialism

The ethical view that the rightness of an action is determined by its consequences; utility-based agents embody a form of consequentialism by choosing actions according to their expected outcomes.

## consistency

A property indicating the absence of contradiction: a set of logical sentences is consistent (satisfiable) if some model makes them all true; a heuristic is consistent if it obeys a form of the triangle inequality.

## consistent

Free of contradiction, or satisfying all applicable constraints or requirements. A consistent heuristic never overestimates the step-plus-remaining cost so as to violate monotonicity.

## consistent plan

A plan in which there are no cycles in the *ordering constraints* and no conflicts with the **causal links**.  

## constraint language

The formal notation used to express the allowed combinations of values in a constraint satisfaction problem, ranging from explicit enumeration of legal tuples to compact operators such as "all different" or arithmetic relations.

## constraint learning

A technique in constraint satisfaction that, upon encountering a dead end, records the discovered conflict as a new constraint (a no-good) so that the same fruitless combination is never explored again elsewhere in the search.

## constraint logic programming

An extension of logic programming in which variables range over domains (such as integers or reals) and goals may include constraints, which are handled by a specialized constraint solver rather than by plain unification.

## constraint optimization problem

A constraint satisfaction problem augmented with an objective function over the variables, where the goal is to find an assignment that satisfies the constraints while optimizing that objective.

## constraint propagation

The process of using the constraints to reduce the set of legal values for variables, propagating these reductions to neighboring variables in order to detect inconsistencies early and shrink the search space.

## constraint satisfaction problem

A problem defined by a set of variables, a domain of possible values for each, and a set of constraints specifying allowable value combinations; a solution is an assignment to all variables that violates no constraint.

## constraint weighting

A local-search technique for constraint satisfaction that attaches weights to constraints and increases the weight of those frequently violated, steering the search toward satisfying the most troublesome constraints.

## consumable

In planning with resources, a resource that is used up by actions and thus decreases over time (such as fuel or money), as opposed to a reusable resource that is freed again after use.

## context-free grammar

A grammar in which each production rule rewrites a single nonterminal symbol into a string of terminals and nonterminals, independently of the surrounding context, generating the class of context-free languages.

## context-specific independence

A form of conditional independence that holds only for particular values of some variables (a specific context), allowing more compact representation of conditional distributions than the full parent set would suggest.

## contingency plan

A plan that includes conditional branches, prescribing different actions depending on the outcomes of sensing actions, so that it can respond to different situations that may arise during execution.

## continuous

Describing a variable, state space, or environment that can take values from an uncountable range (such as the real numbers) rather than a finite or countable set of discrete values.

## contraction

A mapping that brings any two points closer together by at least a constant factor; value iteration relies on the Bellman update being a contraction, guaranteeing convergence to a unique fixed point.

## contradiction

A logical sentence that is false in every model (unsatisfiable); deriving a contradiction (the empty clause) is the goal of a proof by refutation.

## control theory

The engineering and mathematical discipline concerned with designing systems that regulate the behavior of dynamical processes using feedback, providing tools closely related to those used for continuous-time agent control.

## controller

A component that maps observed states or sensor readings to actions in order to keep a system behaving as desired, forming the core of a control or reinforcement-learning agent.

## convention

A shared, arbitrary regularity of behavior adopted by the members of a multiagent group (such as which side of the road to drive on) that coordinates their actions without central enforcement.

## convex set

A set of points such that the straight line segment between any two of its members lies entirely within the set; convexity of a feasible region or objective is important for guaranteeing efficient optimization.

## convolution

A mathematical term which basically means merging two signals to form a third signal.

## cooperative

Describing a multiagent environment in which agents share goals or can benefit from working together, in contrast to a competitive setting where their interests conflict.

## coordination

The problem, in multiagent systems, of getting agents to act in a mutually compatible way to achieve joint or individual goals, often addressed through conventions, communication, or joint planning.

## corpus

A large, structured collection of natural-language text (or speech) used as data for training and evaluating language-processing systems and for gathering statistics about language.

## Cournot competition

A model of oligopoly in which competing firms independently and simultaneously choose the quantities they will produce, and the market price is set by total output; equilibrium is analyzed as a game.

## covariance

A measure of the degree to which two random variables vary together; it is positive when they tend to increase together, negative when one tends to rise as the other falls, and zero when they are uncorrelated.

## covariance matrix

A matrix whose entries are the covariances between each pair of components of a random vector, with variances on the diagonal; it characterizes the spread and orientation of a multivariate distribution such as a Gaussian.

## critic

The component of a learning agent that evaluates the agent's performance against a fixed standard and provides feedback (such as a reward signal) used by the learning element to improve behavior.

## critical path

In a schedule with ordering constraints, the longest path of dependent actions from start to finish, which determines the minimum total duration; delaying any action on this path delays the whole plan.

## critical path  method

A scheduling technique that computes the earliest and latest possible start times for each action given precedence constraints, identifying the critical path and the slack available for other actions.

## cross-correlation

An operation that measures the similarity between a signal and a shifted template as a function of the shift, closely related to convolution and used in signal and image processing.

## crossover point

In a genetic algorithm, the position within two parent strings at which they are split and recombined to form offspring, mixing genetic material from both parents.

## cryptarithmetic

A type of puzzle, often used to illustrate constraint satisfaction, in which the digits of an arithmetic problem are replaced by letters and the task is to find the digit assignment that makes the arithmetic correct.

## cumulative distribution

A function giving the probability that a random variable takes a value less than or equal to a given point, obtained by accumulating the probability mass or density up to that point.

## cumulative probability density function

The function that gives, for a continuous random variable, the probability that the variable falls at or below a given value, computed as the integral of the probability density up to that value.

## current-best-hypothesis

An incremental learning approach that maintains a single hypothesis consistent with the examples seen so far and revises it (by generalizing or specializing) whenever a new example is misclassified.

## cycle cutset

A subset of the variables in a constraint graph whose removal leaves the remaining graph acyclic (a tree); assigning values to this set lets the rest of the problem be solved efficiently by tree-structured methods.

## cyclic solution

In nondeterministic or contingency planning, a solution plan containing loops, so that an action may be retried until it produces a desired outcome, appropriate when an action can fail and be attempted again.

## CYK algorithm

A dynamic-programming parsing algorithm that determines whether and how a string can be derived by a context-free grammar in Chomsky Normal Form, filling a table of possible constituents over substrings in cubic time in the length of the input.

# D
## DARPA Grand Challenge

The DARPA Grand Challenge is a prize competition for American Autonomous Vehicles,funded by the **Defense Advanced Research Projects Agency**,the most prominent research organization of the United States Department of Defense.   

## data association

The problem of deciding, given a set of observations, which observation corresponds to which underlying object or track. It arises in multiple-object tracking, where measurements at each time step must be assigned to the objects that generated them.

## data complexity

A measure of how the running time of an inference procedure grows as a function of the number of ground facts in the knowledge base, holding the query and rule set fixed. It highlights the cost of reasoning over large databases.

## data compression


It is the process of encoding data using fewer bits than were used in the original representation so that the data consumes lesser disk space.

## data matrix

The arrangement of a data set as a matrix in which each row is an example and each column is a feature or attribute. It is a standard input format for many machine learning algorithms.

## data mining

The application of machine learning and statistical methods to discover useful patterns, regularities, or knowledge in large collections of data, often stored in databases.

## data-driven

Describes a reasoning or search process that proceeds forward from the available data or facts toward conclusions, as opposed to being goal-directed. Forward chaining is a data-driven inference method.

## database semantics

A set of simplifying assumptions used to interpret logical sentences: the unique-names assumption (distinct terms denote distinct objects), the closed-world assumption (unstated facts are false), and domain closure (no objects exist beyond those named).

## Datalog

A restricted first-order logic language consisting of function-free definite clauses (Horn clauses with no function symbols). It supports efficient, decidable inference and is used in deductive databases.

## Davis-Putnam algorithm

A complete backtracking search algorithm (more precisely the DPLL variant) for deciding the satisfiability of propositional sentences in conjunctive normal form, enhanced with unit propagation and pure-symbol heuristics.

## decayed MCMC

A Markov chain Monte Carlo variant for inference in temporal models in which older state variables are resampled less frequently, letting recent variables dominate the computation while keeping the chain valid.

## decentralized planning

Planning by multiple agents in which each agent constructs and executes part of a joint plan, coordinating through communication or conventions rather than relying on a single central planner.

## decision analysis

The discipline, drawing on decision theory, concerned with helping a decision maker choose rational actions under uncertainty, typically by eliciting utilities and probabilities and computing expected utilities.

## decision boundary

The surface in the input space that separates examples classified into one category from those classified into another. A linear classifier has a linear (hyperplane) decision boundary.

## decision maker

In decision analysis, the person or agent whose preferences (utilities) determine which action is chosen; distinguished from the decision analyst who structures and evaluates the problem.

## decision network

A graphical representation, also called an influence diagram, that extends a Bayesian network with decision (choice) nodes and a utility node, allowing rational decisions to be computed by maximizing expected utility.

## decision theory

The theory that combines probability theory with utility theory to prescribe rational action under uncertainty: a rational agent should choose the action that maximizes its expected utility.

## decision tree

A decision tree is a construct that uses a tree like graph or model of decisions and their possible consequences,including chance event outcomes,resource costs and utility.

## declarative

Describes an approach to system building in which knowledge is expressed as explicit sentences in a representation language and used by a general reasoning process, rather than being compiled directly into procedures (the procedural approach).

## declarative bias

In inductive learning, prior knowledge stated explicitly (for example, as constraints on the form of allowable hypotheses) that restricts the hypothesis space and guides the learner.

## decomposition

The technique of breaking a problem into smaller, more manageable subproblems whose solutions can be combined. In hierarchical planning, a high-level action is decomposed into a sequence of lower-level actions.

## deduction theorem

The result that a sentence B is entailed by a knowledge base KB if and only if the implication (KB implies B) is valid, connecting the notions of entailment and logical validity.

## deductive learning

Going from a known
general rule to a new rule that is logically entailed (and thus nothing new), but is nevertheless useful
because it allows more efficient processing.  

## deep belief networks

Generative neural network models composed of stacked layers of stochastic latent variables (often restricted Boltzmann machines) that can be trained layer by layer in an unsupervised fashion and were influential in the revival of deep learning.

## deep learning

It is a subfield of Machine Learning that tries to map the working of the human brain in processing data and creating patterns to use in decision making.

## default logic

A formalism for nonmonotonic reasoning that permits default rules of the form "if premise holds and it is consistent to assume conclusion, then conclude it," allowing plausible conclusions that may be retracted when new information arrives.

## definite clause

A disjunction of literals containing exactly one positive literal; equivalently, an implication whose premise is a conjunction of positive literals and whose conclusion is a single positive literal. Forward and backward chaining operate on definite clauses.

## definite clause grammar

A grammar formalism, often implemented in logic programming, in which grammar rules are written as definite clauses so that parsing and generation reduce to logical inference.

## definition of a rational agent

The characterization of a rational agent as one that, for each possible percept sequence, selects an action expected to maximize its performance measure given the evidence provided by the percepts and its built-in knowledge.

## deformable template

A model for object recognition in which an object's shape is represented by a template that can be flexibly stretched or bent to match variations in the appearance of instances.

## degree of belief

The subjective probability an agent assigns to a proposition, expressing its confidence that the proposition is true on a scale from 0 to 1.

## degree of freedom

It is a statistics term which represents the number of variables that you are allowed to change for analysis, without any constraint violation.

## delete list

In a STRIPS-style planning action, the list of literals that become false (are removed from the state) when the action is executed.

## deliberative layer

In a layered agent architecture, the higher-level component responsible for planning and reasoning about goals over longer time horizons, sitting above the reactive layer that handles immediate responses.

## demonic nondeterminism

Nondeterminism in which an adversary chooses the outcomes of the agent's actions in the worst possible way, so a plan must succeed no matter which outcome occurs; contrasted with angelic nondeterminism.

## Dempster-Shafer theory

A theory of evidence that represents uncertainty using belief functions, assigning mass to sets of possibilities and distinguishing degree of belief from the plausibility of a proposition, thereby modeling ignorance separately from probability.

## depth

In a search tree, the number of steps (actions) along the path from the root node to a given node.

## depth of field

In imaging, the range of distances from the camera over which objects appear acceptably sharp; objects outside this range are blurred by the finite aperture of the lens.

## depth-first search

It is an algorithm which allows us to traverse a graph or a tree data structure; it starts from the root node and traverses as far as possible for each branch before backtracking. (In case of graph data structure, the root node would be any arbitrary node that you select).

## depth-limited search

A variant of depth-first search that imposes a fixed cutoff depth, treating nodes at that depth as if they had no successors. It avoids infinite descent in infinite state spaces but may miss solutions deeper than the limit.

## detailed balance

A condition on a Markov chain guaranteeing that a given distribution is stationary: the probability flow from any state to another equals the flow in the reverse direction. It is exploited by MCMC methods such as Metropolis-Hastings.

## Deterministic

Describes an environment (or transition model) in which the next state is completely determined by the current state and the agent's action, with no randomness involved.

## diachronic

Pertaining to relationships that hold across time. A diachronic sentence links variables at different time steps, as in the transition model of a temporal probability model.

## diagnostic

Describes reasoning or a rule that infers from observed effects (evidence) back to their causes, as opposed to causal or predictive reasoning that goes from causes to effects.

## diameter

The maximum over all pairs of states of the shortest-path distance between them; it bounds how far apart any two states can be in a state space.

## Differential GPS

An enhancement to satellite positioning in which fixed ground reference stations broadcast corrections, greatly improving location accuracy for nearby receivers.

## diffuse albedo

The fraction of incoming light that a surface reflects diffusely (equally in all directions), determining the perceived matte brightness or color of the surface independent of viewing angle.

## Diophantine equations

Polynomial equations for which only integer solutions are sought. Determining whether such equations have solutions is a classic problem whose general form is undecidable.

## direct utility estimation

A passive reinforcement learning method that estimates the utility of a state as the average of the total observed reward-to-go over the trials that pass through it, treating utility learning as supervised regression.

## Dirichlet process

A distribution over probability distributions used as a prior in nonparametric Bayesian models, allowing the number of mixture components (e.g., clusters) to be unbounded and inferred from data.

## disambiguation

In natural language understanding, the process of selecting the intended interpretation of an ambiguous word, phrase, or sentence from among its possible meanings using context and world knowledge.

## discount factor

A number between 0 and 1, denoted gamma, that weights future rewards relative to immediate ones in a sequential decision problem, making rewards received later count less and keeping infinite-horizon utilities finite.

## discrete

Describes a variable, environment, or model in which the values or states form a countable, distinct set rather than a continuous range.

## discretization

The process of converting continuous quantities into a finite set of discrete values or intervals, for example to apply methods that require discrete inputs.

## disjoint

Describes sets or events that share no elements or cannot occur together; mutually exclusive events are disjoint.

## disjunction

A logical sentence formed by connecting subsentences with "or"; it is true when at least one of its disjuncts is true.

## disjunctive constraint

In a constraint satisfaction problem, a constraint that is satisfied if any one of several alternative conditions holds, such as requiring that two tasks not overlap by scheduling one before or after the other.

## disparity

In stereo vision, the difference in the image position of a scene point as seen by the two eyes or cameras; disparity is inversely related to the point's distance and is used to recover depth.

## distant point light source

An idealized light source modeled as located infinitely far away, so its rays arrive parallel and with uniform direction and intensity across the scene, like sunlight.

## distortion

A systematic geometric deviation introduced by a lens or imaging process that causes straight lines in the scene to appear curved or otherwise displaced in the image.

## distributed constraint satisfaction

The solving of a constraint satisfaction problem by multiple agents, each controlling a subset of the variables and communicating to satisfy the shared constraints without a central solver.

## DL

An abbreviation for description logic, a family of formal languages for representing structured knowledge about categories, their definitions, and the relationships among individuals, with tractable subsumption inference.

## domain

The set of possible values that a variable may take; in a constraint satisfaction problem, each variable is associated with a domain of candidate values. In first-order logic, the domain is the set of objects a model contains.

## domain closure

The assumption that the objects named in a knowledge base are the only objects that exist, so no unmentioned individuals are considered when evaluating sentences.

## dominant strategy equilibrium

An outcome of a game in which every player is playing a dominant strategy, one that yields at least as good a payoff as any alternative regardless of what the other players do.

## downward refinement property

A property of a hierarchical planning library guaranteeing that every high-level plan that appears to achieve the goal can in fact be refined into a primitive plan that actually achieves it, making abstract reasoning sound.

## dropping conditions

A generalization operation in inductive learning that removes a condition from a rule's premise, thereby making the rule apply to more cases.

## DT

An abbreviation for decision theory, the framework combining probability and utility to determine expected-utility-maximizing actions under uncertainty.

## dual graph

A transformation of a constraint satisfaction problem into an equivalent graph in which each original constraint becomes a variable, allowing multi-way constraints to be treated as binary constraints.

## dualism

The philosophical position that mind and body (or mental and physical phenomena) are fundamentally distinct kinds of substance, contrasted with materialist views that treat the mind as physical.

## duration

The length of time an action or event takes; in temporal and scheduling problems, actions are annotated with durations so that plans respect timing and resource constraints.

## dynamic

Describes an environment that can change while the agent is deliberating, so that the passage of time affects the state and the agent must act in a timely manner; contrasted with a static environment.

## dynamic backtracking

A backtracking search technique that, on encountering a conflict, jumps back to and revises a relevant earlier variable assignment while preserving the intervening assignments that were not implicated in the failure.

## dynamic Bayesian network

A Bayesian network that represents a temporal probability model as a repeated structure of state variables across time slices, compactly encoding how the state evolves and how observations depend on it. Hidden Markov models and Kalman filters are special cases.

## dynamic programming

A method of solving complex problems by breaking them down to sub-problems that can be solved by back tracking from the last stage. Used in popular real-world problems including traveling salesman problem, Fibonacci sequence, knapsack problem, etc.

## dynamic state

The portion of a robot's or agent's state that changes over time and includes rates of change, such as velocities and accelerations, in contrast to the kinematic state, which captures only configuration (positions).

# E

## early stopping
A regularization technique for iterative learning algorithms (such as gradient descent on neural networks) in which training is halted as soon as validation-set performance stops improving, preventing the model from overfitting the training data.

## economy
A market of many agents that produce, exchange, and consume goods; in AI it is studied as a multiagent system whose aggregate behavior emerges from individual agents pursuing their own utility.

## effect
In a planning action schema, the set of literals describing how the state of the world changes when the action is executed; add effects assert new fluents and delete effects remove them.

## effective branching factor
A measure of a search algorithm's efficiency: the branching factor that a uniform tree of the same total number of generated nodes and the same solution depth would have. A value close to 1 indicates a well-informed search.

## efficient
Describing a market or mechanism in which resources end up allocated so as to maximize total utility (Pareto efficiency), leaving no reallocation that would make one agent better off without making another worse off.

## electric motor
An actuator that converts electrical energy into mechanical rotation, commonly used to drive the effectors of a robot.

## eliminative materialism
The philosophical position that ordinary mental states such as beliefs and desires do not really exist and that a mature neuroscience will replace such folk-psychology concepts entirely.

## elitism
A strategy in genetic algorithms that carries the best individuals of one generation unchanged into the next, ensuring that the highest-quality solutions are never lost through crossover or mutation.

## embodied cognition
The view that intelligent behavior arises from the interaction of an agent's body and its environment, so that cognition cannot be fully understood as abstract symbol manipulation divorced from physical interaction.

## emergent behavior
Complex, often unplanned overall behavior that arises from the interaction of many simple components or agents, none of which individually exhibits that behavior.

## empirical gradient
An estimate of the gradient of an objective function obtained by measuring how the function changes for small perturbations of the inputs, used when an analytic gradient is unavailable.

## empirical loss
The average loss a hypothesis incurs on the actual set of observed training examples, used as a stand-in for the true expected loss over the whole distribution.

## empiricism
The philosophical doctrine that knowledge derives primarily from sensory experience, associated with the principle that nothing is in the understanding that was not first in the senses.

## English auction
An ascending-bid open auction in which bidders publicly raise the price until only one bidder remains, who wins and pays the final (highest) bid.

## entailment
The relation between sentences in which one sentence logically follows from another: a knowledge base entails a sentence if the sentence is true in every model in which the knowledge base is true.

## entropy
A measure of the uncertainty or information content of a random variable, equal to the expected number of bits needed to encode its value; it is maximized for a uniform distribution and zero for a certain outcome.

## environment
Everything external to an agent that the agent perceives through sensors and affects through actuators; the setting in which the agent operates and against which its performance is measured.

## environment generator
A program that produces a range of environments (task instances) drawn from some class, used to evaluate an agent's average performance across many situations rather than a single fixed one.

## episodic
A property of a task environment in which the agent's experience is divided into independent episodes, so that the choice of action in one episode does not affect later episodes.

## epsilon-ball
The set of all points within distance epsilon of a given point, used, for example, to characterize local neighborhoods in continuous optimization and convergence arguments.

## equality symbol
A logical symbol used in first-order logic to assert that two terms refer to the same object in the world.

## equilibrium
A joint strategy profile in a game from which no agent has an incentive to deviate unilaterally; the Nash equilibrium is the canonical example.

## ergodic
A property of a Markov process in which, from any state, it is possible to eventually reach any other state, so that the long-run distribution over states is independent of the starting state.

## error rate
The proportion of examples on which a hypothesis makes an incorrect prediction, used as a simple measure of classification performance.

## event
In probability theory, a set of outcomes (a subset of the sample space) to which a probability is assigned; more generally, an occurrence in the world represented at a point or interval in time.

## evidence
The values of variables that have actually been observed, which are conditioned upon when computing posterior probabilities of query variables.

## evidence reversal
A transformation of a Bayesian network (or dynamic model) that reverses the direction of an arc so that evidence variables become parents, sometimes used to simplify inference.

## evolutionary algorithms
A family of stochastic optimization methods inspired by natural selection, in which a population of candidate solutions is repeatedly evaluated, selected, recombined, and mutated to produce better solutions over successive generations.

## evolutionary psychology
The study of human mental traits as adaptations shaped by natural selection, seeking to explain cognition and behavior in terms of their evolutionary function.

## evolutionary strategies
A class of evolutionary algorithms, typically applied to real-valued vectors, in which mutation step sizes are themselves adapted over the course of the search.

## exact cell decomposition
A robot motion-planning method that partitions free space into a finite set of cells whose shapes exactly cover the free space, allowing motion within and between adjacent cells to be planned exactly.

## execution
The stage at which an agent carries out a chosen plan or action in the environment, as distinct from the deliberation or planning that produced it.

## execution monitoring
Watching the world during plan execution to detect when reality diverges from the plan's expectations, so that the agent can repair or replan when necessary.

## executive layer
In a layered robot control architecture, the middle layer that translates high-level goals from the planning layer into sequences of primitive actions and supervises their execution.

## exhaustive decomposition
A partition of a category into subcategories that together cover the whole category, so that every member of the category belongs to at least one subcategory.

## existence uncertainty
Uncertainty about whether an object exists at all, as opposed to uncertainty about its properties; a central issue in open-universe probability models.

## Existential Instantiation
An inference rule of first-order logic that allows an existentially quantified sentence to be replaced by an instance in which the quantified variable is substituted by a new constant symbol (a Skolem constant) not appearing elsewhere.

## expand
In tree and graph search, to generate the successor nodes of a given node by applying the applicable actions, thereby extending the frontier of the search.

## expectation
The average or mean value of a random variable, computed as the sum (or integral) of each possible value weighted by its probability.

## expectation-maximization
An iterative algorithm for finding maximum-likelihood parameter estimates in models with hidden variables, alternating between computing expected values of the hidden variables (E-step) and re-estimating parameters to maximize likelihood (M-step).

## expected value
The probability-weighted average of the possible outcomes of a random variable or of an action, used as the basis for decision making under uncertainty.

## expectiminimax value
The value of a game state in games that combine adversarial moves with chance events, computed by taking minima and maxima at the players' nodes and expectations (probability-weighted averages) at the chance nodes.

## explanation-based learning
A form of analytical learning in which the agent generalizes from a single example by constructing an explanation of why the example satisfies a concept, using its background knowledge, and then extracting a general rule.

## explanatory gap
The apparent difficulty of explaining how and why physical processes in the brain give rise to subjective conscious experience.

## exploitation
In sequential decision making, choosing the action currently believed to yield the highest reward, as opposed to trying alternatives to gather information.

## exploration
In sequential decision making, choosing actions in order to gather information about the environment, even at the cost of short-term reward, so as to improve future decisions.

## exploration problem
The problem of balancing exploration and exploitation when an agent must act in an unknown environment, deciding when to try new actions versus exploiting known good ones.

## expressiveness
The range of facts and relationships that a representation language can capture; more expressive languages can describe more, often at the cost of harder inference.

## extended Kalman filter (EKF)
A variant of the Kalman filter that handles nonlinear transition or measurement models by linearizing them around the current estimate at each step, so that the standard Gaussian update equations can be applied.

## extension
Of a predicate or category, the set of all objects in the world that satisfy it, as contrasted with its intension (defining property).

## extensive form
A representation of a game as a tree that makes explicit the sequence of moves, the information available at each decision point, and the payoffs at the leaves.

## externalities
Costs or benefits of a transaction that fall on parties not directly involved in it, which market mechanisms may fail to account for unless deliberately incorporated.

## extrinsic
Describing a property of an object that depends on its relationships to other things rather than on the object's own substance; such properties are not preserved when the object is divided.

# F
## fact
An atomic sentence asserting a piece of information known to be true, stored in a knowledge base.

## factor
In variable elimination for probabilistic inference, a matrix (or table) indexed by the values of a set of variables, representing part of a joint distribution; factors are combined by pointwise product and reduced by summing out variables.

## factored frontier
An approximate filtering technique for dynamic Bayesian networks that represents the belief state as a product of factors over subsets of state variables rather than as a full joint distribution, trading accuracy for tractability.

## factored representation
A representation in which each state is described by a fixed set of variables or attributes, each with a value, rather than as an indivisible atom or a set of related objects.

## factorial HMM
A hidden Markov model whose hidden state is decomposed into several independently evolving state variables, so that the overall state is the combination (product) of these separate chains.

## factoring
In resolution theorem proving, the step of removing redundant literals from a clause by unifying two literals that can be made identical, so the clause contains only one copy.

## false negative
An example that is actually positive but is classified as negative by a hypothesis or test.

## false positive
An example that is actually negative but is classified as positive by a hypothesis or test.

## feature extraction
The process of computing informative attributes (features) from raw input data, producing a representation more suitable for learning or decision making.

## feature selection
The process of choosing a subset of the available features to use in a model, discarding irrelevant or redundant ones to improve accuracy, efficiency, or interpretability.

## feed-forward network
A neural network in which connections flow in one direction from inputs to outputs through successive layers, without cycles, so that the output is a direct function of the current input.

## FIFO queue
A first-in, first-out queue that returns elements in the order they were added; used by breadth-first search to expand nodes.

## filtering
The task of computing the posterior distribution over the current state of a system given all evidence observed up to and including the present, also called state estimation.

## finite horizon
A setting for sequential decision problems in which there is a fixed number of time steps after which nothing further matters, so optimal behavior may depend on how much time remains.

## first-choice hill climbing
A stochastic variant of hill climbing that generates successors at random one at a time and moves to the first one that is better than the current state, useful when a state has very many successors.

## first-order Markov process
A stochastic process in which the current state depends only on the immediately preceding state and not on any earlier states.

## fixate
In stereo vision, to direct both eyes (or cameras) so that their optical axes intersect at the same point in the scene, aligning that point's image on corresponding positions.

## fixed point
A value that is unchanged by an operator or update rule, so that applying the operator to it returns the same value; iterative algorithms often converge to a fixed point.

## fixed-lag smoothing
The task of computing the posterior distribution over the state at a time step that is a fixed number of steps in the past relative to the current observation, allowing an efficient online estimate that improves on filtering.

## flaw
In partial-order planning, an open condition or a threatened causal link in a partial plan that must be resolved before the plan is complete and correct.

## fluent
A term or predicate whose truth value or value can change over time; fluents are used to describe the aspects of the world that actions affect.

## focal plane
In an imaging system, the plane behind the lens where a sharp image forms; the image sensor or retina is placed at this plane.

## foreshortening
The apparent compression of a surface's dimensions in an image when the surface is slanted away from the viewing direction, an important cue for recovering shape.

## forward-backward algorithm
A dynamic-programming algorithm for smoothing in hidden Markov models that combines a forward pass computing state probabilities given past evidence with a backward pass computing the likelihood of future evidence, yielding the posterior over each state given all evidence.

## forward-chaining
A data-driven inference procedure that repeatedly applies rules to known facts to derive new facts, adding them to the knowledge base until the query is proved or no new facts can be added.

## frame problem
The problem of representing and reasoning efficiently about what stays unchanged when an action occurs, without having to state explicitly every fact that the action does not affect.

## frames
A knowledge-representation formalism that organizes information into structures (frames) with named slots holding attribute values and links to other frames, supporting inheritance and default values.

## framing effect
A cognitive bias in which people's choices depend on how logically equivalent options are described (for example as gains versus losses), violating the assumptions of rational decision theory.

## free space
In robot motion planning, the region of configuration space consisting of all configurations in which the robot does not collide with any obstacle.

## frequentist
An interpretation of probability that defines the probability of an event as the limiting relative frequency of its occurrence in repeated trials, rather than as a degree of belief.

## friendly AI
The design goal, and area of study, of building highly capable AI systems whose objectives remain beneficial to humans and that do not pose existential risks.

## frontier
In search, the set of nodes that have been generated but not yet expanded, representing the boundary between explored and unexplored parts of the state space.

## full joint probability distribution
The probability distribution that assigns a probability to every possible combination of values of all the random variables in a domain, from which any query can in principle be answered by summing out irrelevant variables.

## fully observable

If an agent's sensors give it access to the complete state of the environment at each point in time,then we say that the task environment is fully observable.

## functionalism
The philosophical position that mental states are defined by their functional roles—their causal relations to inputs, outputs, and other mental states—rather than by the physical substance that realizes them, implying that suitably organized systems could have minds.

## futility pruning
A game-search heuristic that avoids exploring moves that appear very unlikely to raise a position's value above the current best, saving effort on near-hopeless branches.

## fuzzy control
A fuzzy control system is a control system based on fuzzy logic—a mathematical system that analyzes analog input values in terms of logical variables that take on continuous values between 0 and 1, in contrast to classical or digital logic, which operates on discrete values of either 1 or 0 (true or false, respectively).

## fuzzy logic

Fuzzy logic is a form of many-valued logic in which the truth values of variables may be any real number between 0 and 1. It is employed to handle the concept of partial truth, where the truth value may range between completely true and completely false.

## fuzzy set theory

Fuzzy sets (aka uncertain sets) are somewhat like sets whose elements have degrees of membership. In classical set theory, the membership of elements in a set is assessed in binary terms according to a bivalent condition — an element either belongs or does not belong to the set. By contrast, fuzzy set theory permits the gradual assessment of the membership of elements in a set; this is described with the aid of a membership function valued in the real unit interval [0, 1].

# G

## G-set

In version-space learning, the G-set (or general boundary) is the set of the most general hypotheses that are consistent with the examples seen so far. Together with the S-set it delimits the version space of all consistent hypotheses.

## gain parameter

A scalar multiplier applied to a feedback signal in a control law; in a proportional controller it sets how strongly the actuator responds to the observed error, trading off responsiveness against stability.

## gain ratio

A measure used in decision-tree learning that normalizes information gain by the intrinsic information of an attribute's split, discouraging the selection of attributes that have many distinct values.

## gait

The characteristic pattern and timing with which a legged robot moves its limbs to locomote, such as a walking or trotting sequence that keeps the robot statically or dynamically stable.

## game theory

Game theory is the study of mathematical models of strategic interaction between rational decision-makers. It has applications in all fields of social science, as well as in logic and computer science. Originally, it addressed zero-sum games, in which one person's gains result in losses for the other participants. Today, game theory applies to a wide range of behavioral relations, and is now an umbrella term for the science of logical decision making in humans, animals, and computers.

## game tree

A tree in which the nodes are game states and the edges are moves, with the root being the initial position and the branches representing the possible ways the game can unfold as players alternate turns.

## Gaussian distribution

A continuous probability distribution, also called the normal distribution, defined by its mean and variance and having the familiar bell-shaped density curve; it is widely used to model noise and uncertain quantities.

## Gaussian error model

A model that assumes observation or process noise follows a Gaussian distribution centered on the true value, commonly used in filtering and sensor models.

## Gaussian filter

A convolution filter whose kernel is sampled from a Gaussian distribution, used in image processing to smooth an image and suppress high-frequency noise.

## Gaussian process

A nonparametric method that defines a probability distribution over functions, such that any finite set of function values has a joint Gaussian distribution; it is used for regression with calibrated uncertainty estimates.

## generalization

The ability of a learned model to make accurate predictions on new, previously unseen inputs rather than merely fitting the training examples.

## generalization hierarchy

A structure in which concepts or categories are organized from more specific to more general, so that a more general term subsumes the instances of the more specific terms below it.

## generalization loss

The expected error of a hypothesis over the full distribution of examples, as opposed to its measured error on the training set; it captures how well the hypothesis will perform on unseen data.

## generalized modus ponens

An inference rule for first-order logic that, given atomic sentences and an implication whose premises unify with them under some substitution, concludes the substituted consequent, combining unification with modus ponens in a single step.

## generating

In problem solving, the act of producing successor states by applying the available actions to a state, as done by a successor or expansion function during search.

## generator

A component that produces candidate outputs or examples; in a generate-and-test scheme it proposes solutions that another component then evaluates.

## genetic algorithms

A family of stochastic local search methods inspired by natural selection, in which a population of candidate solutions is evolved through selection, crossover, and mutation to optimize a fitness function.

## genetic programming

An evolutionary technique that applies genetic-algorithm operators to programs, typically represented as expression trees, evolving executable code that performs well on a given task.

## Gibbs sampling

A Markov chain Monte Carlo algorithm for approximate inference that generates samples by repeatedly resampling each variable in turn, conditioned on the current values of its Markov blanket.

## GLIE

Greedy in the Limit of Infinite Exploration: a property of an exploration strategy guaranteeing that every state-action pair is tried infinitely often while the policy converges to greedy behavior, ensuring convergence of certain reinforcement learning methods.

## global constraint

In constraint satisfaction, a constraint that can involve an arbitrary number of variables, such as Alldiff, which requires that a group of variables all take distinct values.

## global minimum

The point at which a function attains its lowest value over its entire domain, as distinguished from a local minimum that is lowest only within a neighborhood.

## Go

An ancient two-player board game played by placing stones on a grid; its enormous branching factor made it a longstanding challenge for AI until programs using deep learning and Monte Carlo tree search achieved superhuman play.

## goal

A state or set of states that an agent is trying to reach; goals help organize behavior by describing situations that are desirable.

## goal clauses

In resolution-based theorem proving, clauses that encode the negation of the query to be proved; deriving the empty clause from them together with the knowledge base establishes the query.

## goal formulation

The process, based on the current situation and the agent's performance measure, of deciding on a set of goal states that the agent will try to achieve, thereby focusing its subsequent behavior.

## goal monitoring

The process by which an executing agent checks whether its current plan will still achieve the goal, replanning if changes in the environment have made the plan inadequate.

## goal test

A function in a search problem that determines whether a given state is a goal state, allowing the search algorithm to recognize when a solution has been found.

## goal-directed reasoning

Reasoning that works backward from a goal, chaining through rules to find the facts or actions that would achieve it; contrasted with data-driven reasoning that works forward from known facts.

## gold standard

A reference set of correct answers, typically produced or verified by humans, against which a system's outputs are compared to measure its accuracy.

## gorilla problem

The concern that humanity, like gorillas relative to humans, could lose control of its future by creating machines far more intelligent than itself.

## gradient

The vector of partial derivatives of a function with respect to its parameters; it points in the direction of steepest increase and is central to optimization methods.

## gradient descent

An optimization algorithm that iteratively adjusts parameters in the direction opposite to the gradient of a loss function, taking steps proportional to a learning rate in order to reach a minimum.

## grammar

A formal specification of the allowable structure of strings in a language, consisting of rewrite rules over terminal and nonterminal symbols.

## graph

A mathematical structure consisting of nodes (vertices) connected by edges, used to represent relationships such as the connections between states in a search space.

## graph coloring

The problem of assigning colors to the nodes of a graph so that no two adjacent nodes share a color, a canonical example of a constraint satisfaction problem.

## grasping

The robotics task of using a manipulator or hand to securely hold an object, requiring reasoning about contact points, forces, and object geometry.

## greedy agent

An agent that always chooses the action that looks best according to its immediate evaluation, without accounting for longer-term consequences.

## greedy best-first search

A best-first search strategy that expands the node estimated to be closest to the goal by a heuristic function, ignoring the cost already incurred to reach it.

## grid world

A simple, commonly used environment consisting of a rectangular grid of cells through which an agent moves, often used to illustrate search, planning, and reinforcement learning problems.

## ground term

A term in first-order logic that contains no variables, denoting a specific object in the domain.

## grounding

The connection between the symbols an agent manipulates and the real-world entities and situations they are meant to refer to, which gives the symbols their meaning.

# H

## Hamming distance

A measure of dissimilarity between two strings of equal length, equal to the number of positions at which the corresponding symbols differ.

## Hansard

The official transcribed record of parliamentary debates, notably in Canada published in both English and French, which serves as a large parallel corpus for training and evaluating machine translation systems.

## haptic feedback

Sensory feedback conveyed through the sense of touch, such as forces or vibrations, used in teleoperation and robotics to give an operator or system information about physical contact.

## head

In a definite or Horn clause, the single positive literal that is the conclusion of the implication, as opposed to the body of premises.

## heavy-tailed distribution

A probability distribution whose tail decays more slowly than an exponential, so that extreme values occur with non-negligible probability; such distributions arise, for example, in the run times of backtracking search.

## Hebbian learning

A learning rule in which the connection between two neurons is strengthened when they are activated together, often summarized as "cells that fire together wire together."

## Hessian

The square matrix of second-order partial derivatives of a function, describing its local curvature and used in second-order optimization methods.

## heuristic function

A function that estimates the cost of the cheapest path from a given state to a goal, used to guide informed search algorithms toward promising states.

## heuristic search

A search strategy that uses problem-specific knowledge, typically in the form of a heuristic function, to prioritize which states to explore and thereby find solutions more efficiently than uninformed search.

## hidden Markov model

A hidden Markov model (or HMM) is a temporal probabilistic model in which the state of the process is described by a *single discrete*
random variable.

## hierarchical lookahead

Planning that reasons at multiple levels of abstraction, examining the effects of high-level actions before committing to their detailed implementations, so that infeasible plans can be pruned early.

## hierarchical reinforcement learning

An approach to reinforcement learning that decomposes a task into a hierarchy of subtasks or temporally extended actions, allowing an agent to learn and reuse reusable behaviors and to cope with large state spaces.

## high-level action

An abstract action that can be carried out in multiple ways by decomposing it into sequences of lower-level actions or primitive steps, as used in hierarchical planning.

## Hinton diagrams

A graphical way of displaying the weights of a neural network, in which each weight is shown as a square whose size encodes magnitude and whose color or shading encodes sign.

## holdout cross-validation

A validation method in which the available data is split into a training set and a separate held-out test set, with the model trained on the former and evaluated on the latter.

## homeostatic

Describing a system that acts to keep certain internal variables within preferred ranges, regulating its behavior to maintain a stable internal state.

## homophones

Words that are pronounced the same but differ in meaning or spelling, posing a source of ambiguity for speech recognition systems.

## horizon effect

A pitfall in game-playing search whereby a program, limited by its search depth, pushes an unavoidable damaging event beyond its lookahead horizon and thereby fails to account for it.

## Horn clause

A disjunction of literals containing at most one positive literal; such clauses can be written as implications and support efficient inference procedures like forward and backward chaining.

## hub

In a network or link-analysis context, a node that points to many authoritative pages; hub and authority scores are computed together to rank web pages.

## human-level AI

Artificial intelligence that matches human ability across the full breadth of cognitive tasks, rather than excelling only in narrow domains.

## Hungarian algorithm

A combinatorial optimization method that solves the assignment problem, finding a minimum-cost one-to-one matching between two sets in polynomial time.

## hybrid A*

A variant of the A* search algorithm adapted for continuous motion planning, in which states include continuous coordinates and headings so that the resulting paths respect a vehicle's kinematic constraints.

## hybrid agent

An agent that combines multiple reasoning components, such as reactive behaviors together with deliberative planning and logical inference, to select actions in complex environments.

## hybrid architecture

An agent design that integrates several distinct control paradigms, typically layering reactive, executive, and deliberative levels so that fast reflexes and slower planning coexist.

## hybrid Bayesian network

A Bayesian network that contains both discrete and continuous random variables, requiring conditional distributions that link the two kinds of variables.

## hydraulic actuation

The use of pressurized fluid to drive a robot's joints or actuators, providing high force and power density at the cost of complexity and potential leakage.

## hypothesis

In learning, a candidate function drawn from a hypothesis space that is proposed to explain the observed data and to predict future examples.

## hypothesis prior

A probability distribution over the possible hypotheses that expresses prior belief in each one before any data is observed, used in Bayesian learning.

## hypothesis space

The set of all candidate hypotheses that a learning algorithm can consider and choose among when fitting the data.

# I

## i.i.d.

**i.i.d.** denotes **independent and identically distributed** random variables. They are defined on the same probability space, have identical probability distribution functions, and are mutually independent.

## identification in the limit

A criterion for learning success in which a learner is said to succeed if, after some finite amount of data, it settles on the correct hypothesis and never changes its answer thereafter, even though it may never know at which point it has converged.

## identity matrix

A square matrix with ones on the main diagonal and zeros elsewhere, which leaves any vector or matrix unchanged when multiplied by it.

## identity uncertainty

The problem of determining whether two distinct references or observations actually denote the same underlying object, a key issue in data association and record linkage.

## ignore delete lists

A relaxation used to compute planning heuristics in which the negative (delete) effects of actions are ignored, so that once a fluent becomes true it stays true, making the relaxed problem easier to solve.

## ignore preconditions heuristic

A planning heuristic that drops all preconditions from actions, so every action becomes applicable in every state, yielding an easier relaxed problem whose solution length estimates the cost to the goal.

## image

A rectangular array of pixels, each recording the light intensity (and possibly color) arriving at a point in the image plane, serving as the primary input for computer vision.

## imperfect information

A condition in games and decision problems where an agent lacks full knowledge of the current state, such as not seeing an opponent's cards or hidden aspects of the environment.

## implementation

The concrete realization of an agent design as an actual program running on some architecture, mapping the abstract agent function to executable code.

## implementation level

The lowest of the levels for describing a system, concerned with the physical or programmatic mechanisms that actually carry out the computation, as opposed to the knowledge or symbol levels.

## implication

A logical connective, written as a conditional (if-then) sentence, that is false only when its premise (antecedent) is true and its conclusion (consequent) is false.

## importance sampling

A technique for estimating expectations under one distribution by sampling from a different, more convenient distribution and reweighting the samples to correct for the mismatch.

## inclusion-exclusion principle

A counting rule stating that the probability (or size) of a union of sets equals the sum of the individual probabilities minus the pairwise intersections, plus triple intersections, and so on, to avoid double-counting overlaps.

## incompleteness theorem

Gödel's result showing that any consistent formal system powerful enough to express arithmetic contains true statements that cannot be proved within that system.

## incremental belief-state search

A search over belief states in which the solution for reaching a goal from a set of possible states is built up by finding a plan for one physical state and then checking and extending it to cover the remaining states.

## independence

A relationship between two random variables (or events) in which knowing the value of one gives no information about the other, so their joint distribution factors into the product of their marginals.

## independent subproblems

Parts of a problem that share no variables or interactions, so each can be solved separately and the results combined, often giving large computational savings.

## index

A data structure that maps keys to the locations of matching items, allowing sentences or records containing particular terms to be retrieved quickly.

## indexed random variable

A random variable that is one member of a family distinguished by an index, such as a time step or object identifier, used to model repeated or object-specific quantities.

## indexing

The process of organizing a knowledge base or document collection so that relevant items can be fetched efficiently, for example by storing pointers to sentences that could unify with a query.

## individuation

The process of carving the world into distinct objects or entities, which is required before those objects can be counted, named, or reasoned about.

## induction

Reasoning from specific observations to general conclusions or rules, which unlike deduction can produce conclusions that are only probable rather than logically guaranteed.

## inductive learning

Going from a set of specific input-output pairs
to a (possibly incorrect) general rule  is called **inductive learning**.

## inductive logic

The study of principles for reasoning under uncertainty from evidence to hypotheses, in which conclusions are supported to some degree rather than entailed with certainty.

## inductive logic programming

A machine learning approach that combines inductive learning with first-order logic, generating logical rules (as logic programs) that explain a set of examples given background knowledge.

## inference

The process of deriving new sentences or conclusions from existing ones, typically by applying rules or algorithms to a knowledge base to answer queries.

## inference rules

Formal patterns, such as Modus Ponens or resolution, that license deriving new sentences from given ones while preserving truth.

## inferential frame problem

The difficulty of efficiently reasoning through the many facts that remain unchanged by an action, rather than having to explicitly re-derive them at each step.

## infinite

Describing a quantity, set, or sequence without bound, such as an infinite state space or a decision problem with no fixed termination.

## infinite horizon

A setting for sequential decision problems in which there is no fixed limit on the number of time steps, so the agent may act indefinitely, often requiring discounting to keep total reward finite.

## infix

Notation in which an operator is written between its operands, such as writing a plus sign between two numbers, as opposed to prefix or postfix notation.

## information extraction

The task of automatically pulling structured facts or relationships out of unstructured text, such as identifying entities and the relations that hold among them.

## information gain

The expected reduction in entropy about a target variable achieved by learning the value of an attribute, used to select the most informative attribute when building decision trees.

## information gathering

Actions taken by an agent specifically to acquire knowledge (for example, sensing or asking questions) so as to reduce uncertainty and improve later decisions.

## information retrieval

The task of finding documents from a large collection that are relevant to a user's query, as performed by search engines.

## information sets

In games with imperfect information, a set of game states that a player cannot distinguish among given their observations, so the player must choose the same action across all states in the set.

## informed search

A search strategy that uses problem-specific knowledge, typically a heuristic function estimating distance to the goal, to guide node expansion more efficiently than uninformed search.

## inheritance

A mechanism in semantic networks and object hierarchies by which properties of a category are automatically assumed to hold for its members and subcategories.

## initial state

The state in which an agent or search problem begins, from which actions are taken to reach a goal.

## input resolution

A restricted form of resolution in which each resolution step must use at least one of the original input clauses, yielding a more constrained but incomplete proof strategy.

## inside-outside algorithm

A dynamic-programming algorithm for learning the rule probabilities of a probabilistic context-free grammar from unparsed sentences, analogous to the forward-backward algorithm for HMMs.

## insurance premium

The amount a risk-averse agent is willing to pay to avoid a risky outcome, reflecting the gap between the expected monetary value of a gamble and its lower certainty equivalent utility.

## intelligence

The capacity to act rationally, that is, to perceive an environment and choose actions expected to achieve one's goals, which is the property AI seeks to build into machines.

## interleaving

Alternating between planning and execution (or between multiple activities) rather than completing one fully before starting the other, allowing an agent to react to new information as it acts.

## interlingua

A language-neutral intermediate representation of meaning used in machine translation, into which a source sentence is analyzed and from which the target sentence is generated.

## internal state

Information an agent maintains about aspects of the world it cannot currently perceive, updated over time to track how the environment evolves.

## interpretation

A mapping that assigns meaning to the symbols of a logical language by associating constants with objects, predicates with relations, and functions with mappings in a domain.

## interreflections

In rendering and vision, the light that bounces between surfaces so that one surface illuminates another, contributing indirect illumination that complicates recovery of scene properties.

## intrinsic

Describing a property that belongs to the substance or material itself and is shared by any portion of it, such as density or color, as opposed to properties like mass that depend on the whole object.

## intuition pump

A thought experiment, such as Searle's Chinese Room, designed to elicit and shape intuitions about a philosophical question, for instance whether machines can truly think.

## inverse

The operation or element that undoes another, such as a matrix that when multiplied by a given matrix yields the identity, or a function that reverses another function's mapping.

## inverse entailment

An operation used in inductive logic programming that inverts the direction of logical entailment to generate candidate hypotheses that, together with background knowledge, would entail the observed examples.

## inverse kinematics

The problem of computing the joint configurations (such as angles) of a robot needed to place its end effector at a desired position and orientation in space.

## inverse reinforcement learning

Learning an agent's reward function by observing its behavior, inferring what objective would make the observed policy optimal, rather than being told the rewards directly.

## inverted pendulum

A classic control benchmark in which a pole hinged on a moving cart must be balanced upright by moving the cart, testing an agent's ability to stabilize an unstable system.

## inverted spectrum

A philosophical scenario in which two people have systematically swapped color experiences yet behave identically, used to argue about the nature of qualia and subjective experience.

## IR

Abbreviation for **information retrieval**, the task of locating documents relevant to a query within a large collection.

## irreversible

Describing an action or environment in which some choices cannot be undone, so an agent must be cautious because mistakes may be permanent.

## iterative deepening search

An uninformed search that runs depth-limited search with progressively increasing depth limits, combining the low memory use of depth-first search with the completeness and optimality of breadth-first search.

## iterative expansion

A search technique that repeatedly expands and re-expands parts of the search space under growing resource bounds, trading recomputation for reduced memory usage.

## iterative-deepening A*

A memory-efficient variant of A* that performs successive depth-first searches, each bounded by an increasing limit on the f-cost rather than on depth.
# J

## join tree
A tree-structured representation of a constraint graph or Bayesian network in which each node is a cluster of variables, arranged so that inference can be carried out efficiently by message passing between clusters.

## joint action
In a multiagent setting, a combined action consisting of one action chosen by each agent, so that the outcome depends on the simultaneous choices of all agents.

## joint plan
A plan constructed for a group of agents that specifies the coordinated actions each agent must take to achieve a shared goal.

## JTMS
A justification-based truth maintenance system: a reason-maintenance system in which each sentence is annotated with a justification—the set of sentences from which it was inferred—so that beliefs can be efficiently retracted and restored as assumptions change.

## justification
The record attached to a sentence in a truth maintenance system that lists the other sentences on which it depends, allowing the system to track and update belief when premises change.

# K

## k-d tree
A balanced binary tree that recursively partitions k-dimensional space along coordinate axes, used to speed up nearest-neighbor queries in low-dimensional data.

## K-means clustering
An unsupervised learning algorithm that partitions data into k clusters by iteratively assigning each point to the nearest cluster center and then recomputing each center as the mean of its assigned points.

## Kalman filtering
A method for updating the estimate of a continuous state over time from a stream of noisy observations, assuming linear transition and sensor models with Gaussian noise, yielding a Gaussian belief state.

## Kalman gain matrix
The matrix in the Kalman filter update that determines how much the new observation, relative to the predicted state, adjusts the current state estimate.

## kernel
A function that computes the inner product of two input vectors in an implicit higher-dimensional feature space, allowing linear methods to model nonlinear relationships.

## kernel function
A function K(x, z) that returns the dot product of the feature-space mappings of x and z, enabling algorithms such as support vector machines to operate in a transformed space without computing the mapping explicitly.

## kernel trick
The technique of replacing dot products in a learning algorithm with kernel evaluations, so that the algorithm implicitly works in a high-dimensional feature space without ever representing the vectors there.

## kinematic state
A description of a robot's configuration in terms of positions (and sometimes orientations) of its parts, ignoring velocities and forces.

## kinematics
The study of how the positions and configurations of a robot's links relate to one another, without regard to the forces that produce motion.

## King Midas problem
The difficulty that arises when an agent is given an objective that is literally optimized but does not capture what its designer actually wanted, so pursuing it leads to undesirable outcomes.

## knowledge acquisition
The process by which a knowledge engineer, often working with domain experts, gathers and encodes the facts and rules needed to populate a knowledge base.

## knowledge base
A set of sentences, expressed in a knowledge representation language, that represent assertions about the world and can be queried and updated by an agent.

## knowledge engineering
The task of investigating a domain, deciding what concepts to represent, and formalizing them as logical sentences so that a system can reason about the domain.

## knowledge-based agents
Agents whose behavior is driven by an explicit knowledge base and an inference mechanism, allowing them to derive new conclusions and decide on actions from what they know.

## Known
An environment property indicating that the agent's designer or the agent itself has full knowledge of the outcomes (or outcome probabilities) of all actions, as opposed to having to learn them.

## Kriegspiel
A partially observable variant of chess in which each player cannot see the opponent's pieces and must infer their positions from the referee's announcements about legal and illegal moves.

## Kullback-Leibler divergence
A non-symmetric measure of how much one probability distribution differs from another, equal to the expected log ratio of the two distributions.

# L

## label
The correct output value attached to a training example in supervised learning; for classification it is a class, for regression a numeric value.

## Lambert's cosine law
The rule in image formation that the brightness of a diffusely reflecting surface is proportional to the cosine of the angle between the incoming light and the surface normal.

## landmarks
Distinctive, recognizable features in the environment that a robot uses as reference points for localization and mapping.

## language
A set of strings over some alphabet; in AIMA this covers both formal languages, defined by precise grammars, and natural languages used by humans.

## language generation
The task of producing natural-language text (or speech) from an internal representation of meaning or intent.

## language identification
The task of determining which natural language a given piece of text or speech is written or spoken in.

## large-scale learning
Learning in settings where the amount of training data is so large that the main limitation is computation time rather than the availability of examples.

## layers
The groups of units in a neural network, arranged so that the outputs of one layer feed as inputs to the next, from the input layer through hidden layers to the output layer.

## leak node
An extra parent added in a noisy-OR model to account for all the miscellaneous, unmodeled causes that could produce an effect.

## learning

An agent is **learning** if it improves its
performance after making observations about the world.

## learning curve
A plot showing how a learning algorithm's performance changes as a function of the amount of training data or the number of training iterations.

## learning element
The component of a learning agent responsible for making improvements, using feedback from the critic to modify the agent's performance element.

## learning rate
A parameter in gradient-based learning that controls the size of the step taken in the direction of the gradient at each update.

## least commitment
A design principle, common in planning, of deferring choices about ordering, bindings, or other details until they are forced, so as to keep options open and reduce backtracking.

## least-constraining-value
A value-ordering heuristic for constraint satisfaction that prefers the value which rules out the fewest choices for the neighboring unassigned variables.

## leave-one-out cross-validation
A validation method in which, for a data set of n examples, the model is trained n times, each time leaving out one example for testing, and the results are averaged.

## lens
The component of an eye or camera that focuses light onto the imaging surface, allowing objects at varying distances to be brought into focus.

## level cost
In a planning graph, an estimate of the cost of reaching a goal literal based on the level at which it first appears.

## level of abstraction
The granularity at which a problem or representation is described, from very detailed low-level descriptions to coarse high-level ones.

## level sum
A planning-graph heuristic that estimates the cost of a conjunctive goal as the sum of the levels at which each of its constituent literals first appears.

## leveled off
The property of a planning graph when successive levels are identical, meaning no new literals, actions, or mutex relations appear and the graph has reached a fixed point.

## lexical category
A class of words that share syntactic behavior, such as noun, verb, adjective, or preposition; also called a part of speech.

## lexicon
The list of words in a language together with information about their categories and properties, used by a grammar.

## LIFO queue
A last-in, first-out queue (a stack), in which the most recently added element is removed first; used to implement depth-first search.

## lifting lemma
The result in first-order theorem proving guaranteeing that any proof using ground (variable-free) sentences can be reproduced at the level of general first-order sentences via unification.

## likelihood
The probability that a model assigns to the observed data, viewed as a function of the model's parameters.

## likelihood weighting
An approximate inference method for Bayesian networks that samples only the non-evidence variables while weighting each sample by the probability of the evidence given its sampled parents.

## line search
A procedure in continuous optimization that, given a search direction, finds a good step size by searching along that direction.

## linear Gaussian
A distribution in which a variable is normally distributed with a mean that is a linear function of its parents and a fixed variance, used in continuous Bayesian networks.

## linear interpolation smoothing
A language-model smoothing technique that estimates n-gram probabilities as a weighted combination of unigram, bigram, trigram, and higher-order estimates.

## linear programming
A method for optimizing a linear objective function subject to linear equality and inequality constraints over continuous variables.

## linear regression
A supervised learning method that fits a linear function of the input variables to predict a continuous output, typically by minimizing squared error.

## linear resolution
A resolution strategy in which each resolution step uses the most recently derived clause together with another clause, forming a linear chain of inferences.

## linear separator
A hyperplane that divides the input space into two regions, used to classify examples into two classes when the data are linearly separable.

## linkage constraints
Constraints that describe how the rigid parts (links) of a robot are connected, restricting their relative positions and motions.

## links
The rigid segments of a robot connected by joints, or, in a network, the connections between nodes.

## liquid event
A type of event, such as a continuous process, that has the property of being true over any subinterval of the time it holds, in contrast to events that occur as discrete wholes.

## Lisp
A family of programming languages built around symbolic list processing, historically prominent in artificial intelligence research.

## literal
An atomic sentence or its negation; the basic building block of clauses in propositional and first-order logic.

## local consistency
A property of a constraint satisfaction problem in which subsets of variables (single variables, pairs, or larger groups) satisfy their constraints, enforced by algorithms that prune inconsistent values.

## local search
A search technique that keeps only one or a few current states and moves to neighboring states, seeking to improve them without retaining the path taken.

## locality
The property that a change or interaction affects only a small, nearby part of a representation or system rather than everything at once.

## locality-sensitive hash
A hashing scheme designed so that similar inputs are likely to hash to the same bucket, enabling efficient approximate nearest-neighbor search.

## localization
The problem of determining where a robot or agent is located within its environment, given a map and a stream of sensor observations.

## locally structured
Describing a system, such as a Bayesian network, in which each component interacts directly with only a bounded number of others rather than with all of them; also called sparse.

## locally weighted regression
An instance-based regression method that, to make a prediction at a query point, fits a regression to nearby training examples weighted by their distance from the query.

## location sensors
Sensors that report a robot's position, such as GPS receivers or beacons, as opposed to sensors that measure motion or the surroundings.

## locking
A concurrency mechanism that prevents multiple processes from simultaneously modifying shared data, ensuring consistency.

## log likelihood
The logarithm of the likelihood, used because it turns products of probabilities into sums and is often more convenient to maximize.

## logic
A formal system consisting of a syntax defining well-formed sentences and a semantics defining their truth in possible worlds, providing a basis for sound reasoning.

## logical equivalence
The relation between two sentences that are true in exactly the same set of models.

## logical minimization
The process of finding the simplest logical expression that is equivalent to a given one, for example by reducing the number of terms.

## logical omniscience
The idealized assumption that a reasoning agent instantly knows all the logical consequences of everything it believes.

## logicist
Pertaining to the approach to AI that builds intelligent systems on formal logic, representing knowledge in logical sentences and acting by logical inference.

## logistic function
The S-shaped function 1/(1 + e^(-x)) that maps any real number to the interval (0, 1), widely used as a soft threshold in classification and neural networks.

## logistic regression
A classification method that models the probability of a class as the logistic function applied to a linear combination of the input features, fitting the weights to the data.

## long-distance dependencies
Grammatical relationships between words that can be separated by arbitrarily much intervening text, such as the link between a question word and the gap it fills.

## LOOCV
An abbreviation for leave-one-out cross-validation, in which each of the n examples in turn serves as the single test case while the rest are used for training.

## loopy path
A path in a search that revisits a state already on the path, forming a cycle.

## loosely coupled
Describing a system whose components interact through limited, well-defined connections, so that each can operate largely independently of the others.

## loss function
A function measuring the cost of a prediction, giving the amount of utility lost when a learner outputs one value while the true value is another; learning aims to minimize expected loss.

## lottery
In utility theory, a choice with uncertain outcomes, specified by the possible outcomes and their probabilities, used to define an agent's preferences.

## low-dimensional embedding
A representation that maps high-dimensional data into a space with far fewer dimensions while preserving important structure such as distances or neighborhoods.

# M

## machine reading
An approach to information extraction in which a system reads and learns from ordinary text without human supervision, building its own knowledge base without hand-labeled training examples.

## macrops
A macro-operator: a plan-library action assembled from a sequence of primitive actions, treated as a single higher-level step so that planning can reason at a coarser granularity.

## magic set
A logic-programming rewriting technique that transforms a set of rules so that bottom-up (forward-chaining) evaluation only derives facts relevant to a given query, combining the efficiency of forward chaining with the goal-directedness of backward chaining.

## Mahalanobis distance
A distance measure between a point and a distribution that scales each dimension by the covariance, so that correlated or high-variance directions contribute less; it generalizes Euclidean distance to account for the spread and correlation of the data.

## Maintaining Arc Consistency (MAC)
A constraint-satisfaction algorithm that, after each variable assignment during backtracking search, propagates arc-consistency constraints (via a routine like AC-3) to the remaining unassigned variables to prune inconsistent values early.

## makespan
In scheduling, the total elapsed time from the start of the first action to the completion of the last action; a common quantity to minimize when actions may run in parallel.

## margin
In a support vector machine, the distance between the separating decision boundary and the nearest training examples; the classifier is chosen to maximize this margin for better generalization.

## marginalization
The operation of summing (or integrating) a joint probability distribution over the values of some variables to obtain the distribution over the remaining variables.

## Markov blanket
The set of nodes composed of a node's parents, its children, and its children's other parents; a node is conditionally independent of all other nodes in a Bayesian network given its Markov blanket.

## Markov chain
A sequence of random variables representing states over time in which the distribution of each state depends only on the immediately preceding state, satisfying the Markov property.

## Markov decision process
A sequential decision problem for a fully observable, stochastic environment defined by a set of states, a set of actions, a Markovian transition model, and a reward function; its solution is a policy specifying an action for each state.

## Markov localization
A probabilistic robot-localization method that maintains a belief distribution over possible poses and updates it with each motion and sensor observation using Bayesian filtering.

## Markov network
An undirected graphical model in which nodes represent random variables and undirected edges encode conditional independence, with the joint distribution factored as a product of potentials over cliques; also called a Markov random field.

## Markov property
The assumption that the future is independent of the past given the present—that is, the current state depends only on a finite fixed number of previous states (usually just the one immediately before).

## material value
In game-playing evaluation, the component of a position's estimated worth that comes from counting and weighting the pieces or resources each side possesses.

## materialism
The philosophical view that the mind and mental states arise entirely from physical processes in the brain, leaving no need for a separate nonphysical substance.

## matrix
A rectangular array of numbers arranged in rows and columns, used to represent linear transformations, systems of equations, and parameter sets in machine learning.

## max norm
A vector or function norm equal to the largest absolute value among its components; used, for example, to measure the maximum change between successive utility estimates in value iteration.

## max-level
In a planning graph, the level at which a given literal or goal first appears without being mutually exclusive with itself; used as the basis for the max-level admissible heuristic.

## maximin
A decision criterion that chooses the action whose worst-case outcome is best, i.e., the action maximizing the minimum guaranteed payoff.

## maximin equilibrium
A strategy profile in a game in which each player adopts the strategy that maximizes their minimum guaranteed payoff; in zero-sum games this coincides with the value of the game.

## maximum a posteriori
A point-estimate method (MAP) that selects the hypothesis or parameter value with the highest posterior probability given the data, combining the likelihood with the prior.

## maximum expected utility
The principle (MEU) that a rational agent should choose the action that maximizes its expected utility, averaging outcome utilities weighted by their probabilities.

## maximum-likelihood
A parameter-estimation method that chooses the hypothesis or parameter values under which the observed data are most probable, assuming a uniform prior.

## mechanism
In game theory, the rules of a multiagent interaction—including the allowed actions and how outcomes and payoffs are determined from the agents' choices.

## mechanism design
The problem of designing the rules of a game or interaction so that self-interested agents, acting in their own interest, produce a desired overall outcome.

## mel frequency cepstral coefficient (MFCC)
A compact set of features derived from the short-term power spectrum of a sound on the perceptually motivated mel frequency scale, widely used as input for speech recognition.

## memoization
An optimization technique that caches the results of function calls so that repeated calls with the same arguments return the stored result instead of recomputing it.

## memoized
Describing a function or procedure whose previously computed results have been stored (memoized) so they can be reused rather than recalculated.

## mental states
Internal conditions such as beliefs, desires, and intentions attributed to an agent to explain and predict its behavior.

## mereology
The formal study of the part-whole relationship, used in knowledge representation to reason about how objects are composed of their parts.

## metadata
Data that describes other data, such as annotations, source information, or structural tags accompanying a document or dataset.

## metalevel learning
Learning that improves an agent's own reasoning or search process, for example by learning which computations are worthwhile so that future problem solving is more efficient.

## metalevel state space
The space in which each state captures the internal computational state of a program that is searching an object-level problem space; used to reason about and control the search process itself.

## metaphor
A figure of speech in which a phrase describing one kind of thing is applied to another to suggest a resemblance; a challenge for natural language understanding because meaning is non-literal.

## metareasoning
Reasoning about which computations or reasoning steps to perform, allowing an agent to allocate its limited deliberation effort where it is most valuable.

## metonymy
A figure of speech in which one entity is used to refer to a related one (for example, using an author's name to mean their works), requiring interpretation beyond literal reference.

## micromort
A unit of risk equal to a one-in-a-million chance of death, used to quantify and compare small risks in decision-theoretic reasoning about human preferences.

## min-conflicts
A local-search heuristic for constraint satisfaction that repeatedly reassigns the value of a conflicted variable to the value that violates the fewest constraints.

## mind-body problem
The philosophical question of how mental phenomena relate to physical processes in the body, and whether the mind is distinct from or reducible to the physical brain.

## minimax
A decision rule for two-player zero-sum games that computes each player's optimal move under the assumption that the opponent also plays optimally, maximizing one's own minimum guaranteed outcome.

## minimax decision
**minimax decision** is the optimal choice which leads MAX to the state with the highest minimax value and leads MIN to lowest minimax value.

## minimax search
A game-tree search algorithm that recursively computes minimax values, with MAX nodes choosing the maximum and MIN nodes the minimum of their children's values, to determine the optimal move.

## minimax value
The **minimax value** of a node in a game tree is the utility (for MAX) of being in the corresponding state, assuming that both players play optimally from there to the end of the game.

## minimum description length
A model-selection principle (MDL) that prefers the hypothesis minimizing the combined number of bits needed to encode the hypothesis plus the data given that hypothesis, formalizing a preference for simplicity.

## minimum slack
A scheduling heuristic that, at each step, selects the action whose latest possible start time is soonest (least slack) so as to reduce scheduling risk.

## minimum-remaining-values
A constraint-satisfaction variable-ordering heuristic (MRV) that selects next the unassigned variable with the fewest legal values remaining, so failures are detected sooner.

## Minkowski distance
A parameterized family of distance metrics; for parameter p it is the p-th root of the sum of the p-th powers of the coordinate differences, generalizing the Manhattan (p=1) and Euclidean (p=2) distances.

## missing precondition
In learning action models or diagnosing plan failures, a precondition of an action that is required for success but is absent from the current model or the current state.

## missing state variable
A relevant feature of the environment that is not represented in an agent's model, potentially causing incorrect predictions or decisions.

## mixture distribution
A probability distribution formed as a weighted combination of several component distributions, with the weights summing to one.

## mixture of Gaussians
A mixture distribution whose components are Gaussian, commonly used for density estimation and clustering, with parameters typically learned via the EM algorithm.

## mobile manipulator
A robot that combines a movable base with one or more articulated arms, enabling it to both navigate its environment and manipulate objects.

## modal logic
An extension of first-order logic with modal operators for expressing notions such as possibility, necessity, knowledge, or belief, allowing statements about the truth of propositions in different situations.

## model
In logic, a possible world (an assignment of values to symbols) that determines the truth of every sentence; more generally, a formal or statistical representation of some aspect of the world.

## model checking
A method for determining whether a sentence is entailed by a knowledge base by enumerating the possible models and checking that the sentence is true in every model in which the knowledge base is true.

## model selection
The process of choosing among candidate model classes or hypothesis complexities (for example, polynomial degree), typically balancing fit to the data against complexity to avoid overfitting.

## modus ponens
A sound rule of inference stating that from an implication and its antecedent one may derive the consequent: given "P implies Q" and "P," conclude "Q."

## monitoring
The task of continuously tracking the state of a dynamic system over time, such as computing the current belief state from a stream of observations; also called filtering.

## monotonic preference
A preference relation in which increasing the amount of a desirable quantity (such as money) never makes an outcome less preferred.

## monotonicity
A property of a logic in which the set of entailed sentences can only grow as new facts are added, so previously drawn conclusions are never retracted; also a required condition (consistency) on admissible heuristics.

## Monte Carlo
Referring to algorithms that use repeated random sampling to obtain approximate numerical results, especially for estimating quantities that are hard to compute exactly.

## Monte Carlo localization
A robot-localization method (MCL) that represents the belief over the robot's pose with a set of weighted particles and updates them with a particle filter as the robot moves and senses.

## Monte Carlo simulation
An estimation technique that runs many randomized simulations of a process and aggregates their outcomes to approximate probabilities or expected values.

## Monte Carlo tree search
A best-first search method (MCTS) that estimates the value of moves by running many random simulations (playouts) from candidate states, selectively growing a search tree toward the most promising actions.

## motion blur
An image artifact in which moving objects appear smeared because motion occurs during the camera's exposure interval; it also provides a cue for estimating object motion.

## motion model
A probabilistic model describing how an agent's or object's state changes as a result of its actions or dynamics, used in filtering and control.

## multiactor
Describing a setting or system involving more than one acting entity; often used to distinguish environments with multiple physical actors from single-agent settings.

## multiagent
Involving more than one agent, where each agent's decisions may affect and be affected by the others.

## multiagent planning problem
A planning problem in which several agents must coordinate their actions, requiring reasoning about others' plans, cooperation, or competition.

## multiagent systems
Systems composed of multiple interacting agents that may cooperate, compete, or coordinate to achieve individual or shared goals.

## multiplexer
A logic or circuit element that selects one of several input signals to pass to its output based on the value of control (selector) inputs.

## multiplicative utility function
A utility function over multiple attributes expressed as a product of single-attribute terms, valid under mutual utility independence of the attributes.

## multiply connected
Describing a graph, such as a Bayesian network, in which there is more than one undirected path between some pair of nodes (i.e., it contains a loop), making exact inference harder than in singly connected networks.

## multivariate Gaussian
A generalization of the normal distribution to several correlated variables, parameterized by a mean vector and a covariance matrix.

## multivariate linear regression
A regression model that predicts a scalar output as a linear function of several input features, fitting a weight for each feature plus an intercept.

## mutation
In genetic algorithms, a variation operator that randomly alters one or more components of an individual's representation to introduce new genetic diversity.

## mutex
A mutual-exclusion relation, used in planning graphs to indicate that two actions or two literals cannot both occur or hold at the same level.

## mutual preferential independence
A condition (MPI) under which each subset of attributes' preference structure is independent of the levels of the other attributes, justifying an additive value function.

## mutually utility independent
A condition (MUI) on a set of attributes under which each attribute's utility contribution is independent of the others, justifying a multiplicative utility function.

## myopic
Describing a decision strategy that considers only the immediate next step or single observation, ignoring longer-term consequences; for example, myopic value-of-information computations.
# N

## n-armed bandit
A decision problem, also called a multi-armed bandit, in which an agent must repeatedly choose among n actions (the "arms"), each yielding rewards drawn from an unknown distribution. It captures the exploration-exploitation tradeoff: balancing gathering information about the arms against exploiting the arm believed to be best.

## n-gram model
A language model that estimates the probability of a word conditioned on the previous n-1 words, treating text as a sequence generated by an order-(n-1) Markov process. Common cases are the unigram, bigram, and trigram models.

## natural kind
A category of things in the world (such as tomatoes or dogs) that shares many properties but resists a precise, closed definition in terms of necessary and sufficient conditions, in contrast to rigidly defined mathematical categories.

## natural numbers
The non-negative integers 0, 1, 2, ..., which can be defined logically using a constant such as 0 and a successor function, allowing arithmetic to be axiomatized in first-order logic (e.g., using Peano axioms).

## nearest-neighbor filter

The nearest-neighbour filter, which repeatedly chooses the closest pairing of predicted position and observation and adds that pairing to the assignment.

## nearest-neighbors regression
A nonparametric regression method that predicts the output for a query point by averaging (or otherwise combining) the outputs of its k nearest training examples in the input space.

## negation
A logical connective (written not or ¬) that reverses the truth value of a sentence, so that ¬P is true exactly when P is false.

## negative
In classification, an example belonging to the class the learner is trying to distinguish against, i.e., one that does not have the target property; the complement of a positive example.

## neuroscience
The scientific study of the nervous system, especially the brain, which informs AI by revealing how biological systems process information with networks of neurons.

## Newton-Raphson
An iterative numerical method for finding roots or optima of a function that uses the function's derivative (and, for optimization, its second derivative or Hessian) to update the current estimate, converging quadratically near a solution.

## no-good
In constraint satisfaction and clause learning, a recorded partial assignment of variables that has been shown to be inconsistent, so that it can be avoided in the rest of the search.

## no-regret learning
Online learning behavior in which an agent's average per-step performance over time approaches that of the best fixed strategy in hindsight, so the average regret tends to zero.

## noise
Irrelevant or erroneous variation in data—such as measurement errors, mislabeled examples, or unmodeled influences—that can obscure the true underlying relationship and lead to overfitting if a learner tries to fit it exactly.

## noisy channel model
A probabilistic model, used in tasks like spelling correction and machine translation, that treats an observed signal as a correct message passed through a channel that corrupts it, and recovers the intended message by combining a prior over messages with a model of the channel's distortions.

## noisy-OR
A compact conditional probability representation for a Boolean variable with several possible causes, in which each cause independently succeeds in producing the effect unless it is inhibited; the effect is false only if every present cause is independently inhibited.

## nondeterministic
Describing an environment or action whose outcome is not uniquely determined by the current state and action, so that a given action may lead to any of several possible resulting states.

## nonholonomic
Describing a system, such as a car, whose controllable degrees of freedom are fewer than its total degrees of freedom, so that it cannot move arbitrarily in configuration space (e.g., it cannot slide sideways) and its motion is subject to differential constraints.

## nonlinear
Describing a function or model whose output is not a linear (weighted sum plus constant) function of its inputs, so that superposition does not hold.

## nonlinear regression
Regression in which the hypothesis relating inputs to outputs is a nonlinear function of the parameters or inputs, fit by minimizing a loss such as squared error.

## nonmonotonicity
A property of a logic in which adding new premises can invalidate previously derivable conclusions, unlike classical logic where the set of entailed conclusions only grows as premises are added.

## nonparametric
Describing a model whose complexity is not fixed by a bounded set of parameters but is allowed to grow with the amount of training data, so the data itself is effectively retained to make predictions.

## nonparametric density estimation
Estimating a probability density from data without assuming a fixed parametric form, for example by kernel density estimation or k-nearest-neighbor methods that adapt to the observed samples.

## nonparametric model
A model that does not summarize the data with a fixed number of parameters but instead retains the training examples (or a data-dependent number of parameters), such as k-nearest-neighbors, with representational capacity that grows with the data.

## normalization
The process of scaling a set of numbers so they satisfy a constraint, most commonly dividing the entries of a probability distribution by their sum so that they add to 1.

## normalized form
A standardized syntactic form to which sentences are converted to simplify processing, such as conjunctive normal form for resolution or a canonical form for probabilistic or grammatical representations.

## normative theory
A theory that prescribes how an ideal rational agent should behave (for example, decision theory as a standard of rationality), as opposed to a descriptive theory of how humans actually behave.

## NP-complete
The class of the hardest problems in NP: decision problems in NP to which every other NP problem can be reduced in polynomial time, so that a polynomial-time algorithm for any one would solve all of NP efficiently.

## NP-completeness
The property of belonging to the class of NP-complete problems; establishing NP-completeness for a problem provides strong evidence that no efficient (polynomial-time) algorithm exists for it in the worst case.

## null hypothesis
In statistical hypothesis testing, the default hypothesis (typically that an observed effect is due to chance) that a test attempts to reject in favor of an alternative when the data are sufficiently unlikely under it.

# O

## object model
A structured representation of the objects in a domain and their properties and relationships, used to give meaning to the symbols a knowledge-based agent manipulates.

## objective function
The function that an optimization or search procedure seeks to maximize or minimize, such as a utility to be maximized or a loss/cost to be minimized.

## objectivist
Pertaining to the view that probabilities are objective features of the world—long-run frequencies or physical propensities—rather than degrees of belief held by an agent; contrasted with the subjectivist (Bayesian) view.

## occupancy grid
A representation of an environment as a grid of cells, each storing the probability (or a binary estimate) that the corresponding region of space is occupied by an obstacle, commonly built from robot sensor data.

## occupied space
The portion of a robot's configuration space or workspace that is filled by obstacles and therefore cannot be entered, as opposed to free space.

## occur check
A test performed during unification that prevents binding a variable to a term that contains that same variable, which would otherwise create an infinite (cyclic) structure; omitting it makes unification faster but unsound.

## Ockham's razor
The principle that, among competing hypotheses that fit the data equally well, one should prefer the simplest; in machine learning it motivates preferring simpler models to avoid overfitting.

## odometry
Estimation of a robot's change in position and orientation over time by integrating measurements of wheel rotation or similar internal motion sensors; it accumulates error and drifts over long distances.

## off-policy
Describing a reinforcement learning method that learns the value of a target policy (often the optimal one) while the agent's behavior is governed by a different policy used to generate experience, as in Q-learning.

## omniscience
Complete knowledge of the actual outcome of every action; a distinction is drawn between rationality, which requires making the best decision given available information, and omniscience, which no real agent can achieve.

## on-policy
Describing a reinforcement learning method that evaluates and improves the same policy that is being used to select actions and generate experience, as in SARSA.

## online replanning
Interleaving execution with planning so that the agent revises its plan during execution in response to new observations or unexpected outcomes, rather than committing to a complete plan in advance.

## online search
Search in which an agent must interleave computation with action, discovering the search space by actually taking actions and observing their results, rather than exploring a fully known model offline.

## ontological commitment
What a representation language assumes exists in the world—for example, propositional logic commits only to facts, while first-order logic additionally commits to objects and relations among them.

## ontological engineering
The activity of designing and building general-purpose representations of categories, objects, events, time, and other broad aspects of the world, so that many domains can be described within a shared framework.

## ontology
An organized, formal representation of the concepts and relationships in a domain (or in the world generally), providing the vocabulary and structure that knowledge in that domain is expressed in.

## open list
In graph search, the collection of nodes that have been generated but not yet expanded—the frontier from which the next node to explore is selected.

## open-code
To implement a construct by inserting its code inline at each point of use (for example, expanding a routine in place) rather than invoking it through a general mechanism, often to gain efficiency.

## open-loop
Describing control or plan execution that carries out a predetermined sequence of actions without using feedback from observations to adjust to the actual results.

## operationality
The property of a learned or derived concept description being expressed in terms that the agent can efficiently evaluate and use directly during problem solving, a key criterion in explanation-based learning.

## operations research
A field that applies mathematical modeling and optimization to decision-making problems such as scheduling, routing, and resource allocation, sharing many techniques with AI.

## optimal brain damage
A neural network pruning technique that removes weights judged to have the least impact on the training error, using second-derivative (curvature) information, to produce a smaller network that generalizes better.

## optimal controllers
Controllers derived by optimizing a cost functional over the trajectories of a dynamical system; in the linear-quadratic case these yield closed-form optimal control laws.

## optimally efficient
Describing a search algorithm that expands no more nodes than necessary—no other algorithm using the same heuristic information is guaranteed to expand fewer—as A* is among algorithms using a given admissible heuristic.

## optimization
The problem of finding an input that maximizes or minimizes an objective function, possibly subject to constraints, using methods such as gradient descent, linear programming, or local search.

## optimizer's curse
The tendency for the estimated value of the option that appears best to be systematically too optimistic, because selecting the maximum of noisy estimates favors options whose estimation errors happen to be positive.

## optogenetics
An experimental neuroscience technique that uses light to control neurons that have been genetically modified to be light-sensitive, allowing precise activation or silencing of specific cells.

## ordering constraints
In partial-order planning, constraints specifying that one action must occur before another, without fully committing to a total sequence of actions.

## OR-parallelism
A form of parallelism in logic programming in which the different clauses that could satisfy a goal (the alternative ways to prove it) are explored simultaneously.

## orientation
The direction a robot or object is facing, part of its pose along with position; in configuration space it is represented by angular degrees of freedom.

## origin function
In probabilistic models with objects of uncertain provenance, a function that maps an observed percept or track to the object in the world from which it originated, used in data association.

## Othello
A two-player, deterministic, fully observable board game (also known as Reversi) that has served as a test bed for game-playing programs, in which computer players surpassed the best humans.

## out of vocabulary
Referring to a word or token encountered at test time that did not appear in the training vocabulary of a language model, often handled by a special unknown-word symbol.

## outcome
A specific resulting state or event that may follow from an action or a random process; in decision theory, agents assign utilities to outcomes and choose actions to maximize expected utility.

## overall intensity
In computer vision, the total brightness recorded at an image location, resulting from the combination of scene illumination, surface reflectance, and geometry.

## overfitting
The phenomenon in which a model fits the training data too closely—including its noise and idiosyncrasies—so that it captures spurious patterns and generalizes poorly to new data.

# P

## PAC learning

A framework in computational learning theory in which a learning algorithm is judged by whether it can, with high probability, return a hypothesis whose error is small. An algorithm is said to be "probably approximately correct" if it achieves this using a number of examples that is polynomial in the relevant parameters.

## PageRank

An algorithm for ranking web pages by importance, based on the idea that a page is important if many important pages link to it. It models a random surfer who follows links, and the PageRank of a page corresponds to the long-run probability that the surfer lands on it.

## parameter independence

An assumption in Bayesian learning that the parameters governing different conditional probability distributions in a model are independent of one another, allowing the joint prior over parameters to factor into a product of separate priors.

## parameter learning

The task of estimating the numerical values (such as probabilities or weights) of a model whose structure is already fixed, typically from a set of training data.

## parametric model

A statistical model that summarizes data using a fixed, finite number of parameters, so that the amount of storage does not grow with the size of the training set. This contrasts with nonparametric models.

## Pareto dominated

An outcome is Pareto dominated by another outcome if the second is at least as good for every player and strictly better for at least one player. Such an outcome is generally undesirable because everyone could be made better off.

## parse tree

A tree structure that represents the syntactic derivation of a string according to a grammar, with the root labeled by the start symbol, internal nodes by nonterminal categories, and leaves by the words of the sentence.

## parsing

The process of analyzing a string of words to determine its grammatical structure with respect to a given grammar, typically producing one or more parse trees.

## partial assignment

An assignment of values to some, but not necessarily all, of the variables in a constraint satisfaction problem.

## partial information

A situation in which an agent does not have complete knowledge of the state of the world, either because it cannot observe everything or because the environment is uncertain.

## partial program

An agent program that leaves some choices unspecified, so that a learning process (such as reinforcement learning) can fill in the remaining decisions to complete the behavior.

## partially observable

A property of an environment in which the agent's sensors do not give it access to the complete state of the world, so it must maintain a belief about the possible states it might be in.

## particle filtering

An approximate inference algorithm for temporal probabilistic models that represents the belief state by a set of sampled states (particles), which are propagated forward, weighted by their likelihood given new evidence, and resampled.

## partition

A division of a set into disjoint, exhaustive subsets, so that every element belongs to exactly one subset.

## passive learning

A passive learning agent learns from its observations, but the actions the agent takes are not influenced by the learning process.
This is in contrast to an active learning agent, which chooses actions that will facilitate its own learning.

## path

A sequence of states connected by a sequence of actions, typically representing a route from an initial state through a state space.

## path planning

The problem of finding a collision-free route for a robot from a start configuration to a goal configuration through its environment.

## paths

A sequence of states connected by actions in a state space; the plural is used when discussing the collection of possible routes considered during search.

## pattern matching

The process of comparing a data structure against a pattern (which may contain variables) to determine whether it matches, and if so, to bind the variables to corresponding parts of the data.

## payoff function

A function in game theory that gives the utility a player receives for each possible combination of strategies chosen by all the players.

## PD controller

A proportional-derivative controller that generates a control signal from a weighted sum of the current error and the rate of change of that error, using the derivative term to dampen oscillations.

## PDDL

The Planning Domain Definition Language, a standardized formalism for describing planning problems in terms of states, actions with preconditions and effects, an initial state, and a goal.

## Peano axioms

A set of axioms that formally define the natural numbers and the operations on them, using a successor function together with the principle of induction.

## PEAS

A framework for specifying the design of an agent's task environment in terms of its Performance measure, Environment, Actuators, and Sensors.

## peeking

An error in evaluating a learning algorithm in which information from the test set inadvertently influences the training or model-selection process, producing overly optimistic performance estimates.

## percept

The term **percept** refers to the agent's perceptual inputs at any given instant.

## percept schema

In planning under partial observability, a description that specifies what an agent will perceive as a function of the state, analogous to an action schema but for sensing.

## percept sequence

An agent's **percept sequence** is the complete history of everything the agent has ever perceived.

## perception

The process by which an agent interprets its sensory inputs to build a representation of the state of the world.

## perception layer

In a layered agent or robot architecture, the component responsible for processing raw sensor data into a usable interpretation of the environment.

## perceptron

A single-layer feedforward network consisting of a linear threshold unit, which computes a weighted sum of its inputs and outputs a value based on whether the sum exceeds a threshold. It can represent only linearly separable functions.

## perceptron network

A network composed of perceptron units arranged in a single layer, with each output computed independently from the same set of inputs.

## perfect rationality

The idealized standard of always taking the action that maximizes expected utility, given all available information and without any limits on computation.

## performance element

The component of a learning agent that selects external actions, corresponding to what would otherwise be considered the entire agent; the learning element works to improve it.

## perplexity

A measure of how well a probabilistic language model predicts a sample, defined so that a lower perplexity indicates a better model; it can be interpreted as the effective average branching factor of the model's predictions.

## persistence arc

In a dynamic Bayesian network, a link connecting a variable at one time step to the same variable at the next time step, representing how a state feature tends to persist over time.

## persistent failure model

A model of sensor or component failure in which, once a failure occurs, it continues over time, rather than being transient.

## perspective projection

A geometric projection that maps points in a three-dimensional scene onto an image plane through a single center of projection, so that objects farther away appear smaller.

## phone model

In speech recognition, a probabilistic model describing the sequence of acoustic feature vectors produced by a single phone (speech sound).

## phoneme

The smallest unit of sound in a language that can distinguish one word from another; it is an abstract category that may be realized by several acoustically distinct sounds.

## phrase structure

The hierarchical organization of a sentence into constituent phrases (such as noun phrases and verb phrases), typically described by a phrase structure grammar.

## physical symbol system

A hypothesis, due to Newell and Simon, that a system manipulating physical patterns as symbols has the necessary and sufficient means for general intelligent action.

## physicalism

The philosophical position that everything, including mental states, is ultimately physical and can be explained in terms of physical processes.

## piano movers

A classic motion-planning problem concerned with finding a way to move a large rigid object (such as a piano) through a cluttered environment without collisions.

## pictorial structure model

A model for object recognition that represents an object as a collection of parts arranged in a deformable configuration, with costs for both part appearance and the geometric relationships between parts.

## PID controller

A proportional-integral-derivative controller that computes its control signal from a weighted sum of the current error, the accumulated (integral) error, and the rate of change of the error.

## plan monitoring

The process by which an agent checks, during execution of a plan, whether the world is unfolding as expected and whether the plan's remaining steps are still applicable.

## plan recognition

The task of inferring an agent's goals and plans from observations of its actions.

## planning graph

A layered data structure, alternating between levels of literals and levels of actions, that provides useful information and heuristic estimates for planning problems, notably as used in the Graphplan algorithm.

## PlanSAT

The decision problem of determining whether a given planning problem has any solution at all.

## playout

In Monte Carlo tree search, a simulated game played out from a given state, often using random or lightweight policies, whose outcome is used to estimate the value of that state.

## ply

One move by a single player in a game; a full turn by both players in a two-player game corresponds to two ply.

## pneumatic actuation

The use of compressed gas or air to drive a robot's mechanical movements.

## point-to-point motion

A motion-control task in which the goal is only that the end effector reach a specified target configuration, without constraints on the path taken between points.

## poker

A card game of imperfect information often used as a testbed in AI, notable for requiring reasoning about hidden cards, probability, and opponent modeling including bluffing.

## policy

A specification of what action an agent should take in each possible state; in sequential decision problems it is often denoted by the mapping from states to actions.

## policy evaluation

The step in solving a Markov decision process that computes the utility (value) of each state under a fixed policy.

## policy gradient

A reinforcement learning approach that directly adjusts the parameters of a policy by following the gradient of expected return with respect to those parameters.

## policy improvement

The step in policy iteration that computes a new, better policy by acting greedily with respect to the utilities calculated for the current policy.

## policy iteration

An algorithm for solving Markov decision processes that alternates between policy evaluation and policy improvement until the policy no longer changes.

## policy loss

The difference between the value obtained by an agent's chosen policy and the value of the optimal policy; it measures how much utility is lost by not acting optimally.

## policy search

A class of reinforcement learning methods that search directly in the space of policies to find one that maximizes expected performance, rather than first estimating value functions.

## policy value

The expected utility obtained by following a particular policy starting from a given state (or set of states).

## polynomial kernel

A kernel function used in support vector machines that computes similarity as a polynomial of the dot product of two input vectors, implicitly mapping inputs into a higher-dimensional feature space.

## pose

The combined position and orientation of an object or robot in space.

## positive

In the context of learning, an example that belongs to the target category (a positive example), as opposed to a negative example that does not.

## possibility axiom

In situation calculus, an axiom stating the conditions under which a given action is possible to perform in a situation.

## possibility theory

A formalism for reasoning about uncertainty, using degrees of possibility and necessity rather than probabilities, often as an alternative treatment of vagueness and incomplete information.

## possible world

One of the complete ways the world could be; in logic and probability, propositions are true or false and probabilities are assigned relative to the set of possible worlds.

## post-decision disappointment

The tendency for the realized value of a chosen option to fall short of its estimated value, because the selection process favors options whose values happen to be overestimated (a form of the winner's curse).

## pragmatics

The study of how context contributes to meaning in language, including how the situation and speaker intentions affect interpretation of an utterance.

## precedence constraints

In scheduling, constraints specifying that certain actions or tasks must be completed before others can begin.

## precision

**Precision** is a performance measure often used to describe some model, alongside other measures like *accuracy*, *recall* etc. Precision can be thought of as an efficiency measure of a model. It is given as:  
***Precision = True Positives / (True Positives + False Positives)***

## precondition

A condition that must hold in a state for an action to be applicable in that state.

## prediction

The task of computing the probability distribution over a future state given evidence up to the present; more generally, estimating an unknown outcome from a model.

## preference elicitation

The process of determining an agent's or user's utility function by asking questions or observing choices.

## preference independence

A condition in multiattribute utility theory in which the preference between outcomes over some attributes does not depend on the fixed values of the remaining attributes.

## prefix

An initial segment of a sequence; in language and coding, a string that forms the beginning of a longer string.

## premise

A statement offered as support in an argument or an antecedent of an implication, from which a conclusion may be drawn.

## presentation

In speech and language processing, the surface form in which information is delivered; more generally, how content is rendered for a receiver.

## principle of indifference

The rule that, in the absence of any distinguishing information, equal probabilities should be assigned to possibilities that are symmetric or otherwise indistinguishable. Also known as the principle of insufficient reason.

## principle of insufficient reason

The rule, also called the principle of indifference, that propositions should be assigned equal probability if there is no reason to favor one over another.

## principle of trichromacy

The observation that human color perception is based on three types of color receptors, so that any perceived color can be matched by a combination of three primary colors.

## prioritized sweeping

A reinforcement learning technique that focuses updates on states whose estimated values are likely to change substantially, using a priority queue to propagate large changes efficiently.

## priority queue

A data structure that maintains a collection of elements each with an associated priority and supports efficient removal of the element with the best (highest or lowest) priority; used, for example, to order nodes in best-first search.

## prisoner's dilemma

A classic two-player game in which each player is individually better off defecting regardless of the other's choice, yet mutual cooperation would leave both better off than mutual defection, illustrating the tension between individual and collective rationality.

## probabilistic checkmate

In a game with chance elements, a strategy that guarantees a win with probability 1 (though not necessarily in a bounded number of moves) against any opponent play.

## probabilistic Horn abduction

A framework combining logic-programming-style Horn clauses with probabilities to perform abductive reasoning, inferring likely explanations for observations.

## probabilistic inference

The computation of posterior probabilities for query variables given observed evidence, using a probabilistic model such as a Bayesian network.

## probability

A number between 0 and 1 that quantifies the degree of belief in, or the relative frequency of, a proposition or event.

## probability density function

A function describing the relative likelihood of a continuous random variable taking each possible value, such that its integral over any interval gives the probability of the variable falling in that interval.

## probability distribution

A specification of the probabilities assigned to each possible value of a random variable (or combination of values for several variables).

## probability model

A formal description of a set of random variables together with the probabilities of their possible values, defining a joint distribution over all outcomes.

## probit distribution

A distribution based on the cumulative normal (Gaussian) function, sometimes used in place of the logistic function to model the probability of a binary outcome as a function of a continuous variable.

## problem

A formal specification of a task to be solved by search, typically defined by an initial state, actions, a transition model, a goal test, and a path cost function.

## problem formulation

The process of deciding what states, actions, goals, and costs to use to represent a task as a well-defined problem that can be solved by search.

## problem-solving agent

An agent that decides what to do by finding sequences of actions that lead to desirable states, using search over an explicit problem formulation.

## procedural attachment

A technique in knowledge representation in which certain reasoning steps are carried out by calling attached procedures (code) rather than by general logical inference.

## process

In event or temporal reasoning, a category of event that occurs over an interval of time and holds over any subinterval, in contrast to a discrete event with definite endpoints.

## product rule

The rule of probability stating that the probability of a conjunction equals the probability of one event times the conditional probability of the other given the first.

## progression planning

Planning by searching forward from the initial state, applying actions to generate successor states until a goal state is reached.

## Prolog

A logic programming language based on Horn-clause resolution, in which programs are stated as facts and rules and computation proceeds by backward-chaining inference.

## pronunciation model

A model that specifies the possible sequences of phones corresponding to a word, used in speech recognition to relate words to their acoustic realizations.

## proof

A sequence of inference steps that derives a conclusion from a set of premises using sound rules of inference, demonstrating that the conclusion follows logically.

## proof-checker

A program that verifies whether a given proof is valid by checking that each step follows correctly from earlier steps and the inference rules.

## proposition symbol

An atomic symbol in propositional logic that stands for a proposition that can be either true or false.

## propositionalize

To convert a set of first-order logic sentences into an equivalent (or entailment-preserving) set of propositional sentences by instantiating quantified variables with ground terms.

## protein design

The problem of determining an amino acid sequence that will fold into a protein with a desired three-dimensional structure or function, often posed as a search or optimization problem.

## provably beneficial

A property of an AI system that can be formally shown to act in the interests of humans, a goal proposed as an alternative to simply maximizing a fixed given objective.

## pruning

The process of eliminating parts of a search space or tree from consideration without examining them, on the grounds that they cannot affect the final result; alpha-beta pruning in game trees is a prominent example.

## psychological reasoning

Reasoning about the mental states of agents, such as their beliefs, desires, and intentions, in order to understand or predict their behavior.

## PUMA

The Programmable Universal Machine for Assembly, a well-known industrial robot arm used as a standard example of a manipulator.

## pure strategy

In game theory, a strategy that deterministically selects a single action, as opposed to a mixed strategy that randomizes over actions.

## pure symbol

In the context of satisfiability solving, a propositional symbol that appears with the same sign (always positive or always negative) in all clauses in which it occurs, allowing it to be assigned to satisfy those clauses.
# Q

## Q-learning
A model-free reinforcement learning method that learns an action-utility function Q(s, a), giving the expected value of taking action a in state s. The agent updates its Q estimates from experienced transitions without needing a model of the environment's transitions or rewards.

## QALY
Quality-adjusted life year, a metric used in decision making about medical and public-policy questions. It combines the length and quality of life so that outcomes with differing health states can be compared on a common utility scale.

## quadratic programming
An optimization problem in which a quadratic objective function is minimized or maximized subject to linear constraints. It arises, for example, when training support vector machines.

## qualia
The subjective, intrinsic quality of conscious experiences, such as how the color red actually looks to an observer. Whether machines can have qualia is a central question in philosophical debates about machine consciousness.

## qualification problem
The difficulty of listing every precondition that must hold for an action to have its intended effect. Because there are countless possible exceptions, it is impractical to fully qualify the rules describing an action.

## qualitative physics
A subfield of knowledge representation that reasons about physical systems using qualitative abstractions (such as increasing, decreasing, or the sign of a quantity) rather than precise numerical equations.

## quantification
In logic, the use of quantifiers such as the universal quantifier (for all) and the existential quantifier (there exists) to express statements about collections of objects rather than individual objects.

## quantization factor
A parameter that determines how a continuous range of values is divided into discrete levels, for example when discretizing signals or representing sensor readings at limited precision.

## quasi-logical form
An intermediate representation in natural language processing that stays close to the syntactic structure of a sentence while capturing much of its meaning, deferring the resolution of ambiguities to a later stage.

## question answering
A natural language processing task in which the system produces a direct answer to a question posed in natural language, rather than returning a list of relevant documents.

## queue
A data structure used to store the frontier in search algorithms, supporting operations to test emptiness, add nodes, and remove nodes. Different orderings (FIFO, LIFO, priority) yield different search strategies.

## quiescence search
A technique in game-playing programs that extends search beyond the nominal depth limit at positions that are not "quiescent," meaning positions where a pending capture or other volatile move could dramatically change the evaluation. This mitigates the horizon effect.

# R

## radial basis function
A function whose value depends only on the distance from a center point, often a Gaussian. Radial basis functions are used as features or kernels so that a weighted combination of them can approximate arbitrary functions.

## radiometry
The study of the measurement of light and other electromagnetic radiation, providing the physical basis for modeling how much light reaches a camera sensor in computer vision.

## random surfer model
A model of web browsing underlying the PageRank algorithm, in which a hypothetical user follows links at random and occasionally jumps to a random page. A page's rank reflects the long-run probability of the surfer being there.

## random-restart hill climbing
A local search strategy that runs hill climbing repeatedly from randomly chosen initial states, keeping the best result found. The random restarts help escape local maxima that trap a single hill-climbing run.

## randomized weighted majority algorithm
An online learning algorithm that combines the predictions of a set of experts by maintaining weights, choosing an expert probabilistically according to its weight, and reducing the weights of experts that make mistakes. It has strong regret guarantees.

## rational agent
A rational agent selects an action that is expected to maximize its performance measure,given the evidence provided by the
*percept sequence* and whatever built-in knowledge the agent has.

## rationalism
The philosophical view that emphasizes reason and logical inference, rather than sensory experience, as the primary source of knowledge and the basis for intelligent behavior.

## rationality
The property of choosing actions that are expected to achieve the best outcome according to a performance measure, given what the agent knows. It concerns making good decisions, not necessarily being correct in an unpredictable world.

## reachable set
In planning and control, the set of states that an agent or system can reach from a given starting state under its available actions or control inputs.

## reactive control
A control approach in which actions are chosen as direct responses to current sensory input, without building or consulting an internal model of the world.

## reactive layer
The lowest level in a layered robot architecture, providing fast, reflex-like responses to sensor input while higher layers handle deliberative planning.

## real-time AI
The design of AI systems that must produce acceptable decisions within strict time constraints, often trading off decision quality against the time available, as in anytime algorithms.

## realizable
Describing a hypothesis space that contains the true function generating the data, so that in principle a learning algorithm could find a hypothesis with zero error.

## reasoning
The process of deriving new conclusions or decisions from existing knowledge and percepts, for example through logical inference or probabilistic computation.

## recall
**Recall** is a measure of performance used alongside ***Precision***, ***Accuracy*** and ***F-score***. It is defined as the ratio of the *true positives* to the *summation of true positives and false negatives*.

## reciprocal rank
An information retrieval metric equal to one divided by the rank position of the first relevant result. Averaging it over queries gives the mean reciprocal rank.

## recognition
The task of identifying what an object, pattern, or category is from observed data, such as recognizing an object in an image or a word in speech.

## recombine
In genetic algorithms, the operation of combining parts of two parent individuals (via crossover) to produce offspring that inherit features from both.

## reconstruction
In computer vision, the process of recovering a three-dimensional description of a scene from one or more two-dimensional images.

## record linkage
The task of determining which records, across one or more data sources, refer to the same real-world entity despite differences in how they are described.

## rectangular grid
A discretization of a continuous space into a regular array of rectangular cells, commonly used to represent maps or configuration spaces for search and planning.

## recurrent network
A neural network with connections that form cycles, allowing it to maintain an internal state and process sequences of inputs. This lets earlier inputs influence the handling of later ones, making it suited to temporal data.

## recursive
Describing a definition or procedure that refers to itself, as in recursive functions or recursively defined data structures and algorithms.

## recursive best-first search
A best-first search algorithm that runs in space linear in the depth of the solution by using recursion and tracking the best alternative value along the current path, backing up when the current path exceeds that bound.

## reduct
In default and nonmonotonic reasoning, a simplified version of a theory obtained relative to a candidate set of assumptions, used to test whether that set forms a consistent extension.

## reference class
The population or category of cases to which an individual is assigned in order to estimate a probability for that individual from frequency data. Choosing the right reference class is a well-known difficulty in applying frequentist probabilities.

## reference controller
In robotics, a controller whose goal is to keep the robot following a specified reference path or trajectory over time.

## reference path
A desired trajectory through the state or configuration space that a robot controller attempts to track.

## reflect
To reason about one's own reasoning processes; an agent that can reflect can examine and modify its own computations or knowledge.

## reflective architecture
An agent architecture that can represent and reason about its own internal state and decision-making, enabling metalevel control of deliberation.

## refutation
A proof that a set of sentences is unsatisfiable, typically by deriving a contradiction. Resolution theorem proving works by refutation: it proves a goal by showing that its negation leads to a contradiction.

## regions
In image segmentation, contiguous groups of pixels that share similar properties such as color, texture, or brightness and are treated as coherent parts of an image.

## regression
In machine learning, the task of learning a function with a continuous (numerical) output. In planning, it also refers to reasoning backward from goals to determine required preconditions.

## regression planning
A planning approach that searches backward from the goal, computing the set of states from which applying an action would achieve the goal, until it reaches the initial state.

## regression to the mean
The statistical tendency for extreme measurements to be followed by ones closer to the average, which can create misleading impressions of cause and effect.

## regret
The difference between the reward actually obtained and the reward that could have been obtained by the best available choice in hindsight. Minimizing regret is a goal in online learning and bandit problems.

## regular expression
A formal notation for describing sets of strings using patterns, widely used in text processing to match and extract sequences of characters.

## regularization
A technique that adds a penalty on model complexity to the loss function during learning, discouraging overly complex hypotheses in order to reduce overfitting and improve generalization.

## reinforcement
A reward or punishment signal that an agent receives from its environment, indicating how good or bad its situation is and driving reinforcement learning.

## reinforcement learning
In **reinforcement learning** the agent learns from a series of
reinforcements-rewards or punishments.

## rejection sampling
An approximate inference method that generates samples from a prior distribution and discards those inconsistent with the observed evidence, estimating the posterior from the samples that remain.

## relational extraction
The task of identifying relationships between entities mentioned in text, producing structured relational facts from unstructured natural language.

## relational uncertainty
Uncertainty about which objects are related to which, as opposed to uncertainty about attribute values; it arises in probabilistic models over relational domains.

## relative error
An error measure expressed as a fraction of the true or reference value, allowing errors of different magnitudes to be compared on a common scale.

## relative likelihood
The ratio of the likelihoods of two hypotheses given the same data, used to compare how well competing hypotheses explain observations.

## relaxed problem
A version of a problem with some restrictions on the actions removed, making it easier to solve. The cost of an optimal solution to a relaxed problem gives an admissible heuristic for the original problem.

## relevance
The property of a piece of information being pertinent to answering a query or solving a problem; in information retrieval, whether a document meets the user's information need.

## relevance feedback
An information retrieval technique that improves query results by incorporating a user's judgments of which returned documents are relevant, refining the query accordingly.

## relevant
Describing information, actions, or facts that bear on the current goal or query, so that ignoring them could change the outcome of reasoning.

## relevant-states
In backward search for planning, the set of states from which a given action or plan could lead to the goal, used to guide regression search.

## renaming
In logic, replacing the variables in a sentence with new ones so that two sentences share no variables, a step required before unification and resolution.

## rendering
In computer graphics, generating a two-dimensional image from a three-dimensional scene model; it is the inverse of the vision reconstruction problem.

## rendering model
A model of how a scene produces an image, describing geometry, lighting, and reflectance so that image formation can be predicted or inverted.

## repeated state
A state that is generated more than once during a search, which can cause redundant work or infinite loops if not detected and pruned.

## resolution
A sound and complete inference rule for propositional and first-order logic that, given two clauses containing complementary literals, produces a new clause (the resolvent). Refutation-based resolution proves a sentence by deriving the empty clause from the negated goal.

## resolvent
The new clause produced by applying the resolution rule to two clauses, formed by combining their literals and removing a complementary pair.

## result set
In information retrieval, the collection of documents returned by the system in response to a query.

## Rete algorithm
An efficient pattern-matching algorithm used in forward-chaining production systems, which shares computation across rules so that the matching cost does not grow linearly with the number of rules on each cycle.

## retrograde
Describing analysis that works backward from terminal positions, as in retrograde endgame databases that solve games by computing outcomes from the end back toward earlier positions.

## reusable
Describing a resource that is occupied while an action uses it but becomes available again afterward, as opposed to a consumable resource that is used up.

## revelation principle
A result in mechanism design stating that any outcome achievable by some mechanism can also be achieved by a truthful mechanism, in which participants have an incentive to reveal their true preferences.

## revenue equivalence theorem
A result in auction theory stating that, under certain conditions, a broad class of auction formats yields the same expected revenue to the seller.

## reward
The scalar feedback signal an agent receives from its environment in reinforcement learning and Markov decision processes, which the agent seeks to maximize over time.

## reward shaping
A technique that adds extra, informative reward signals to guide a reinforcement learning agent toward good behavior more quickly, ideally without changing the set of optimal policies.

## reward-to-go
The sum of rewards an agent expects to receive from a given state onward to the end of the episode, which utility estimates aim to approximate.

## risk-averse
Describing an agent whose utility function is concave, so that it prefers a sure outcome over a gamble with the same expected monetary value.

## risk-neutral
Describing an agent whose utility is linear in value, so that it judges gambles solely by their expected monetary value.

## risk-seeking
Describing an agent whose utility function is convex, so that it prefers a gamble over a sure outcome of equal expected value.

## Robocup
An international competition in which teams of autonomous robots or software agents play soccer, used as a standard testbed for research in robotics and multiagent systems.

## robot navigation
The problem of moving a robot from a starting location to a goal while avoiding obstacles, typically combining localization, mapping, and motion planning.

## robotic soccer
A domain in which teams of robots play soccer, serving as a benchmark for perception, control, and coordination in dynamic multiagent environments.

## robust control theory
An area of control theory that designs controllers guaranteed to perform well despite bounded uncertainty in the system model or disturbances.

## ROC curve
A receiver operating characteristic curve, which plots the true positive rate against the false positive rate of a classifier across different decision thresholds, summarizing the tradeoff between sensitivity and specificity.

## rollout
In game playing and Monte Carlo methods, the technique of estimating a position's value by simulating one or more complete plays forward, often using a fast default policy, and averaging the outcomes.

## Roomba
A commercial autonomous vacuum-cleaning robot, cited as an example of a widely deployed consumer robot.

## Root Mean Square (RMS)
A mathematical formula, **root mean square** is often used as an error estimator. It is described as the root of the summation of all the squared errors. The formula is given as:
<br>***square_root[summation(a1^2 + a2^2 + ...)], where a1, a2...are some entities***
<br> In RMS error estimation, the above squared entities are replaced with squared errors.


## rules
Conditional statements of the form "if antecedent then consequent" used to represent knowledge, as in production systems and logical implications, forming the basis of rule-based reasoning.

# S

## S-set

In the version-space learning algorithm, the S-set is the set of most specific hypotheses that are still consistent with the observed examples; together with the G-set it bounds the version space.

## sample complexity

The number of training examples an agent needs before its learned hypothesis is, with high probability, approximately correct. It characterizes how much data a learning algorithm requires as a function of the desired accuracy and confidence.

## sample space

The set of all possible outcomes of a random experiment; each possible world is a member of the sample space, and probabilities are assigned so that they sum to one over the whole space.

## sampling rate

The frequency at which a continuous signal, such as an audio waveform, is measured to produce a discrete sequence of samples. In speech processing, for example, the analog signal is sampled thousands of times per second.

## SARSA

A model-free reinforcement learning algorithm (State-Action-Reward-State-Action) that updates the action-value function Q using the actually chosen next action rather than the best possible one, making it an on-policy method.

## SAT

The Boolean satisfiability problem: deciding whether there is an assignment of truth values to variables that makes a given propositional logic sentence true. SAT was the first problem proven NP-complete.

## satisfiability

A sentence is satisfiable if it is true in, or satisfied by, at least one model. Determining satisfiability is a central problem in logic and is closely tied to determining validity and entailment.

## satisfiability threshold conjecture

The conjecture that for random k-SAT problems there is a sharp threshold ratio of clauses to variables below which almost all instances are satisfiable and above which almost all are unsatisfiable, with the hardest instances clustered near the threshold.

## satisficing

Choosing an action that is good enough to meet some acceptability threshold rather than one that strictly maximizes expected utility. The term describes decision makers who settle for adequate outcomes given bounded resources.

## scaled orthographic projection

An approximation to perspective projection in computer vision in which objects lie within a small depth range, so the scene can be treated as orthographically projected and then uniformly scaled.

## scanning lidars

Range-finding sensors that sweep a laser beam across the scene and measure time-of-flight or phase shift to compute distances, producing a set of range measurements used for mapping and localization.

## scene

In computer vision, the three-dimensional world (objects, surfaces, and light sources) that gives rise to an image when projected onto the image plane.

## schedule

A function used in algorithms such as simulated annealing that specifies how a control parameter (for example, the temperature) decreases over time, governing the balance between exploration and convergence.

## schedulers

Components in a planning-and-execution system that assign start times and durations to actions, respecting resource limits and temporal constraints so that a plan can actually be carried out.

## schema

In a genetic algorithm, a pattern (template) describing a subset of strings that share fixed values at certain positions; schemas are used to analyze how promising building blocks propagate through the population.

## Scrabble

The word-forming board game, used in AI as an example domain combining a large branching factor, partial observability of opponents' tiles, and stochastic tile draws, which programs address with search and probabilistic reasoning.

## sealed-bid second-price auction

An auction (also called a Vickrey auction) in which bidders submit sealed bids, the highest bidder wins, but pays the amount of the second-highest bid. This makes truthful bidding a dominant strategy.

## search

The process of systematically exploring a space of possible action sequences or states to find one that achieves a goal. A problem-solving agent uses search to compute a solution before acting.

## search cost

The computational resources, such as time and memory, consumed while searching for a solution, as distinct from the cost (path cost) of the solution that is eventually found.

## search tree

A tree whose root is the initial state and whose branches correspond to actions, generated during search; nodes represent states reached along paths, and the same state may appear at multiple nodes.

## segmentation

In vision, the task of partitioning an image into regions of coherent pixels that likely belong to the same surface or object; more generally, dividing a signal into meaningful parts.

## selection

In a genetic algorithm, the step that chooses which individuals in the current population will reproduce, typically favoring those with higher fitness.

## semantic ambiguity

The situation in which a phrase or sentence can be assigned more than one meaning even though its syntactic structure is fixed, as when a word has multiple senses.

## semantics

The study of meaning: the rules that determine the truth of a sentence with respect to a possible world or model. In a knowledge representation language, semantics defines what each sentence asserts about the world.

## semi-supervised learning

Learning that uses a small amount of labeled data together with a large amount of unlabeled data, exploiting structure in the unlabeled examples to improve the learned model.

## semidynamic

A property of an environment that does not change while the agent is deliberating, but in which the agent's performance score does change with the passage of time.

## semiotics

The study of signs and symbols and how they convey meaning; referenced in discussions of how symbols in a representation come to be about things in the world.

## sensitivity analysis

Studying how the output of a model or decision changes as its inputs, parameters, or assumptions are varied, in order to identify which factors most strongly affect the conclusion.

## sensor interface layer

The lowest layer of a robot software architecture, responsible for communicating directly with sensors and actuators and providing timely data to higher-level control processes.

## sensor Markov assumption

The assumption that the current evidence variable depends only on the current state, so that the sensor model can be written as a probability of the observation given the current state alone.

## sensorless

Describing a problem or agent that must act without any percepts, so plans must succeed for every possible state consistent with the agent's belief; also called conformant planning.

## sequence form

A compact representation of strategies in an extensive-form game that describes behavior in terms of sequences of moves, allowing equilibria to be computed with a number of variables linear in the size of the game tree.

## sequential

An environment in which the current decision affects future decisions, so that short-term actions can have long-term consequences, as opposed to an episodic environment.

## sequential Monte Carlo

A family of algorithms, including particle filtering, that approximate a sequence of probability distributions by propagating and reweighting a set of weighted samples over time.

## set of support

A strategy for restricting resolution inference to clauses that are connected, directly or indirectly, to a designated set (often the negated goal), which reduces the number of irrelevant inferences.

## set semantics

An interpretation in which a collection is treated as a set, so that duplicate elements are ignored and only membership matters, as opposed to bag or list semantics.

## set-cover problem

The problem of choosing the fewest subsets from a given collection whose union covers all elements of a universe; it is NP-hard and arises, for example, in analyzing planning heuristics.

## set-level

A heuristic derived from a planning graph that estimates the cost of achieving a set of goals by the level at which they all appear together without a mutex between any pair.

## shading

The variation in brightness across a surface in an image caused by the geometry of the surface, the light sources, and the surface's reflectance; a cue used to recover shape.

## shadow

A region of a scene that receives less illumination because light is blocked by an intervening object; shadows provide cues about geometry and lighting but complicate image interpretation.

## shape

The geometric form of an object's surface, one of the properties a vision system may attempt to recover from images using cues such as shading, texture, motion, and stereo.

## shaving

A constraint-propagation technique that tentatively assigns a value to a variable and checks whether it leads to inconsistency; if so, the value is removed from that variable's domain.

## shortcuts

In a hierarchical or learned representation, connections or macro-actions that let an agent bypass intermediate steps and reach useful states more directly.

## shoulder

In a state-space landscape, a flat region from which progress is possible in one direction; unlike a local maximum, a hill-climbing search can eventually escape a shoulder by moving across the plateau.

## sibyl attack

An attack on a reputation or voting system in which a single adversary creates many false identities to gain disproportionate influence.

## sideways move

A move in local search that goes to a neighboring state of equal value, used to traverse a plateau or shoulder in the hope of eventually finding a way to improve.

## sigmoid perceptron

A linear classification unit whose weighted sum of inputs is passed through a sigmoid (logistic) function, producing a smooth, differentiable output between 0 and 1 rather than a hard threshold.

## significance test

A statistical procedure for deciding whether an observed effect, such as a difference in performance between two algorithms, is unlikely to have arisen by chance under a null hypothesis.

## similarity networks

A graphical modeling technique in which several local network fragments, each valid under a particular context, are combined to represent a large probabilistic model more conveniently.

## simulated annealing

A stochastic local search algorithm that sometimes accepts moves to worse states, with a probability that decreases according to a temperature schedule, allowing escape from local optima while gradually converging.

## simultaneous localization and mapping (SLAM)

The robotics problem of building a map of an unknown environment while at the same time keeping track of the robot's location within that map, using noisy sensors and motion.

## single agent

An environment in which only one agent operates, so there are no other decision makers whose behavior must be predicted or reacted to, in contrast to a multiagent environment.

## singly connected

A property of a network (also called a polytree) in which there is at most one undirected path between any two nodes; probabilistic inference in such Bayesian networks is efficient (linear in network size).

## singular

Describing a matrix that has no inverse because its determinant is zero; such matrices arise, for example, when data are linearly dependent and cause difficulties in certain estimation methods.

## singularity

A hypothesized future point at which self-improving artificial intelligence produces accelerating technological change beyond human ability to predict or control.

## situation

In the situation calculus, a logical term denoting a snapshot of the world, produced from an initial situation by a sequence of actions; fluents are evaluated relative to situations.

## situation calculus

A formalism in first-order logic for representing and reasoning about change, in which actions, situations, and fluents are objects, and the effects of actions are described by axioms.

## skeletonization

A method for robot motion planning that reduces the free space to a one-dimensional network of paths (a skeleton), so that navigation becomes a graph search along that network.

## Skolemization

**Skolemization** is the process of removing the existential quantifiers by elimination. This is similar to the inference rule ***Existential Elimination*** where an inference involving sentence *a*, variable *v* and constant *k* can be made provided *k* does not occur anywhere in the knowledge base.

## slack

In scheduling, the amount of time an action can be delayed without postponing the overall completion of the plan; the difference between the latest and earliest possible start times.

## slant

In vision, the angle by which a textured surface is tilted away from the viewing direction; together with tilt it describes surface orientation and can be estimated from texture gradients.

## sliding window

A technique in which a fixed-size region is moved across a signal or image at many positions and scales, applying a detector or model at each location, as in object detection.

## small-scale learning

Learning in a regime where the amount of training data is limited relative to model capacity, so that the main concern is fitting the available data well rather than computational cost.

## smoothing

**Smoothing** is the process of computing the distribution over past states given evidence up to the present.

## soccer

Robot soccer, used as a benchmark domain (for example in RoboCup) that combines real-time perception, control, multiagent coordination, and adversarial play.

## social laws

Conventions or rules adopted by a group of agents to constrain their behavior so as to avoid conflicts and enable smooth coordination without continual negotiation.

## Socratic reasoner

A reasoning system that answers questions by posing and answering a series of subquestions, guiding inference through a structured line of interrogation.

## soft margin

In support vector machines, a formulation that allows some training points to violate the margin or be misclassified, trading off margin width against classification errors to handle non-separable data.

## softmax function

The **softmax function** is a mathematical function often used for classification tasks. This function calculates the probability distribution of one class, over all the available classes. It's formula is given as:

*F(X) = exp(X) / summation(exp(X))* with the summation being over all the classes.

## software architecture

The organization of a program into components and the conventions by which they communicate; in agent design, it defines how perception, reasoning, and action modules are structured and interact.

## sokoban

A puzzle in which an agent pushes boxes onto target locations within a grid, used as a challenging planning benchmark because moves are irreversible and dead-ends are common.

## solution

A sequence of actions (a path) leading from the initial state to a goal state; an optimal solution is one with the lowest path cost among all solutions.

## sonar sensors

Range sensors that emit sound pulses and measure the time for echoes to return, computing distance to nearby objects; common on mobile robots but subject to noise and specular reflection.

## sound

A property of an inference procedure: an inference procedure is sound (truth-preserving) if it derives only sentences that are actually entailed by the premises.

## spam detection

The task of classifying email messages as unwanted (spam) or legitimate, commonly treated as a supervised text-classification problem using features of message content.

## sparse

Describing a representation, matrix, or model in which most entries are zero or most variables are directly connected to only a few others, a property often exploited for efficiency.

## sparse model

A learned model in which most parameters are zero, so predictions depend on only a small subset of the available features; sparsity aids interpretability and generalization.

## spatial reasoning

Reasoning about the positions, shapes, orientations, and relationships of objects in space, such as containment, adjacency, and connectivity.

## specialization

The operation of making a hypothesis more restrictive so that it covers fewer examples, used in version-space and inductive learning to remove false positives.

## specular reflection

Mirror-like reflection in which light bounces off a surface at an angle equal to the angle of incidence, producing highlights whose position depends on the viewpoint.

## specularities

Bright highlights on a surface caused by specular reflection of light sources; they signal glossy materials but can mislead algorithms that assume matte surfaces.

## speech act

An utterance viewed as an action performed by a speaker to affect a hearer, such as informing, requesting, promising, or commanding, in the theory of language as action.

## Speech Recognition

**Speech recognition** is the process of analyzing audio and ***recognizing*** parts related to speech within the audio file. This process of ***recognition*** may involve simply identifying the speech part, gender identification and also as complex as identifying the words spoken when given an audio. This field often overlaps into the domain of artificial intelligence and machine learning.

## split point

In learning a decision tree over a continuous attribute, the threshold value chosen to divide examples into two groups, selected to best separate the classes.

## stable

Describing a state, model, or matching that has no incentive or tendency to change; for example, a matching is stable if no pair of agents both prefer each other to their current partners.

## stable model

In answer set programming, an interpretation that is a minimal model consistent with a logic program under the stable-model (answer-set) semantics, providing a meaning for programs with negation as failure.

## standard normal distribution

The Gaussian distribution with mean zero and variance one; any normal random variable can be transformed into a standard normal by subtracting its mean and dividing by its standard deviation.

## standardizing apart

Renaming the variables in logical sentences so that no variable name is shared between sentences being combined, preventing accidental variable clashes during unification and inference.

## Starcraft

A real-time strategy video game used as an AI research domain that features partial observability, large action spaces, real-time decisions, and long-term strategy.

## start symbol

The distinguished nonterminal symbol in a grammar from which all valid derivations begin; sentences of the language are those derivable from the start symbol.

## state abstraction

Grouping many low-level states into a single abstract state so that planning or learning can reason at a coarser level, ignoring details irrelevant to the task.

## state estimation

Inferring a probability distribution over the current state of a partially observable system from the sequence of observations and actions; equivalently, maintaining a belief state.

## state space

The set of all states reachable from the initial state by any sequence of actions, together with the actions connecting them; searching for solutions amounts to exploring this space.

## state-space landscape

A visualization of a state space in which each state has a location and an elevation given by an objective (or heuristic) function; local search can be pictured as moving over this landscape seeking peaks or valleys.

## static

A property of an environment that does not change while the agent is deliberating, so the agent need not track the passage of time during decision making.

## stationarity assumption

The assumption that the process generating data does not change over time, so that the same probability model applies at every time step; it underlies many time-series and learning methods.

## stationary distribution

The distribution over states that a Markov process converges to and thereafter maintains, such that applying the transition dynamics leaves the distribution unchanged.

## stationary process

A stochastic process whose statistical properties, such as the distribution over states and transitions, are the same at every point in time.

## stemming

Reducing words to a common root form by stripping affixes (for example, mapping "running" to "run") so that related word forms are treated as the same term in text processing.

## step cost

The cost incurred by taking a single action to move from one state to another; path cost is the sum of the step costs along a path.

## step size

In gradient-based optimization, the factor (learning rate) by which the gradient is multiplied to determine how far to move at each iteration.

## stereo vision

Recovering depth by comparing two images taken from slightly different viewpoints and measuring the disparity between corresponding points, which is inversely related to distance.

## stochastic

A property of an environment (or process) whose outcomes are not fully determined by the current state and action but instead involve randomness, described by probabilities.

## stochastic beam search

A variant of beam search that selects the next generation of states probabilistically, with better states more likely to be kept, so as to maintain diversity and avoid premature convergence.

## stochastic games

Games with multiple states and probabilistic transitions in which one or more players take actions each turn; they generalize both Markov decision processes and matrix games.

## stochastic hill climbing

A hill-climbing variant that chooses at random among the uphill moves, with selection probability possibly depending on how steep each move is, rather than always taking the steepest ascent.

## stochastic policy

A policy that, in each state, specifies a probability distribution over actions rather than a single deterministic action; useful in partially observable or adversarial settings.

## straight-line distance

The Euclidean distance between two locations, commonly used as an admissible heuristic for route-finding problems because it never overestimates the actual road distance.

## strategic form

A representation of a game (also called normal form) as a matrix giving the payoff to each player for every combination of the players' strategies, abstracting away the order of moves.

## strategy

In game theory, a complete plan that specifies what action a player will take in every situation that could arise during the game.

## strategy profile

An assignment of one strategy to each player in a game; the payoffs to all players are determined once a strategy profile is fixed.

## strategy-proof

A property of a mechanism under which no participant can gain by misrepresenting their true preferences, so that honesty is a dominant strategy.

## strong AI

The claim, or the goal, that machines can be made to actually think and have minds in the full sense, as opposed to merely simulating intelligent behavior (weak AI).

## structural EM

An expectation-maximization method that learns the structure of a probabilistic model, not just its parameters, by searching over structures using expected sufficient statistics computed from incomplete data.

## structured representation

A representation in which each state is described in terms of objects and the relations among them, allowing variables and relations; more expressive than atomic or factored representations.

## stuff

A category of substance (a mass noun such as water or gold) that, unlike a discrete object, has no intrinsic individuation and any portion of which is still the same kind of thing.

## subcategory

A category that is a subset of another, more general category; for example, "dog" is a subcategory of "animal" in a taxonomy.

## subgoal independence

The assumption, used in some planning heuristics, that the cost of achieving a set of goals equals the sum of the costs of achieving each goal separately, ignoring interactions.

## subject-verb agreement

The grammatical requirement that a verb match its subject in features such as number and person; a constraint used in parsing and generating natural language.

## subjectivist

The view that probabilities represent an agent's degrees of belief rather than objective frequencies in the world, so that different agents may rationally hold different probabilities.

## subproblem

A smaller instance derived from a larger problem, whose solution contributes to solving the whole; used in decomposition methods and in deriving heuristics by relaxation.

## substitution

A mapping from variables to terms, applied to a logical expression to replace its variables; unification produces the substitution that makes two expressions identical.

## subsumption

The relation in which one description is more general than another, so that everything satisfying the more specific description also satisfies the more general one; used to organize categories and clauses.

## subsumption architecture

A reactive robot control design in which behavior is produced by layers of simple finite-state machines, with higher layers able to suppress or override lower ones, and no central world model.

## subsumption lattice

The partial ordering of terms or expressions by the subsumption (more-general-than) relation, forming a lattice used to reason about unification and generalization.

## successor

A state that can be reached from a given state by applying a single action; the successors of a state are generated by the successor function during search.

## successor-state axiom

An axiom in the situation calculus that fully specifies the truth value of a fluent after an action, stating that the fluent holds afterward if and only if the action made it true or it was already true and no action made it false.

## Sudoku

The number-placement puzzle, used as an example of a constraint satisfaction problem in which each row, column, and box must contain all digits exactly once.

## sum of squared differences

A measure of dissimilarity between two image patches or signals computed by summing the squares of the differences of corresponding values; used, for example, in stereo matching to find correspondences.

## superpixels

Groups of adjacent, perceptually similar pixels treated as single units in image processing, reducing the number of elements a vision algorithm must consider while respecting object boundaries.

## supervised learning

In **supervised learning** the agent observes some example
input-output pairs and learns a function that maps from input to
output.

## support vector machine

A supervised learning method that finds the decision boundary maximizing the margin between classes, defined by a subset of training points (support vectors), and can use kernels to form nonlinear boundaries in the input space.

## symmetry-breaking constraint

An extra constraint added to a problem so that among symmetric (interchangeable) solutions only one representative is allowed, reducing the search space without losing genuine solutions.

## synchro drive

A robot locomotion system in which all wheels turn and steer together, so the robot can move in any direction while keeping a fixed orientation.

## synchronic

Describing relations that hold among aspects of the world at a single moment in time, as opposed to diachronic relations that connect states across time.

## synchronization

Coordinating the timing of multiple agents, processes, or actions so that they proceed in a consistent order or at compatible times.

## syntactic ambiguity

The situation in which a sentence can be assigned more than one grammatical structure (parse), leading to more than one possible meaning; also called structural ambiguity.

## syntactic theory

A theory of how phrases and sentences of a language are structured, specifying the grammatical rules that determine which sequences of words are well formed.

## syntax

The rules that specify which sequences of symbols form well-formed sentences in a language, independent of their meaning; in logic, syntax defines legal sentences of the representation.

## synthesis

The construction of an artifact, such as a program or a plan, that provably meets a given specification; program synthesis derives code guaranteed to satisfy its requirements.
# T

## table lookup
Retrieving a stored answer directly from a table indexed by the input, rather than computing it. A table-driven agent maps every possible percept sequence to an action, an approach that is simple but infeasible for all but the tiniest domains because the table grows explosively.

## tabu search
A local search technique that keeps a short-term memory (the tabu list) of recently visited states or moves and forbids returning to them, helping the search escape local optima and avoid cycling.

## tactile sensors
Touch sensors, such as whiskers, bump panels, or pressure-sensitive skin, that let a robot detect physical contact with objects in its environment.

## taxonomy
An organization of objects into a hierarchy of categories linked by subclass (subset) relationships, allowing properties to be inherited from more general categories to more specific ones.

## Taylor expansion
A representation of a function as an infinite sum of terms computed from the function's derivatives at a single point, often truncated to a low-order polynomial to locally approximate the function.

## technological singularity
A hypothesized future point at which machine intelligence, through recursive self-improvement, rapidly surpasses human intelligence, making subsequent progress impossible for humans to predict or control.

## template
In information extraction and pattern matching, a fixed structure with slots to be filled in, used to recognize and extract instances of a stereotyped relation or event from text.

## temporal logic
A form of logic that reasons about propositions whose truth changes over time, providing operators to describe relationships between events and time points or intervals.

## temporal-difference
A reinforcement learning method that updates value estimates based on the difference between successive predictions, adjusting the value of a state toward the reward received plus the estimated value of the next state.

## temporal-projection
Predicting the state of the world at future times by reasoning forward from the current state through the effects of actions and events.

## term
In first-order logic, a logical expression that refers to an object; terms include constant symbols, variables, and function symbols applied to other terms.

## terminal states
The states of a game in which the game is over, at which point the outcome (win, lose, draw, or numeric payoff) is determined by the utility function.

## terminal test
A test applied to a game state that returns true when the game is over, distinguishing terminal states from those in which play continues.

## test set
A set of examples, kept separate from the training data, used to measure the accuracy of a learned model after training is complete.

## text classification
The task of assigning a text document to one or more predefined categories, such as labeling an email as spam or not spam, or sorting news articles by topic.

## texture
The pattern of visual variation on a surface, such as regular repeated elements, that provides cues in vision for segmentation, shape, and distance.

## theorem proving
Establishing that a sentence follows logically from a set of axioms by applying rules of inference, as opposed to checking entailment by enumerating models.

## thrashing
Excessive movement of a search or system back and forth between states or resources without making useful progress; in memory-bounded search it refers to repeatedly regenerating nodes that were discarded to save space.

## tiling
A method of representing a continuous space by covering it with overlapping regions (tiles), where an input's features are encoded by which tiles it falls into; used in function approximation for reinforcement learning.

## tilt
The rotation of a camera or sensor up or down about a horizontal axis, one of the standard degrees of freedom (along with pan) for aiming a viewing device.

## time and tense
Aspects of natural language that convey when an event occurs relative to the moment of speaking, requiring representations that link linguistic tense to points or intervals on a time line.

## time line
An ordered representation of time points and intervals along which events and the durations between them can be located and reasoned about.

## time of flight camera
A depth-sensing camera that measures distance to each point in a scene by timing how long emitted light takes to travel to the surface and reflect back.

## time to answer
A measure of how quickly a system produces a response to a query, used to evaluate the responsiveness of a question-answering or information-retrieval system.

## tit-for-tat
A strategy in repeated games in which a player cooperates on the first move and thereafter copies the opponent's previous move, rewarding cooperation and punishing defection.

## topological sort
An ordering of the nodes of a directed acyclic graph such that every edge goes from an earlier to a later node, used, for example, to order variables consistent with a partial-order plan or dependency structure.

## total Turing Test
An extended version of the Turing Test that also requires the machine to process visual input and manipulate objects, thereby testing perception and robotic capabilities in addition to conversation.

## toy problem
A simplified, precisely specified problem used to illustrate or benchmark methods, in contrast to a real-world problem whose formulation is messier and more open-ended.

## trace
A record of the sequence of steps an algorithm or agent takes during execution, often used for debugging or for explaining the derivation of a proof or plan.

## tractability
The property of a problem or algorithm being solvable in a practical amount of time, typically meaning solvable in time polynomial in the input size.

## tragedy of the commons
A situation in which individuals acting in their own self-interest deplete a shared resource, producing an outcome that is worse for everyone than if they had cooperated.

## trail
In logic-programming implementations, a data structure that records variable bindings made during unification so they can be undone when the system backtracks.

## training curve
A plot of a learning system's performance (such as accuracy or error) against the amount of training experience, showing how the model improves as it sees more data.

## training set
The collection of examples given to a learning algorithm, from which it induces a model or hypothesis.

## transfer model
In machine translation and related tasks, a model that converts a representation in one language into a corresponding representation in another by mapping structures at some intermediate level of analysis.

## transhumanism
A movement advocating the use of technology, including artificial intelligence and biotechnology, to enhance human physical and mental capabilities and transcend current biological limitations.

## transition model
A description of what each action does, specifying the state (or distribution over states) that results from taking a given action in a given state.

## transition probability
In a stochastic model such as a Markov process or MDP, the probability of moving to a particular next state given the current state and (in an MDP) the action taken.

## transpose
The matrix obtained by interchanging the rows and columns of a given matrix, so that entry (i, j) becomes entry (j, i).

## transposition table
A cache used in game-tree and other searches that stores results for states already evaluated, so that a state reached by different move orderings (transpositions) is not re-searched.

## traveling salesperson problem
The problem of finding the shortest tour that visits each of a set of cities exactly once and returns to the start; a classic NP-hard combinatorial optimization problem.

## tree decomposition
A way of organizing the variables of a constraint graph into overlapping clusters arranged as a tree, so that the problem can be solved efficiently when the clusters are small.

## tree width
A measure, derived from a tree decomposition, of how close a graph is to being a tree; it is one less than the size of the largest cluster, and low tree width permits efficient inference.

## treebank
A corpus of sentences that have been annotated with their syntactic parse structures, used to train and evaluate parsers and other language models.

## truth
The correspondence between a sentence and the state of the world; a sentence is true in a world (model) when the world is arranged as the sentence asserts.

## truth value
One of the values a proposition can take, in classical logic either true or false, assigned by a model.

## truth-preserving
Describing an inference rule that produces only conclusions that are true in every model in which its premises are true, so that it never derives a falsehood from truths (i.e., a sound rule).

## truth-revealing
Describing a mechanism, such as an auction, in which the best strategy for each participant is to report their true preferences or valuations.

## turbo decoding
An iterative decoding technique for error-correcting codes in which two or more decoders exchange probabilistic estimates back and forth, closely related to loopy belief propagation.

## Turing Test
  The **Turing Test** is a test proposed by Alan Turing in 1950, which is used to determine whether a computer is intelligent by evaluating the "human-ness" of its responses.

## type A strategy
A game-playing strategy that considers all possible moves to a fixed shallow depth and then applies a heuristic evaluation, searching wide but not deep.

## type B strategy
A game-playing strategy that explores promising lines of play to greater depth while ignoring moves judged unpromising, searching selectively rather than uniformly.

## type constraint
A restriction that limits the values a variable, slot, or argument may take to members of a specified category or type.

## type signature
A specification of the number, types, and order of the arguments a function or relation takes, and the type of value it returns.

# U

## ultraintelligent machine
A hypothetical machine, in I. J. Good's terminology, whose intellectual abilities far exceed those of any human, and which could design even better machines, potentially triggering an intelligence explosion.

## unary constraint
In a constraint satisfaction problem, a constraint that restricts the value of a single variable on its own, independent of any other variable.

## unbiased
Describing an estimator whose expected value equals the true quantity being estimated, so that on average it neither overestimates nor underestimates.

## uncertainty
The condition of an agent not knowing for certain the current state of the world or the outcomes of its actions, typically handled by reasoning with probabilities.

## underconstrained
Describing a problem that has too few constraints relative to its variables, so that it admits many solutions and is generally easy to satisfy.

## understanding
In natural language processing, the process of deriving meaning from an utterance, going beyond surface form to build an internal representation of what was said.

## unification
The process of finding a substitution of terms for variables that makes two logical expressions identical, a key operation in first-order inference.

## unifier
A substitution that, when applied to two expressions, makes them syntactically identical; the most general unifier makes the fewest commitments necessary to unify them.

## uniform-cost search
A search strategy that expands the unexpanded node with the lowest path cost from the start, guaranteeing an optimal solution when step costs are nonnegative.

## Unimate
An early industrial robot arm, one of the first to be deployed on a factory assembly line, notable in the history of robotics.

## uninformed search
A search strategy, also called blind search, that uses no information about the problem beyond its definition, exploring the state space without any estimate of distance to a goal.

## unique action axioms
Axioms stating that distinct action names denote distinct actions, used in logical formulations of planning to keep actions from being conflated.

## unique string axiom
An axiom asserting that syntactically different string (or name) constants refer to different objects, so that distinct symbols are guaranteed to denote distinct entities.

## unit clause
A clause containing exactly one literal, which forces the truth value of that literal in any satisfying assignment.

## unit preference
A resolution search heuristic that prefers to perform resolutions involving unit clauses, since these tend to shorten clauses and drive the search toward the empty clause.

## unit propagation
An inference process that repeatedly assigns the value forced by each unit clause and simplifies the remaining clauses accordingly, central to modern satisfiability solvers.

## unit resolution
A resolution inference rule in which one of the two clauses being resolved is a unit clause, producing a shorter clause.

## units function
A function that attaches a unit of measure to a numeric quantity, allowing an ontology to represent measures such as lengths, weights, and durations.

## universal grammar
The hypothesis that there is a body of grammatical structure common to all human languages, reflecting innate constraints on the form that natural languages can take.

## Universal Instantiation
An inference rule that allows any ground term to be substituted for a universally quantified variable, yielding a specific instance of the general sentence.

## unknown
Describing an environment or problem in which the agent lacks knowledge of the rules or the effects of its actions, so that it must discover them through experience.

## unobservable
Describing an environment in which the agent's sensors provide no information about the state, so the agent must act without perceiving its situation.

## unrolling
Expanding a temporal or recursive model into an explicit sequence over time, such as replicating a dynamic Bayesian network across time slices to form a static network for inference.

## unsupervised clustering
The task of detecting groupings of similar examples in unlabeled data, inferring categories without being told in advance what they are.

## unsupervised learning

In **unsupervised learning**
the agent learns patterns in the input without any explicit feedback.

## upper confidence bounds on trees
A Monte Carlo tree search algorithm (UCT) that selects which moves to explore by balancing their estimated value against the uncertainty of that estimate, favoring promising but under-sampled lines.

## upper ontology
A general, top-level ontology of broad concepts such as objects, events, time, and substances, meant to serve as a common framework under which more specialized, domain-specific ontologies can be organized.

## Urban Challenge
A 2007 DARPA competition in which autonomous vehicles had to drive through a mock urban environment, obeying traffic rules and interacting with other traffic, advancing self-driving car research.

## utility
A number expressing the desirability of a state or outcome to an agent; a rational agent chooses actions so as to maximize its expected utility.

## utility independence
A property of a multi-attribute utility function in which preferences over the outcomes of one set of attributes do not depend on the values of the remaining attributes, allowing the utility function to be decomposed.

# V

## vague

A property of natural-language predicates whose boundaries are not sharply defined, so that borderline cases exist for which it is unclear whether the predicate applies (for example, "tall" or "bald").

## validation set

A part of the dataset that is used to tune the parameters of a machine learning model. It can also be used to determine a stopping point for the back-propagation algorithm.

## validity

A property of a sentence in logic: a sentence is valid if it is true under every possible interpretation (in every model). Valid sentences are also called tautologies.

## value

A quantity assigned to an outcome or state, typically expressing its utility or desirability to an agent; in decision making and reinforcement learning, the value of a state is the expected sum of rewards obtainable from it.

## value alignment

The problem of ensuring that the objectives and behavior pursued by an AI system correspond to the true values and intentions of humans, rather than to a misspecified or literal interpretation of a stated goal.

## vanishing point

A vanishing point of a function means the function has a zero at the point or on the set.

## variable

A variable is a symbol on whose value a function, polynomial, etc., depends.

## variational approximation

A technique for approximate inference that replaces an intractable probability distribution with a simpler, tractable one and adjusts the simpler distribution to be as close as possible to the true one, turning inference into an optimization problem.

## variational parameters

The adjustable parameters of the simplified distribution used in a variational approximation; they are optimized to make the approximation as close as possible to the true distribution.

## VCG

See Vickrey-Clarke-Groves.

## vector

A vector is formally defined as an element of a vector space R^n. A vector is given by n coordinates and can be specified as (A_1,A_2,...,A_n).

## vector field histograms

A robot motion technique that summarizes local occupancy-grid data as a histogram of obstacle density in different directions, allowing the robot to steer toward directions with low obstacle density.

## vehicle interface layer

The software layer in an autonomous vehicle that translates high-level control decisions into low-level commands for the vehicle's actuators (steering, throttle, brakes) and abstracts away the hardware details.

## verification

The process of proving that a program or system meets its specification and behaves correctly under all inputs, for example by formal proof rather than testing.

## version space

The set of all hypotheses consistent with the training examples seen so far. The candidate elimination algorithm maintains this set, narrowing it as new examples arrive.

## Vickrey-Clarke-Groves

A general mechanism-design scheme for multi-agent settings that generalizes the Vickrey auction; it charges each agent an amount equal to the loss its participation imposes on the other agents, making truthful reporting a dominant strategy and yielding socially optimal allocations.

## virtual counts

Fictitious observation counts, encoded by the hyperparameters of a Beta or Dirichlet prior, that represent prior belief about a probability as if a certain number of outcomes had already been observed.

## virtual support vector machine

A support vector machine trained on additional examples generated by applying known invariance transformations to the support vectors, so that the learned classifier respects those invariances.

## VLSI layout

A combinatorial optimization problem, often used as a search benchmark, of arranging the components and wiring of an integrated circuit on a chip so as to minimize area, wire length, and other costs while satisfying design constraints.

## vocabulary

The set of distinct words or tokens used in a language model or text corpus.

## Voronoi graph

A representation of free space for robot motion planning obtained from the Voronoi diagram, consisting of the set of points equidistant from two or more obstacles; paths along it keep maximum clearance from obstacles.
# W

## weak AI

The view that machines can be built to act as if they were intelligent, without any claim that they actually possess minds or consciousness. Contrasted with strong AI, which claims that suitably programmed machines genuinely have minds.

## weak learning

A learning setting in which a hypothesis need only perform slightly better than random guessing; such weak learners can be combined, for example by boosting, into a strong, highly accurate ensemble.

## weight

A numerical coefficient associated with an input or feature that determines its contribution to a model's output, as in a linear function or the connections of a neural network; learning consists largely of adjusting these values.

## weight space

The space of all possible settings of a model's weights; learning can be viewed as searching this space for the point that minimizes the loss.

## weighted A* search

A variant of A* search that multiplies the heuristic term by a factor greater than one in the evaluation function f(n) = g(n) + W x h(n), biasing the search toward the goal to find solutions faster at the cost of possibly suboptimal paths.

## weighted training set

A training set in which each example carries a weight indicating its relative importance; learning algorithms such as boosting minimize a weighted error that emphasizes the more heavily weighted examples.

## wide content

In philosophy of mind, the view that the content of a mental state depends partly on the external environment and history of the agent, not solely on the agent's internal (brain) state. Contrasted with narrow content.

## Widrow-Hoff rule

A gradient-based weight-update rule (also called the delta rule or LMS rule) that adjusts weights in proportion to the error between predicted and target output, used to train linear units.

## Winnow algorithm

An online learning algorithm for linear threshold functions that updates weights multiplicatively rather than additively; it performs well when only a few of many features are relevant.

## workspace representation

A representation of a robot's environment in terms of the physical (Cartesian) space in which the robot and obstacles reside, as opposed to configuration space.

## wrapper

A layer of software that surrounds another program or data source, converting its inputs and outputs into a standard form; for example, a wrapper extracts structured information from a web page by exploiting its formatting regularities.

## wumpus world

A grid-based cave environment used as a running example for logical agents. An agent must find gold and avoid pits and a wumpus, reasoning from local percepts such as breezes and stenches to infer the locations of hidden hazards.
# Z

## zero-sum games

In game theory and economic theory, a **zero-sum game** is a mathematical representation of a situation in which each participant's gain or loss of utility is exactly balanced by the losses or gains of the utility of the other participants. If the total gains of the participants are added up and the total losses are subtracted, they will sum to zero.
