# Glossary for *Artificial Intelligence: A Modern Approach*

[A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [J](#j) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w) | [X](#x) | [Y](#y) | [Z](#z)

## 8-puzzle

**8-puzzle** consists of a 3x3 grid containing 8 numbered tiles and a blank space. A tile adjacent to the blank space can slide into that space. The object is to reach a specified **goal state** from a given **initial state**.

# A

## A3C

Short for Asynchronous Advantage Actor-Critic, a policy-gradient reinforcement learning technique used to train agents for tasks such as playing Go and controlling robots.

## absolute error

For a value estimated by an algorithm (such as a utility computed by an iterative method), the magnitude of the difference between the estimate and the true value. It is used, for example, in bounding how far value iteration's utility estimates can be from the correct utilities.

## abstraction

The process of removing detail from a representation, so that abstract states and actions correspond to large sets of detailed world states and detailed action sequences. An abstraction is valid if any abstract solution can be elaborated into a solution in the more detailed world, and useful if carrying out the abstract actions is easier than solving the original problem. Good problem formulation chooses the right level of abstraction.

## abstraction hierarchy

A layered organization of abstractions in which higher levels omit detail present at lower levels, so that a solution can be found at an abstract level and then refined into progressively more detailed solutions.

## acceptance probability

In the Metropolis-Hastings sampling algorithm, the probability used to decide whether to accept or reject a candidate state drawn from the proposal distribution, chosen so that the sampling converges to the target distribution.

## accessibility relations

In modal logic, a relation between possible worlds that specifies, for each world, which other worlds are accessible from it. Modal operators are interpreted using these relations; for example, a proposition is necessarily true in a world if it holds in all worlds accessible from that world.

## accidental checkmate

In games of imperfect information, a checkmate that results from following a strategy that happens to succeed for the true board state, even though the player could not have known in advance that it would produce checkmate.

## action cost function

A function c(s, a, s') that gives the numeric cost of applying action a in state s to reach the resulting state s'.

## action monitoring

A form of execution monitoring in which, before executing each action, the agent verifies that the action's preconditions still hold, rather than checking the preconditions of the entire remaining plan. If a precondition fails, the agent replans.

## action schema

In classical planning (PDDL), a lifted representation of a family of ground actions, consisting of the action's name, its list of variables, a **precondition**, and an **effect**, where the precondition and effect are each conjunctions of literals. Choosing constants to instantiate the variables yields a concrete ground action.

## action-utility function

A function, also called the **Q-function** and written **Q(s, a)**, giving the expected utility of taking action *a* in state *s*. It is related to the state utility by U(s) = max_a Q(s, a), and it lets an agent choose the best action directly, without needing a transition model of the environment.

## actions

The things an agent can do in an environment. In a search problem this is modeled by a function **Actions(s)** that returns the set of actions that can be executed in state *s*.

## activation

In a neural network, the output value produced by a unit, obtained by applying the unit's (nonlinear) activation function to the weighted sum of its inputs. This value is passed on as a signal to units in the next layer.

## activation function

The nonlinear function *g* associated with a unit in a neural network that is applied to the weighted sum of the unit's inputs to produce its output. Its nonlinearity is essential, since a network built only from linear functions would itself be just a linear function; common choices include the logistic/sigmoid, ReLU, and tanh functions.

## active learning

Learning in which the agent gets to decide what actions to take rather than following a fixed policy, using its freedom to explore and gather informative experiences. It contrasts with a passive learning agent, which has a fixed policy and simply learns from the experiences that policy generates.

## actor

In an actor-critic reinforcement learning architecture, the component that represents and selects actions according to the current policy. The actor adjusts the policy using feedback from the critic, which estimates how good the current policy is via a value function.

## adaptive dynamic programming

A reinforcement learning method (ADP) in which the agent learns the transition model connecting the utilities of states and the reward function, then solves the resulting Markov decision process using dynamic programming (e.g., by solving the Bellman equations). For a fixed policy these equations are linear and can be solved directly.

## adaptive sampling

A statistical technique, applied to Monte Carlo algorithms, that adjusts the sampling process as it proceeds in order to speed up convergence.

## adaptive utility

A model in which an agent is uncertain about its own utility function and can refine it by gathering more information through experience.

## add list

In the STRIPS/PDDL representation of an action's effect, the set of positive literals (ADD(a)) that are added to the state when the action is applied. It is contrasted with the delete list, DEL(a), the negative literals removed from the state.

## additive value function

A value function over multiple attributes that can be expressed as the sum of the value contributions of the individual attributes, valid when the attributes are mutually preferentially independent.

## admissible heuristic

A heuristic that never overestimates the cost to reach a goal; such a heuristic is therefore optimistic. When A* search uses an admissible heuristic, it is cost-optimal.

## adversarial search

Search in competitive (multi-agent) environments where two or more agents have conflicting goals. In AI it is studied through games such as chess, Go, and poker, where states are easy to represent and actions have precisely defined effects; each move is evaluated in terms of gain for one player and loss for the opponent.

## adversary argument

A technique for proving lower bounds on a problem's difficulty by imagining an adversary that constructs the problem instance while the algorithm runs, answering queries in the worst consistent way. For example, an adversary can place goals and dead ends so as to force an online search agent to fail or to follow an arbitrarily inefficient route.

## agent

Anything that can be viewed as perceiving its **environment** through **sensors** and acting upon that environment through **actuators**.

## agent function

The abstract mathematical description of an agent's behavior, mapping any given percept sequence to an action.

## agent program

The concrete implementation, running on some physical architecture, of the agent function for an artificial agent. It is important to distinguish this from the agent function: the function is an abstract mathematical description, while the program is a concrete implementation.

## agglomerative clustering

A form of hierarchical clustering that proceeds bottom-up: every data point starts in its own cluster, and at each step the two most similar clusters are merged, building a hierarchy that can be displayed as a dendrogram.

## aggregation

A technique of grouping individual objects into quantities when the objects are indistinguishable, representing them as numbers (e.g., Inspectors(2)) rather than as named entities. Aggregation reduces complexity, letting a planner detect resource conflicts immediately instead of trying every assignment of individuals.

## AI Index

An annual report produced by Stanford's One Hundred Year Study on AI (AI100), available at aiindex.org, that tracks progress and trends in the field of artificial intelligence.

## algorithm

An effective procedure: a finite sequence of unambiguous steps that, when carried out on a given input, produces the intended output and terminates. The word derives from the 9th-century mathematician al-Khwarizmi.

## alignment method

A technique aimed at solving the value alignment problem: ensuring that the objectives an AI system actually pursues match what its designers really want, so that a utility-maximizing agent does not optimize a misspecified objective and produce harmful behavior (the "King Midas problem").

## alpha-beta

The two bounds maintained during alpha-beta pruning of a minimax tree: **alpha (α)** is the value of the best (highest-value) choice found so far along the current path for MAX ("at least"), and **beta (β)** is the value of the best (lowest-value) choice found so far for MIN ("at most").

## alpha-beta pruning

A technique applied to a standard minimax search that prunes away branches which cannot possibly influence the final minimax decision. It updates the α and β bounds as it goes and cuts off the recursive search at a node as soon as the node's value is known to be worse than the current bound, returning the same move as full minimax but examining far fewer nodes.

## ambient illumination

A constant illumination term added to the predicted intensity of a surface to account for the diffuse light a surface receives from sources other than the direct light—such as sky light and light reflected from other surfaces (interreflections)—so that shadowed regions are not rendered uniformly black.

## ambiguity

The property of a representation, phrase, or situation of having more than one possible interpretation. Natural languages suffer from ambiguity, which is a difficulty for using them as representation languages.

## ambiguity aversion

A tendency, illustrated by the Ellsberg paradox, for people to prefer known (well-defined) probabilities over unknown ones, choosing gambles whose odds are specified over otherwise comparable gambles whose odds could lie anywhere in a range—behavior that violates the standard axioms of rational preference.

## analogical reasoning

Reasoning that draws a conclusion about one situation from its similarities to another: given that two systems share certain known properties, it infers that they are likely to share some further property.

## analysis of algorithms

The study of how the resource requirements (typically running time) of an algorithm grow as a function of the size of its input, usually characterized in terms of worst-case or average-case behavior.

## anchoring effect

A cognitive bias in which an initial reference value (the "anchor") skews subsequent judgments. For example, a restaurant lists an expensive bottle that nobody buys, which raises customers' estimate of the value of all wines and makes a cheaper bottle seem like a bargain.

## And-Elimination

A sound inference rule of propositional logic stating that from a conjunction, any of its conjuncts may be inferred (from A ∧ B, infer A, and infer B).

## AND-parallelism

A form of parallel execution in logic programming in which the multiple conjuncts (subgoals) of a single goal are solved simultaneously. It contrasts with OR-parallelism, in which the alternative clauses that could satisfy a goal are explored in parallel.

## angelic nondeterminism

A form of nondeterminism (borrowed from the programming-languages notion contrasted with "demonic" nondeterminism) in which the agent itself, rather than an adversary, gets to choose which of an action's possible outcomes occurs. It underlies the angelic semantics of high-level actions.

## angelic semantics

An interpretation of high-level actions in hierarchical planning based on the **reachable set** of an HLA—the set of states reachable by some implementation of it. Because the agent can choose which element of the reachable set to land in, a high-level plan achieves the goal if its reachable set intersects the goal states, letting the agent commit to the plan knowing a successful refinement exists.

## answer literal

An extra literal attached to the negated goal in resolution theorem proving so that answers can be extracted: it carries the variables of interest, and when the resolution proof derives the empty (answer) clause, the variable bindings that make the query true can be read off from it.

## answer set programming

A declarative programming paradigm (ASP), based on the stable-model (answer-set) semantics, that extends Prolog by allowing disjunction and negation and is aimed at hard (chiefly NP-hard) search problems.

## answer sets

The stable models that are the solutions of an answer set program: consistent sets of ground literals, each satisfying the program's rules under the stable-model semantics and representing one acceptable way the world could be.

## aortic coarctation

A medical condition consisting of a narrowing of the aorta; used in the book as an example domain for probabilistic/diagnostic reasoning.

## Appearance

Shorthand for what an object tends to look like, used as the basis for recognizing objects from their visual characteristics.

## appearance model

A description of what a person or object looks like, used in visual recognition and tracking to match image data against expected appearance.

## applicable

Describes an action that can legally be taken in a given state; an action is applicable in state s if s satisfies the action's preconditions.

## apprenticeship learning

Learning to perform a task from demonstrations by an expert rather than from an explicit reward signal, so that the observed expert behavior serves in place of the reward information normally used in reinforcement learning.

## approximate near-neighbors

The problem of finding, with high probability, an example point in a data set that lies near a given query point, without requiring the exact nearest neighbor.

## architecture

The underlying computing machinery (and any fixed software layer) on which an agent program runs; the agent program plus the architecture together produce the agent. More broadly, the fixed structure over which the details of a system (such as a neural network's layers) are arranged.

## arity

The fixed number of arguments taken by a function or predicate symbol. Every predicate and function symbol in first-order logic comes with an arity.

## Artificial General Intelligence

A research movement (AGI) that seeks to build machines with broad, general-purpose intelligence rather than systems specialized to narrow tasks.

## artificial intelligence

The field concerned not only with understanding intelligence but also with building intelligent entities, measured either by fidelity to human performance or against an abstract standard of rationality.

## artificial life

A field of study that examines lifelike systems, their processes, and their evolution, typically by simulating them with computer models, robotics, or biochemistry; it extended ideas from genetic and evolutionary computation to try to create or model life-like behavior.

## ascending-bid

The ascending-bid (or English) auction, the best-known auction mechanism, in which the auctioneer starts at a minimum (reserve) bid and solicits successively higher bids; the auction ends when no one will bid higher, and the last (highest) bidder wins and pays that price. It is usually both efficient and revenue-maximizing.

## Asilomar AI Principles

A set of voluntary self-governance guidelines for AI research and development, put forward by the Future of Life Institute.

## Asilomar Principles

A set of guidelines for beneficial AI research and development, produced at the 2017 Asilomar conference organized by the Future of Life Institute, addressing research goals, ethics, values, and longer-term issues of safety and control for AI systems.

## assignment

In a constraint satisfaction problem, a choice of values for variables. A **consistent** (or legal) assignment violates no constraints; a **complete** assignment gives a value to every variable; a **partial** assignment leaves some variables unassigned; and a **solution** to a CSP is a consistent, complete assignment.

## assistance

A decision-theoretic model in which a robot tries to determine and help achieve a human's goal about which it is initially uncertain.

## associative memory

A memory in which an entire stored pattern can be retrieved by presenting a fragment of it, indexing into the memory by content rather than by address. Hopfield networks are an example of neural networks that learn to store patterns as an associative memory.

## assumed-density filter

An approximate filtering technique that maintains a tractable approximation of the posterior distribution by assuming it belongs to a restricted family of densities.

## asymptotic analysis

Analysis of an algorithm's resource usage as a function of input size, abstracting away constant factors, typically expressed with O() notation. It lets one make claims such as that, as input size grows toward infinity, an O(n) algorithm is better than an O(n²) one, without relying on any single benchmark.

## asymptotic bounded optimality

A relaxed form of bounded optimality: an agent program is asymptotically bounded optimal if it performs at least as well as any other program running on a machine that is a constant factor slower or smaller, making the goal of doing the best possible with limited resources attainable in the limit.

## asynchronous policy iteration

A general form of policy iteration in which states are updated individually and in any order, rather than sweeping through all states in each iteration.

## ATMS

An assumption-based truth maintenance system, which supports efficient context-switching between hypothetical worlds by representing all previously considered states at once. Rather than merely labeling each sentence "in" or "out," an ATMS records, for each sentence, the sets of assumptions under which it would be true.

## atom

The smallest unit at a given level of description whose only relevant property is identity—being the same as or different from another atom; in an atomic representation of state, an entire world state is treated as a single atom (a black box with no internal structure).

## atomic representation

A representation in which each state of the world is indivisible—a "black box" with no internal structure, whose only testable property is whether it is identical to or different from another state. Standard search, game-playing, hidden Markov model, and MDP algorithms all work with atomic representations. It contrasts with factored and structured representations.

## atomic sentence

The simplest kind of sentence, with no logical connectives. In propositional logic it is a single proposition symbol (or the constants True/False); in first-order logic it is a predicate applied to terms. A literal is an atomic sentence (positive) or a negated atomic sentence (negative).

## attribute-based extraction

An information-extraction task in which the text is treated as describing a single object, and the goal is to extract the values of that object's attributes—for example, reading the price, size, or manufacturer of a product from its description.

## augmented grammar

A grammar whose nonterminals are not just atomic symbols (like NP) but structured representations carrying features (such as case, number, person, and meaning), so that productions can be augmented with conditions on those features to make finer distinctions and build up semantics compositionally.

## authority

In web search using the HITS algorithm, a page that is a good source of content on a query topic. A page's authority score is high when it is pointed to by many good hub pages, and hub and authority scores are computed jointly, each reinforcing the other.

## autoencoder

An unsupervised deep learning model consisting of an encoder that maps an input to an internal representation and a decoder that reconstructs the input from that representation.

## automated reasoning

The capability that lets a system use stored knowledge to answer questions and draw new conclusions.

## automatic assembly sequencing

A robotics planning task of finding a feasible order in which to put a product's parts together (as for assembling an electric motor). Generating the legal actions—reasoning about geometry and reachability so each part can be added without collision—is the expensive part of the problem.

## automatic differentiation

A method that systematically applies the rules of calculus to compute exact gradients for any numeric program; back-propagation in deep learning is an application of its reverse mode.

## autonomic computing

An approach in operating systems and networks in which computer systems monitor and control themselves using a perceive-act loop and machine learning methods.

## autonomy

The extent to which an agent relies on its own percepts and learning rather than on the prior knowledge built in by its designer. A rational agent should be autonomous—it should learn what it can to compensate for partial or incorrect prior knowledge—though complete autonomy is seldom required from the very start.

## average reward

An optimality criterion for infinite (continuing) sequences that compares policies by the average reward obtained per time step, rather than by a discounted sum of rewards. It is a useful criterion for some ongoing tasks in Markov decision processes and reinforcement learning.

## axiom

A sentence taken as given in a knowledge base, providing basic factual information from which conclusions can be derived. Axioms may be definitions (of the form ∀x,y P(x,y) ⇔ …) or may state more general information; sentences entailed by the axioms are called theorems.

# B

## back-propagation

The method for computing the gradient of a neural network's loss with respect to its weights by passing the error at the output backward through the network. Because the perceived error at the input to one unit is the error at a later unit multiplied by the information along the path connecting them, the error is effectively propagated backward from the output layer to the earlier layers—hence the name.

## back-propagation through time

The name for the back-propagation algorithm when it is applied to a recurrent network that has been unrolled over a sequence of time steps; it is the same gradient-computation procedure as ordinary back-propagation, just applied through the temporal structure.

## backed-up value

In search and game-tree algorithms, a value assigned to a node by propagating a value up from its descendants. For example, in recursive best-first search the best f-value of a forgotten subtree is backed up to its parent; in minimax the value of a node is backed up from the appropriate child.

## backgammon

A typical stochastic game combining luck and skill, in which players move pieces around a board according to the roll of two dice, aiming to move all their pieces off the board. Because the dice outcomes are unknown in advance, its game tree must include chance nodes in addition to MAX and MIN nodes.

## background knowledge

The initial knowledge a knowledge-based agent's knowledge base is given before it begins operating, providing a starting store of facts about the world upon which the agent builds through perception and inference.

## background subtraction

A computer-vision technique for isolating the changing foreground of an image sequence by comparing each frame against a model of the static background, so that only the differing (foreground) pixels are extracted for further processing such as object recognition.

## backjumping

An improvement to backtracking search for CSPs in which, at a dead end, the search backtracks to the most recent variable in the conflict set—the set of previously assigned variables that caused the failure—rather than to the immediately preceding variable. In forward-checking search, however, simple backjumping is redundant.

## backmarking

A backtracking technique for constraint satisfaction that remembers which consistent and inconsistent pairwise assignments have already been checked, so redundant constraint checks can be avoided. It can be combined with conflict-directed backjumping.

## backoff model

A language-model smoothing method in which n-gram counts are estimated, but for any sequence with a low or zero count the model "backs off" to shorter (n-1)-gram estimates. Linear interpolation smoothing is a backoff model that combines trigram, bigram, and unigram models.

## backpropagation

The algorithm that computes how each weight and bias in a neural network affects the loss (the partial derivatives of the loss with respect to each parameter) by propagating the output error backward through the network, enabling gradient-descent training.

## backtracking search

A memory-efficient variant of depth-first search that generates only one successor at a time (rather than all of them), remembering which successors remain to be tried so it can back up and continue when a branch is exhausted.

## Backus-Naur form (BNF)

A formalism for writing context-free grammars, with four components: a set of terminal symbols, a set of nonterminal symbols, a start symbol, and a set of rewrite rules of the form LHS → RHS. It is used throughout the book to specify the syntax of languages, including logical languages and English grammars.

## backward-chaining

A goal-directed inference algorithm that works backward from a query: if the query is not already known, it finds implications whose conclusion is the query and tries to prove their premises recursively by backward chaining. It touches only relevant facts, and is useful for answering specific questions.

## bag of words

The application of naive Bayes to text, treating a document as an unordered collection of words. It is a generative model in which words are imagined to be drawn independently (with replacement) from a category-specific "bag," ignoring grammar and word order; despite being clearly wrong about independence, it supports accurate text classification.

## bag of words model

An application of naive Bayes to text in which a document is treated as an unordered collection of words; it is a generative model where the probability of each word is estimated independently, ignoring word order and context.

## bagging

An ensemble method (short for "bootstrap aggregating") that generates K distinct training sets by sampling with replacement from the original data, trains a hypothesis on each, and aggregates their predictions (plurality vote for classification, average for regression). It tends to reduce variance and helps when data is limited or the base model overfits.

## balance

A fairness criterion requiring that a predictive algorithm produce roughly equal false-positive and false-negative rates across the different groups (e.g., races) being considered.

## bandit superprocess

A generalization of the bandit process (abbreviated BSP) in which each arm is itself a full Markov decision process with multiple possible actions, rather than a Markov reward process offering only a single action.

## bang-bang control

A control regime in which the action switches abruptly between extreme values rather than taking intermediate ones—for example, the discrete "jerk left or jerk right" actions used in the cart–pole (inverted pendulum) problem.

## baseline

In binocular stereopsis, the displacement between the two camera (or eye) positions. Horizontal disparity equals the ratio of the baseline to the depth, so knowing the baseline lets one recover the depth of a scene point from its measured disparity; a larger baseline aids depth estimation.

## batch gradient descent

A gradient-descent learning rule (also called deterministic gradient descent) that sums the gradient over all training examples before taking each update step. For a convex loss it is guaranteed to converge to the global minimum, but each step can be very slow because it must sum over the entire training set.

## batch size

In stochastic gradient descent, the hyperparameter m specifying the number of training samples in each minibatch used to estimate the gradient at a given update step.

## Bayes' rule

The equation P(a|b) = P(b|a)P(a)/P(b) (also called Bayes' law or theorem), obtained by equating the two forms of the product rule. It lets one compute the probability of a cause given an observed effect from the reverse (causal) conditional probability, and underlies most modern AI systems for probabilistic inference.

## Bayes-Nash equilibrium

An equilibrium concept for games where players are uncertain about one another's strategies: it is an equilibrium with respect to a player's prior probability distribution over the other players' strategies, expressing that player's beliefs about the strategies the others are likely to adopt.

## Bayesian learning

A learning approach that calculates the probability of each hypothesis given the data and makes predictions by using all hypotheses weighted by their (posterior) probabilities, rather than committing to a single best hypothesis. In this way learning is reduced to probabilistic inference.

## Bayesian network

A directed acyclic graph in which nodes are random variables and edges encode conditional dependencies, together with a conditional probability distribution for each node given its parents. It compactly represents a full joint distribution and supports computing probabilities of query variables given evidence.

## Bayesian reinforcement learning

An approach to reinforcement learning that maintains a prior probability over hypotheses about the true model of the environment and updates it to a posterior via Bayes' rule as observations accumulate, then acts to optimize with respect to that distribution.

## beam search

A memory-limited variant of best-first search that keeps only the k nodes with the best scores on the frontier, discarding the rest. This makes it fast and incomplete/suboptimal, exploring only a focused portion of the search space; a variant instead keeps every node whose score is within a fixed amount of the best.

## behaviorism

A movement in psychology, led by John Watson, that rejected any theory involving internal mental processes on the grounds that introspection could not provide reliable evidence, insisting instead on studying only objective measures of the stimuli (percepts) given to a subject and its resulting responses (actions).

## belief function

In Dempster–Shafer theory, a measure (written Bel) that generalizes probability to interval-valued degrees of belief; it gives, for a proposition, the total probability mass committed to it, leaving the remaining mass free to reflect ignorance rather than being distributed over specific outcomes.

## belief propagation

A message-passing algorithm for inference in graphical models in which nodes exchange messages to compute posterior marginals. It is exact on polytrees; applied to graphs with loops it becomes "loopy belief propagation," an approximate method that can nonetheless work well in practice.

## belief revision

The process of updating a knowledge base when new information contradicts what was previously inferred or asserted, requiring retraction of the old sentence (and of anything inferred from it) to avoid contradiction. It is contrasted with belief update, which reflects a change in the world rather than new information about a fixed world.

## belief state

A representation of what an agent believes about the current situation: in nondeterministic or partially observable settings, the set of physical states the agent thinks it might be in; in probabilistic settings, a probability distribution over those states.

## Bellman equation

A recursive equation (named after Richard Bellman) that expresses the utility of a state as the immediate reward plus the discounted expected utility of the best available action's successor states. The utilities of the states are the unique solutions of the set of Bellman equations.

## Bellman update

The iteration step of value iteration that takes the right-hand side of the Bellman equation and plugs it into the left-hand side, thereby updating each state's utility from those of its neighbors. Applied repeatedly (simultaneously to all states), it is guaranteed to converge to the true utilities.

## benchmarking

Running a suite of standard problems (a benchmark) to measure and compare the performance and quality of algorithms or systems. Shared benchmark problem sets became a norm for demonstrating progress in AI.

## Bernoulli

A distribution for a Boolean variable whose range is {0,1}, so the variable takes the value 1 with some probability and 0 otherwise.

## Bernoulli bandit

A simple, well-known bandit problem in which each arm yields a reward of 0 or 1 according to a fixed but unknown success probability associated with that arm.

## best-first search

A general search strategy that repeatedly expands the frontier node with the minimum value of an evaluation function f(n). Different choices of f yield different algorithms, including greedy best-first search and A*.

## bias

In a model such as logistic regression or a neural network layer, the learned additive parameter b (added to the weighted sum of inputs before applying an activation function).

## biconditional

The logical connective "if and only if" (⇔): the sentence P ⇔ Q is true exactly when P and Q have the same truth value, and is equivalent to the conjunction of P ⇒ Q and Q ⇒ P.

## bidirectional search

A search strategy that runs two simultaneous searches—one forward from the start state and one backward from the goal state(s)—maintaining two frontiers and two tables of reached states, hoping the two meet in the middle.

## binary constraint

In a CSP, a constraint that relates exactly two variables. A CSP with only unary and binary constraints (a binary CSP) can be represented as a constraint graph.

## binary resolution

An inference rule that resolves exactly two complementary literals from two clauses to produce a resolvent. By itself it is not a complete inference procedure for first-order logic, but combined with factoring (removing redundant literals via unification) it becomes complete.

## binding list

Also called a substitution: the list of variable-to-value bindings, produced by ASKVARS or unification, that specifies which value each logical variable takes to make a query true. It is typically used with knowledge bases of Horn clauses, where each way of satisfying a query binds the variables to specific values.

## binocular stereopsis

The recovery of depth by combining the two slightly different images of a scene obtained from two eyes (or cameras) placed apart. It requires solving the correspondence problem to find matching points, whose disparity is then used to infer distance.

## biological naturalism

A position associated with John Searle holding that mental states are high-level emergent features caused by low-level physical processes in the neurons, and that the (unspecified) physical properties of neurons are essential—so that, on this view, appropriately organized computation alone would not produce real understanding.

## block sampling

An MCMC technique that helps the chain mix more quickly by sampling several variables jointly (conditioned on the combined Markov blanket of the block) rather than sampling each variable one at a time.

## blocks world

A famous simplified microworld consisting of a set of solid blocks on a tabletop that a robot hand can pick up one at a time, with tasks such as rearranging the blocks into a specified configuration. It served as the domain for classic early work in vision, planning, learning, and language understanding.

## bluff

In games such as poker, betting as if one's hand is good even when it is not, in order to deceive opponents. Bluffing is a core part of poker strategy that a naive game-solving agent will fail to perform.

## body

In a definite clause or rule written in Horn form (head ← body), the premise: the conjunction of conditions (the antecedent) that must hold in order to infer the head (conclusion).

## boid

A simulated bird-like agent in an artificial-life flocking model (introduced by Craig Reynolds) whose behavior emerges from simple local rules governing how it moves relative to nearby flockmates. The name is short for "bird-oid object."

## boosting

The most popular ensemble method, which uses a weighted training set. It starts with equal example weights and trains a hypothesis; it then increases the weights of misclassified examples and decreases those of correctly classified ones, trains the next hypothesis on the reweighted set, and repeats, combining the resulting hypotheses.

## boundary set

In version-space learning, the compact representation of all hypotheses consistent with the examples, given by two boundaries—the most general (G) and the most specific (S) consistent hypotheses—such that every consistent hypothesis lies between them.

## bounded cost search

A search that looks for any solution whose total cost is less than a given fixed constant C.

## bounded optimality

A criterion for rational agents holding that an agent program should produce behavior at least as good as that of any other program running on the same (fixed, resource-limited) machine—that is, be as good as possible given its computational constraints, rather than perfectly rational in an unconstrained sense.

## bounded PlanSAT

The decision problem of whether a planning problem has a solution of length k or less; it can be used to find optimal plans. Like PlanSAT it is decidable for classical planning and lies in the complexity class PSPACE.

## bounded rationality

The idea, associated with Herbert Simon, that real agents have limited computational resources and cannot always compute the fully optimal decision, so instead they "satisfice"—use heuristics and settle for a good-enough choice rather than the best one.

## bounded suboptimal search

A search that looks for a solution guaranteed to cost no more than a constant factor W times the cost of the optimal solution (as achieved, for example, by weighted A* search).

## bounds consistent

A property of a CSP with ordered numeric domains: for every variable, both its lower-bound and upper-bound values can be extended to a value of every other variable that satisfies the constraint between them. Achieving this is done by bounds propagation.

## bounds propagation

A constraint-propagation technique for variables with large ordered numeric domains that represents domains by upper and lower bounds and repeatedly tightens those bounds to make the CSP bounds-consistent. It is widely used in practical constraint problems such as scheduling.

## branching factor

The number of successors of a node in a search tree or graph (often referred to by its average value, b). It is a key parameter determining the size of the search space and the time and space cost of search.

## breadth-first search

A search strategy, appropriate when all actions have equal cost, that expands the root node first, then all of its successors, then all of their successors, and so on level by level.

## bridge

A card game for four players in two partnerships that, like other card games, features stochastic partial observability because the missing information comes from the random dealing of cards; it is a notable example of a partially observable multiplayer game in AI.

## bunch

In knowledge representation, a composite object formed by grouping a collection of parts, treated as a single (albeit unstructured) object—distinct from the abstract mathematical set of those parts. For example, BunchOf({Apple1, Apple2, Apple3}) denotes the composite object having the three apples as parts, so it can be assigned properties such as weight.

# C

## calculative rationality

Not explicitly defined in the book. In standard AIMA usage it refers to the property of a program that, given unlimited computation, would eventually return the rational (optimal) choice — i.e., it computes the right decision in principle but not necessarily in time to be useful. The book instead emphasizes limited rationality (acting appropriately when there isn't time to do all the ideal computations) as the practical alternative to perfect rationality.

## canonical distribution

A standard, patterned way of specifying a node's conditional probability table in a Bayesian network without listing all entries. When the parent-child relationship follows a recognizable pattern, the distribution can be given just by naming the pattern and supplying a few parameters; the simplest case is a deterministic node whose value is fixed exactly by its parents, and the noisy-OR relation is a common example.

## CART

Short for Classification And Regression Trees, a name used to cover both decision trees that predict discrete class labels and those that predict continuous numeric values.

## cart-pole

The cart-pole balancing problem (also called the inverted pendulum): the task of keeping a hinged pole roughly upright by applying left/right forces to move a cart, while keeping the cart's position within the track limits. It is a benchmark in reinforcement learning and control, notable for continuous state variables and discrete "bang-bang" (jerk left/jerk right) actions.

## cascaded finite-state transducers

A series of finite-state transducers chained together for information extraction, in which each transducer's output feeds the next. A typical cascade has stages that handle tokenization, complex words, basic groups, complex phrases, and merging structures across sentences.

## case agreement

A grammatical constraint, handled by an augmented grammar, requiring a noun phrase to have the correct grammatical case for its role; for example, a subject must be in the subjective case ("I") rather than the objective case ("me"), so "I saw he" is ungrammatical. It is one of the agreement conditions (along with subject-verb agreement) enforced via augmentation of grammar rules.

## case-based planning

An approach to planning that solves new problems by retrieving and adapting stored plans from previously solved cases rather than constructing each plan from scratch.

## causal

Describing the direction of a relationship that runs from cause to effect. AIMA contrasts the causal direction (e.g., P(symptom | disease), which a doctor typically knows) with the diagnostic direction (P(disease | symptom)); Bayes' rule lets you convert causal conditional probabilities into diagnostic ones.

## causal graphs

Graphical models, developed in Judea Pearl's theory of causality, that represent cause-and-effect relationships among variables.

## causal link

In partial-order planning, a protective link written from the effect of one action to a precondition of another, recording that the first action supplies a condition needed by the second. The link signals that no other action may be interposed that would negate (clobber) the protected condition.

## causal network

A restricted class of Bayesian networks that permits only causally compatible node orderings. Unlike ordinary Bayes nets, which capture probabilistic influence, causal networks capture genuine causal structure and support reasoning about interventions and decision making.

## cell decomposition

A motion-planning method that divides the robot's free space into a finite number of contiguous regions called cells; an exact cell decomposition yields a complete planning algorithm.

## center

In mechanism design, the distinguished agent that collects the strategy-choice reports from the agents playing the game and, via a known outcome rule, determines each agent's payoff. For example, the auctioneer serves as the center in an auction.

## central limit theorem

The theorem stating that the distribution formed by summing (or averaging) many independent random variables tends toward a normal (Gaussian) distribution as the number of variables grows large. It holds for almost any collection of variables provided no finite subset's variance dominates the rest, and it helps explain why the Gaussian distribution appears so frequently.

## certainty effect

An observed behavioral tendency (Kahneman and Tversky) in which people are strongly attracted to outcomes that are certain relative to those that are merely probable, leading to preferences inconsistent with any utility function. It persists even when the probability computations involved are trivially easy.

## certainty equivalent

The guaranteed value an agent will accept in lieu of taking a lottery (gamble). Most people accept a certainty equivalent less than the lottery's expected monetary value (reflecting risk aversion); e.g., accepting about $400 instead of a gamble paying $1000 half the time and $0 otherwise, whose EMV is $500. The gap between EMV and certainty equivalent is the insurance premium.

## certification

An instrument for establishing trust in a system, whereby an independent body attests that the system meets required standards of safety or quality (as Underwriters Laboratories did for electrical products).

## CFG

Context-free grammar — a grammar formalism in which rules rewrite a single nonterminal symbol regardless of context, defining a language as the set of terminal strings derivable from the start symbol. AIMA notes their limits (e.g., certain agreement/coordination phenomena) and discusses the probabilistic context-free grammar (PCFG) extension that assigns probabilities to rules.

## chain rule

The identity expressing a joint probability distribution as a product of successive conditional probabilities, each variable conditioned on its predecessors in some ordering. Derived by repeated application of the product rule, it (with a topological ordering) underlies how a Bayesian network's structure specifies the full joint distribution. (The term is also used for the calculus rule central to backpropagation.)

## chain rule of calculus

The calculus identity for differentiating composite functions on which all algorithms for automatically computing gradients (such as back-propagation) are based.

## chance nodes

Nodes in a game tree that represent random events, such as dice rolls in backgammon, in addition to the MAX and MIN nodes; their value is computed as an expected value.

## channels

The separate arrays into which a feature map is split in a neural network, allowing multiple feature maps to be stacked together.

## characters

The individual symbols of a writing system — letters, digits, punctuation, and spaces. AIMA describes character-level language models, in which the probability of each character depends on the previous n characters; these are useful for handling unknown words, languages that run words together, and language identification.

## chart

A data structure used by chart parsers to hold the partial and completed results of parsing a string of words (e.g., that a span forms a noun phrase), so that subphrases are analyzed only once rather than re-derived across branches of the search. AIMA presents the bottom-up probabilistic CYK algorithm as an example.

## checkers

A board game (also called draughts), central to AI history through Arthur Samuel's 1950s program, which learned its own evaluation function by playing many games against itself using early reinforcement-learning methods, reaching a strong amateur level. The game was later solved (2007) — a draw under perfect play.

## checkpointing

A strategy in automatic differentiation that recomputes intermediate values during the backward pass rather than storing them all, trading extra computation for reduced memory use.

## chess

The classic two-player, deterministic, fully observable, zero-sum board game long used as a benchmark for AI. In 1997 IBM's DEEP BLUE defeated world champion Garry Kasparov using heuristic alpha-beta search with sophisticated evaluation functions on specialized hardware; AIMA also discusses Type A vs. Type B search and modern neural-network/search engines.

## child node

A node created in a search tree by applying an applicable action to a parent node's state, representing a resulting successor state.

## Chinese room

John Searle's thought experiment arguing that a system manipulating symbols by following rules (like a person in a room translating Chinese via a rule book) can produce correct outputs without genuine understanding.

## Chomsky Normal Form

A restricted grammar format required by the CYK parsing algorithm, in which every rule is either a lexical rule producing a single word (X → word) or a syntactic rule with exactly two categories on the right-hand side (X → Y Z). Any context-free grammar can be automatically transformed into this form, so the restriction loses no expressive power.

## chronological backtracking

A backtracking strategy that, upon reaching a dead end, revisits the most recent decision point (the one made latest in time).

## circuit verification

The process of confirming that a circuit's design is correct by encoding the circuit's structure and gate behavior in a logical knowledge base. One can query the knowledge base (e.g., whether the outputs match a specification for all inputs) and, by perturbing or omitting facts, pinpoint errors in the design.

## circumscription

A nonmonotonic logic that can be seen as a more powerful, precise version of the closed-world assumption: certain named predicates are assumed "as false as possible," i.e., false for every object except those known to make them true. It is a model preference logic in which a model with fewer abnormal objects is preferred, letting default conclusions (e.g., "birds fly") be drawn unless explicitly overridden.

## Clark Normal Form

Not found in the book. (The book discusses Clark's negation-as-failure and the Clark completion for logic programs, but not a term "Clark Normal Form.") In standard usage it refers to a normalized form of a logic program's clauses, grouping all clauses defining a predicate into a single if-and-only-if equivalence as a step toward computing the Clark completion.

## Classical planning

The task of finding a sequence of actions to accomplish a goal in a discrete, deterministic, static, and fully observable environment.

## classification

In description logics, one of the principal inference tasks: checking whether a given object belongs to a category (contrast with subsumption, which checks whether one category is a subset of another). More broadly, classification is the machine-learning task of assigning inputs to a discrete set of categories.

## clause

A disjunction of literals. A clause with a single literal is a unit clause. Clauses are the building blocks of conjunctive normal form, and the resolution inference rule takes two clauses and produces a new clause containing all their literals except a pair of complementary literals.

## closed-loop

A control approach in which the agent continues to monitor its percepts while executing actions, keeping the feedback loop between agent and environment intact. It is used when the environment may be nondeterministic or the model incorrect, in contrast to open-loop execution that ignores percepts during execution.

## closed-world assumption

The assumption that any atomic sentence not known to be true is taken to be false.

## clustering

An exact Bayesian-network inference technique that joins individual network nodes into cluster (mega)nodes so that the resulting network becomes a polytree, which can then be solved efficiently. The combined nodes take on all joint value combinations of their members. (More broadly, clustering is the unsupervised grouping of data instances with similar features.)

## clutter

False alarms in a data-association / tracking problem: reported observations (blips) that are not caused by any real object. Clutter is one of the complications (along with detection failure and objects arriving/departing) that make real multitarget tracking harder.

## CMAC

Not found in the book. In standard AIMA/reinforcement-learning usage, a CMAC (Cerebellar Model Articulation Controller) is a function-approximation scheme that represents a value function using multiple overlapping tilings (tile coding) of the state space, giving a coarse-coded, computationally efficient set of features that support efficient local generalization.

## co-NP

The complexity class that is the complement of NP: for every decision problem in NP there is a corresponding co-NP problem with the "yes" and "no" answers reversed. P is a subset of both NP and co-NP, and it is believed (though unproven) that there are problems in co-NP that are not in P.

## co-NP-complete

The hardest problems within the class co-NP. AIMA notes, for example, that deciding propositional entailment (validity/tautology) is co-NP-complete, meaning it is probably no easier than NP-complete problems.

## coarse-to-fine

A search technique (coarse-to-fine search) that defines landmarks or abstractions at various hierarchical levels, solving the problem first at a coarse level of resolution and then progressively refining. Under certain conditions it provides an exponential speedup over ordinary search; it is used in areas such as image processing and search.

## coarticulation

Not found in the book. In standard usage (speech recognition), coarticulation is the phenomenon whereby the acoustic realization of a phoneme is influenced by the neighboring sounds, so that a phoneme's pronunciation blends with and varies according to the sounds before and after it.

## coastal navigation

A robot navigation heuristic that changes the cost function to encourage information-gathering — specifically, requiring the robot to stay near known landmarks (such as walls) so it can localize itself and reduce positional uncertainty, even at the cost of a longer path. It is a scripted alternative to methods that reason explicitly about expected information gain.

## cognitive architectures

Computational models of human reasoning, such as ACT and Soar, which are often built using production systems.

## cognitive psychology

The view of the brain as an information-processing device. Traceable to William James and revived by Craik, it holds that the mind translates stimuli into internal representations, manipulates those representations via cognitive processes to derive new ones, and retranslates them into action.

## cognitive science

The interdisciplinary field that combines computer models from AI with experimental techniques from psychology to build precise, testable theories of the human mind.

## collusion

An unfair or illegal agreement among two or more bidders to manipulate prices in an auction. It can occur through secret backroom deals or tacitly within the rules of the mechanism; discouraging collusion is a key goal of auction and mechanism design.

## color constancy

The perceptual ability whereby human observers largely ignore the color of the illuminating light and estimate the color a surface would have under white light, so a surface appears to have a roughly stable color despite changing illumination.

## communication

The intentional exchange of information among agents through the production and perception of signs (such as words in a language). In AIMA's modern view, language serves as a medium for conveying meaning between a speaker and listener; the meaning conveyed depends on both the utterance and the context in which it is spoken.

## commutativity

The property that the order in which a given set of actions is applied does not affect the outcome. AIMA exploits this in CSPs: because variable assignments commute, the search need only consider a single variable per node, collapsing the search tree from n!·d^n leaves down to d^n complete assignments.

## competitive

Describing a multiagent environment in which one agent's success comes at the expense of others, so their goals conflict. AIMA's example: in chess, the opponent tries to maximize its own performance measure, which by the rules minimizes the other agent's, making it a competitive — specifically zero-sum — situation.

## competitive ratio

In online search/exploration, the ratio of the total path cost an agent actually incurs (not knowing the environment in advance) to the optimal cost it would incur if it had known the search space beforehand. Adversarial environments can make this ratio unbounded; a ratio close to 1 indicates good online performance.

## complementary literals

A pair of literals in which one is the negation of the other (e.g., P and ¬P). The resolution inference rule operates by canceling such a complementary pair from two clauses to produce a new clause (the resolvent).

## complete assignment

In a constraint satisfaction problem, an assignment in which every variable has a value (contrasted with a partial assignment, which leaves some variables unassigned). A solution to a CSP is a complete assignment that is also consistent — i.e., satisfies all the constraints.

## complete data

In statistical learning, a data set in which every attribute (including the label) is observed for every example, with no missing or hidden values. AIMA notes that maximum-likelihood parameter learning with complete data decomposes into separate problems, one per parameter, making learning much simpler.

## complete-state formulation

A problem formulation in which every state contains all the components of a solution, though they may not yet be correctly arranged; used in local search methods such as hill climbing.

## completeness

A property of an inference algorithm: it is complete if it can derive any sentence that is entailed by the knowledge base. For a search algorithm, completeness is the guarantee of finding a solution whenever one exists (and correctly reporting failure when none exists).

## completing the square

An algebraic technique for rewriting a quadratic expression as a squared term plus a residual independent of the variable. In AIMA it is used to simplify Gaussian integrals (e.g., in Kalman filter derivations) by pulling the residual outside the integral.

## completion

For a logic program, the completion (Clark's completion) is a logical assertion that the program's clauses give not just sufficient but necessary conditions for each predicate — effectively turning each rule's "if" into "if and only if." This provides the semantic basis for negation as failure.

## Complex sentences

Sentences in propositional (or first-order) logic built from simpler sentences using logical connectives such as negation, conjunction, disjunction, implication, and biconditional.

## complexity analysis

The study of the inherent difficulty of computational problems (rather than particular algorithms), notably classifying problems by whether they can be solved in polynomial time.

## compliant motion

Robot motion performed while maintaining contact with a surface or object, so that the robot yields to (complies with) the motion constraints imposed by the contact rather than following a purely free-space trajectory. AIMA's example is peg-in-hole assembly, where the robot slides along the surface to reduce uncertainty and guide the peg into the hole.

## composition

The combination of two operations (or substitutions) so that the result is equivalent to applying one and then the other. For substitutions, the composition of θ₁ and θ₂ satisfies SUBST(COMPOSE(θ₁, θ₂), p) = SUBST(θ₂, SUBST(θ₁, p)). More generally, it refers to building complex structures or meanings from parts.

## compositional semantics

The principle that the meaning (semantics) of a phrase is a function of the meanings of its subphrases and the way they are syntactically combined. AIMA's grammar rules obey this: e.g., the meaning of "3 + (4 ÷ 2)" is built up by applying operator meanings to the meanings of the constituent expressions.

## compositionality

A property of a representation language whereby the meaning of a sentence is a function of the meanings of its parts. Propositional logic is compositional: the meaning of "S1,4 ∧ S1,2" is systematically related to the meanings of "S1,4" and "S1,2."

## computable

A function is computable if it can be computed by an effective procedure — identified, via the Church-Turing thesis, with the functions computable by a Turing machine. Some functions (e.g., the halting problem) are not computable by any algorithm, and are termed noncomputable.

## computational graphs

Data structures used to describe neural networks, representing the sequence of operations that transform inputs into outputs.

## computational learning theory

The field, at the intersection of AI, statistics, and theoretical computer science, that studies how many examples are needed for learning, based on the idea that a seriously wrong hypothesis will likely be exposed by a small number of examples (leading to the PAC framework).

## computational linguistics

The hybrid field, born from the intersection of modern linguistics and AI, concerned with getting computers to process and understand natural language; often used synonymously with natural language processing. AIMA notes that understanding language requires understanding subject matter and context, not just sentence structure.

## computational neuroscience

The field that builds computational and mathematical models capturing important, specific properties of actual biological neural systems and how information is processed in nervous systems — as opposed to artificial neural networks designed purely from computational or statistical concerns.

## computer vision

The capability that enables a computer to perceive objects and interpret its visual environment.

## conclusion

In an implication of the form "premise ⇒ conclusion," the conclusion is the consequent — the part asserted to hold when the premise is true. (The premise/antecedent is the "if" part; the conclusion/consequent is the "then" part. In a definite/Horn clause it is the single positive literal, the head.)

## concurrent action list

In multiagent/multieffector planning, a component of an action schema stating which actions must or must not be executed at the same time as a given action (a concurrent action constraint). For example, it can specify that two hands must act concurrently to carry a heavy object, or that two actions must not co-occur.

## condition-action rule

A rule used by a simple reflex agent that links a perceived condition directly to an action; also called a situation-action rule, production, or if-then rule.

## conditional effect

An effect in an action schema that applies only under a stated condition, written "when condition: effect." The condition is tested against the current state, and its effect is added only if the condition holds; unlike a precondition, an unsatisfied condition simply omits that effect rather than making the action inapplicable, allowing a single schema to have context-dependent results.

## conditional Gaussian

In a hybrid Bayesian network, the distribution of a continuous variable given its parents: for each assignment to the discrete parents, the continuous variable is Gaussian with a mean that is a linear function of the continuous parents (and a fixed variance). A network built from these is a linear-Gaussian / conditional Gaussian model.

## conditional independence

The property that two variables are independent of each other given knowledge of a third variable, even if they are not unconditionally independent.

## conditional plan

A solution to a problem in partially observable or nondeterministic environments that specifies actions to take depending on percepts received during execution; also called a contingency plan or strategy.

## conditional probability table

The local probability information (CPT) attached to a discrete node in a Bayesian network. Each row gives the conditional probability of the node's values for one conditioning case (a specific combination of values of the parent nodes), and each row must sum to 1; a node with k Boolean parents has 2^k rows.

## conditional random field

A discriminative, undirected probabilistic model (a Markov network) that directly represents the conditional distribution of a set of output/target variables given observed inputs, rather than modeling the joint distribution. AIMA highlights the linear-chain CRF for sequence-labeling tasks such as named-entity recognition and part-of-speech tagging.

## conditioning

A rule, derived using the product rule, that expresses a probability by summing over conditional probabilities weighted by the probabilities of the conditioning variables — i.e., updating or computing a distribution given observed evidence. Together with marginalization it is a widely useful tool for manipulating probability expressions.

## conditioning case

A particular combination of values for a node's parent variables; each row of a conditional probability table gives the conditional probabilities for one conditioning case.

## configuration space

A representation of a robot's state in terms of the parameters that define its configuration, making motion planning easier; it divides into free space (attainable configurations) and occupied space (unattainable ones).

## confirmation

In AIMA's discussion of philosophy, confirmation theory (developed by Carnap and Hempel) attempts to analyze how knowledge is acquired from experience by quantifying the degree of belief that should be assigned to a hypothesis based on observations that confirm or disconfirm it; in probabilistic terms, evidence confirms a hypothesis when it raises the hypothesis's probability.

## confirmation theory

The theory of Carnap and Hempel that sought to analyze how knowledge is acquired from experience by quantifying the degree of belief that should be assigned to logical sentences given evidence.

## conflict set

For a variable in a CSP, the set of previously assigned variables that are connected by constraints to that variable and thereby ruled out its possible values (caused its assignment to fail). Backjumping uses it to jump back to the most recent variable in the set; it can be accumulated cheaply by forward checking.

## conflict-directed backjumping

A backjumping algorithm that uses a deeper notion of conflict set: for a variable, the set of preceding variables that caused it (together with any subsequent variables) to have no consistent solution. These conflict sets are propagated back from later variables to earlier ones, so the search backtracks directly to the true source of the failure.

## conformant

A synonym for "sensorless": a conformant (sensorless) problem is one in which the agent receives no percepts, so it must find a plan that reaches the goal regardless of its actual state, working over belief states rather than physical states. Sometimes a conformant plan is preferable even when sensing is available.

## conjugate gradient

Not found as a defined term in the book. In standard AIMA usage, conjugate gradient is an iterative optimization method that minimizes a function by searching along a sequence of mutually conjugate directions, converging faster than plain gradient descent for well-behaved (quadratic-like) objectives.

## conjugate prior

A prior distribution that, when combined with data from a given family of likelihoods, yields a posterior in the same family (for example, the beta family is the conjugate prior for a Boolean variable's parameter).

## conjunct ordering

The problem, arising in first-order forward/backward chaining, of finding an ordering in which to solve the conjuncts of a rule's premise so that the total cost of matching is minimized. Finding the optimal order is NP-hard, but heuristics such as minimum-remaining-values help.

## conjunction

A sentence whose main logical connective is ∧ (and); its component parts are called conjuncts. It is true exactly when all of its conjuncts are true.

## conjunctive normal form

A sentence expressed as a conjunction of clauses, where each clause is a disjunction of literals. Every propositional sentence is logically equivalent to some sentence in CNF, and there is a standard procedure to convert any sentence into it (this extends to first-order logic); CNF is required for resolution.

## connectionist

Describing the neural-network ("parallel distributed processing") approach that reemerged in the mid-1980s with back-propagation, in which knowledge is represented by patterns of connection strengths among simple interconnected units rather than explicit symbolic rules. Connectionist models were seen as competitors to symbolic AI and can form internal concepts in a fluid, imprecise way by adjusting parameters to reduce prediction error.

## consciousness

The issue underlying debates about "strong AI": awareness of the outside world and of the self, and the subjective experience of living (the intrinsic character of experiences being termed qualia). AIMA notes it remains ill-defined and (following Turing) argues its resolution is not necessary for building machines that behave intelligently.

## consequentialism

The moral stance AI generally adopts: an agent's behavior is evaluated by its consequences. As an agent acts, its actions drive the environment through a sequence of states, and the desirability of that sequence (captured by a performance measure) determines how well the agent performed.

## conservative approximation

An approximate belief-state representation that forms a simple outer envelope around the exact belief state, guaranteed to include all truly possible states.

## consistency

A property of a heuristic (slightly stronger than admissibility): h is consistent if for every node n and every successor n′ reached by action a, h(n) ≤ c(n, a, n′) + h(n′) — a form of the triangle inequality. Every consistent heuristic is admissible, and A* with a consistent heuristic is cost-optimal. (A set of logical sentences is likewise consistent if some model makes them all true.)

## consistency regularization

A semi-supervised learning technique that encourages a model to produce equivalent outputs when given equivalent (e.g., slightly perturbed) inputs.

## consistent

In CSPs, an assignment is consistent (or legal) if it violates none of the constraints. (Also used of heuristics — see "consistency" — where a consistent heuristic satisfies the triangle-inequality condition and never violates monotonicity.)

## consistent plan

Not found as a defined term in the book. In standard AIMA usage, from partial-order planning, a consistent plan is one with no contradictions in its ordering constraints (no cycles) and no unresolved conflicts with its causal links (threats); a consistent plan with no open preconditions is a solution.

## constrained optimization

Optimization of an objective function subject to constraints on the allowable values of the variables.

## constrained optimization problem

A constraint satisfaction problem that also includes preference constraints, so that a solution is sought which best satisfies the preferences; abbreviated COP.

## constraint graph

A graph representing a CSP in which nodes correspond to variables and edges connect any two variables that participate in a constraint.

## constraint hypergraph

A representation of a CSP with higher-order constraints, consisting of ordinary nodes for variables and hypernodes representing n-ary constraints that involve several variables.

## constraint language

Not found as a defined term in the book (it appears only in a bibliography reference to the cc(FD) constraint language). In standard AIMA usage, a constraint language is the formal notation used to express the allowed constraints in a CSP or constraint-logic-programming system — the special relations and operators, such as Alldiff or linear inequalities, that a given solver can handle.

## constraint learning

The technique of, upon reaching a contradiction, finding a minimum subset of the conflict set responsible for the failure — a set of variables and their values called a no-good — and recording it (as a new constraint or in a cache) so the same conflict is never explored again elsewhere in the search.

## constraint logic programming

An extension of logic programming (CLP) that allows variables to be constrained rather than merely bound to values. A CLP solution is the most specific set of constraints on the query variables derivable from the knowledge base (ordinary logic programming being the special case where solution constraints are equalities/bindings). CLP systems incorporate constraint-solving algorithms (e.g., linear programming) for their constraint language.

## constraint optimization problem

A constraint satisfaction problem augmented with preferences encoded as costs on variable assignments (AIMA also calls it a constrained optimization problem, COP), solved by optimizing an objective function rather than merely finding any consistent assignment. Linear programs are one class of COPs.

## constraint propagation

A type of inference in CSPs that uses the constraints to reduce the number of legal values for a variable, which can in turn reduce the legal values for other variables, and so on. It works by enforcing local consistency to eliminate inconsistent values throughout the constraint graph, and can be interleaved with search or done as preprocessing.

## constraint satisfaction problem

A problem given a factored representation and defined by three components: a set of variables X, a domain of possible values for each, and a set of constraints C specifying allowable value combinations. A solution is a complete, consistent assignment; CSP algorithms exploit state structure and general (rather than domain-specific) heuristics.

## constraint weighting

A local-search technique that concentrates effort on the important constraints: each constraint carries a numeric weight (all starting at 1); the algorithm changes the variable/value pair that most lowers the total weight of violated constraints, then increments the weight of each currently violated constraint. This adds topography to plateaus and lets the search learn which constraints are difficult.

## consumable

A category of resource in scheduling/planning: a consumable resource is one that is used up and no longer available after use (such as fuel, money, or bolts), in contrast to a reusable resource that is freed again after use.

## context-free grammar

A grammar in which any rule can be used in any context — the rules for a phrase (e.g., a noun phrase) are the same regardless of where in a sentence it occurs. It defines a language as the set of terminal strings derivable from the start symbol and is widely used to describe programming languages and (approximately) natural languages; AIMA often uses the probabilistic form (PCFG).

## context-specific independence

Also called CSI. A conditional distribution exhibits context-specific independence if a variable is conditionally independent of some of its parents given certain (specific) values of the other parents, rather than for all values. Like determinism, its presence in a Bayes net can be exploited to represent distributions more compactly and to speed up inference.

## contingency plan

The solution to a problem in a partially observable or nondeterministic environment: rather than a fixed sequence of actions, it is a conditional plan (also called a strategy) with branches specifying what to do depending on the percepts received during execution.

## continuous

Describing a variable, state, or environment that can take on a range of values varying smoothly (e.g., the real numbers) rather than a finite or discrete set. AIMA lists discrete-vs-continuous as a key environment dimension; continuous state and action spaces (such as a robot's position and joint angles) require different techniques than discrete ones.

## continuous domains

Domains for CSP variables that take on continuous (real-valued) rather than discrete values; common in real-world problems and studied heavily in operations research.

## contraction

A function of one argument that, when applied to two different inputs, brings their outputs closer together by at least a constant factor than the inputs were. A contraction has a single fixed point; the Bellman update is a contraction, which guarantees that value iteration converges to a unique fixed point.

## contradiction

A sentence that is false in every model (unsatisfiable). It underlies proof by contradiction (reductio ad absurdum / refutation): one assumes a sentence is false and shows this yields a contradiction (the empty clause), proving the query is entailed. In CSP search, reaching a contradiction (a dead end) triggers techniques such as backjumping.

## control theory

The field, developed by Wiener, Ashby, and others, concerned with designing artifacts (controllers) that operate under their own control via feedback so as to behave optimally — typically minimizing an error signal or cost function over time. It addresses continuous variables and provides mathematical tools closely related to AI's treatment of agents.

## controller

A system or component that maps sensed state (or error) into action so as to keep a dynamical system in a desired regime. AIMA identifies the control-theory notion of a controller with the AI notion of an agent; controllers may be based on feedback laws (e.g., PD/PID) or on optimizing a cost function, and correspond to the policy an agent uses to select actions.

## convention

An arbitrary but agreed-upon regularity of behavior that coordinates a group of agents as long as all adopt it — such as which side of the road to drive on. Conventions can arise from established practice or be designed; when widespread they are called social laws, and language itself is a system of conventions. Absent a convention, agents can instead coordinate through communication.

## convex optimization

A general class of constrained optimization in which the constraint region is any convex region and the objective is a function that is convex within that region; linear programming is a special case.

## convex set

A set of points such that the straight line segment joining any two of its members lies entirely within the set. Convexity of a feasible region or objective (convex optimization) is important for guaranteeing efficient optimization, since convex functions have no local minima distinct from the global minimum; linear-programming constraints form a convex set.

## convolution

In a convolutional neural network, the operation of applying a kernel — a pattern of weights replicated across many local regions — to the pixels of an image (or to spatially organized units in a later layer), taking a weighted sum of neighboring values at each position. The same kernel is applied at all positions, reflecting spatial invariance. AIMA notes that in signal-processing terms this operation is technically a cross-correlation.

## cooperative

Describing a multiagent situation in which agents share a common (or aligned) goal and can benefit by working together, in contrast to a competitive setting. In a cooperative game agents can make binding agreements (enforced by contracts, social norms, or inspectable source code); a situation can be partly cooperative and partly competitive (e.g., cars avoiding collisions while competing for road space).

## coordination

The problem, in multiagent planning, of getting multiple agents to act coherently — "pulling in the same direction" — so their joint plan succeeds without accidentally interfering with one another. Approaches include agreeing on a joint plan in advance, adopting conventions, communicating, and plan recognition.

## corpus

A large body of text (or speech) used as data for building and evaluating language models and other NLP systems and for gathering language statistics. A corpus typically contains at least a million words and tens of thousands of distinct words, with modern corpora reaching billions of words (e.g., Wikipedia, the iWeb corpus).

## correlated sampling

A technique underlying policy-search algorithms (such as Pegasus) that reuses the same random samples across candidate policies so that comparisons reflect genuine differences rather than sampling noise.

## count nouns

Nouns that denote individually countable things, such as aardvarks, holes, and theorems, in contrast to mass nouns like butter or water.

## Cournot competition

A model of duopoly/oligopoly (due to Cournot) in which competing firms independently and simultaneously choose the quantities (or prices) of a product from a continuous range; the market price depends on total output, and the outcome is analyzed as a Nash equilibrium. AIMA cites it as an example of extending game theory to continuous action spaces.

## covariance

The expectation of the product of two random variables' deviations from their means, cov(X, Y) = E[(X − μ_X)(Y − μ_Y)], measuring how much two variables vary together: positive when they tend to increase together, negative when one rises as the other falls, and zero when there is no linear relationship.

## covariance matrix

A matrix (often denoted Σ) whose (i, j) entry is the covariance between the i-th and j-th components of a vector of random variables, with variances on the diagonal. It parameterizes the spread and orientation of a multivariate Gaussian distribution.

## critic

The component of a learning agent that tells the learning element how well the agent is doing with respect to a fixed performance standard, by observing the percept stream. It is essential because the percepts themselves give no indication of success; the learning element uses this feedback to decide how to modify the performance element.

## critical path

The path through the directed graph of a partial-order plan whose total duration is longest, and which therefore determines the overall duration of the whole plan. Delaying any action on the critical path delays the entire schedule, so these actions have zero slack.

## critical path  method

A technique (CPM) applied to the directed graph of ordering constraints in a partial-order plan to compute the earliest and latest possible start times (ES, LS) of each action, thereby identifying the critical path; the window between ES and LS is the action's slack, and the zero-slack actions form the critical path.

## critical path method

A technique (CPM) applied to a plan's schedule graph to determine the possible start and end times of each action; the critical path is the longest-duration path, which determines the total plan duration.

## cross-correlation

Not defined as a distinct term in the book. In standard AIMA usage it denotes the sliding dot-product operation of a kernel/template against a signal or image — computationally the same pattern as convolution but without flipping the kernel. AIMA notes that the operation neural networks call "convolution" is technically cross-correlation in signal-processing terms.

## cross-validation

A method for estimating how well a hypothesis will generalize by splitting the available data into a training set to form the hypothesis and a test set to evaluate it; variants include k-fold and leave-one-out cross-validation.

## crossover point

In a genetic algorithm's recombination step, the randomly selected position at which two parent strings are split so their pieces can be recombined, each child taking the substring before the point from one parent and the rest from the other.

## cryptarithmetic

A class of puzzles used to illustrate constraint satisfaction, in which the letters of an arithmetic equation (e.g., TWO + TWO = FOUR) stand for distinct digits, and the task is to find the digit assignment making the arithmetic correct (with no leading zeroes). It is typically modeled with an Alldiff global constraint plus column-addition constraints using auxiliary carry variables.

## cumulative distribution

A function that gives the probability that a random variable takes a value less than or equal to a given point, obtained by accumulating (summing or integrating) the probability up to that value. For a continuous variable, the probability density function is its derivative.

## cumulative probability density function

Not defined as a distinct term in the book, which uses "cumulative distribution" / "cumulative distribution function." Consistent with AIMA usage, it refers to the function giving P(X ≤ x) for a continuous random variable — the integral of the probability density up to x, increasing monotonically from 0 to 1 — with the density being its derivative.

## current-best-hypothesis

Not found as a defined term in the book. In standard AIMA usage (from concept learning), current-best-hypothesis search maintains a single hypothesis and adjusts it as new examples arrive — generalizing it to cover a false negative and specializing it to exclude a false positive — so that it remains consistent with all examples seen so far.

## curriculum learning

A learning approach that begins with easy examples and progressively introduces harder ones, so that knowledge gained on simple cases enables the system to handle more complex ones.

## curse of dimensionality

The phenomenon whereby the difficulty of problems (such as nearest-neighbor methods) grows dramatically as the number of dimensions increases, since data become sparse in high-dimensional spaces.

## cutoff test

A function that replaces the terminal test in a game search; it returns true for terminal states but may also stop the search earlier based on depth or state properties, at which point an evaluation function estimates utility.

## cutset conditioning

An algorithmic approach that identifies a set of variables (a cycle cutset) whose removal makes the remaining graph acyclic, then reasons about each assignment to that set; used in CSPs and in Bayesian network inference.

## cybersecurity

The domain of protecting computer systems against attack, in which machine learning serves as a tool for both attackers and defenders.

## cycle

A path in a search tree that returns to a previously visited state, generating a repeated state; also known as a loopy path and a special case of a redundant path.

## cycle cutset

A subset of a CSP's variables chosen so that removing them (and their constraints) leaves the remaining constraint graph as a tree, which can then be solved efficiently. The algorithm tries each assignment to the cutset variables and solves a tree problem for each; finding the smallest cycle cutset is NP-hard but has good approximation algorithms. This technique is called cutset conditioning.

## cyclic solution

A solution to a nondeterministic (AND-OR search) problem that contains a loop — for example, "keep trying Right until it works" in a slippery vacuum world. It is a valid solution when every leaf is a goal state and every leaf is reachable from every point in the plan, so the loop will eventually escape to a goal (assuming the action eventually produces the needed outcome).

## CYK algorithm

A probabilistic bottom-up chart-parsing algorithm (named after Cocke, Younger, and Kasami) that, given a sequence of words and a grammar in Chomsky Normal Form, finds the most probable parse tree for the sequence and its subsequences. It fills a table of possible constituents over substrings, running in O(n³·|grammar|) time and O(n²·|grammar|) space.

# D

## DARPA Grand Challenge

A prize competition organized by DARPA in 2004 and 2005 in which autonomous vehicles had to travel more than 200 kilometers across the desert in under ten hours. In 2004 no robot got past eight miles; in 2005 Stanford's robot Stanley won. It, and the follow-on Urban Challenge, spurred major advances in self-driving cars.

## data association

In multiple-object tracking, the problem of associating observation data with the objects that generated them, when it is uncertain which observation came from which object. Originally studied for radar tracking of multiple targets, where blips at successive time steps must be matched to tracks.

## data complexity

The complexity of inference measured as a function of the number of ground facts in the knowledge base, holding rule sizes and predicate arities bounded. For example, the data complexity of forward chaining on a Datalog knowledge base is polynomial rather than exponential.

## data compression

The encoding of data using fewer bits than the original representation, so it occupies less space; in AIMA, learning a good model of data (e.g., predicting the next symbol) is closely tied to compressing it.

## data matrix

The matrix of inputs used in learning, with one example per row (each row a feature/attribute vector). It is a standard input format for algorithms such as linear regression, where the optimal weights can be computed analytically from the data matrix.

## data mining

The application of machine learning and statistical methods to discover useful patterns and regularities in large collections of data, typically stored in databases.

## data sheets

Standardized documents that accompany a component (analogous to the data sheets shipped with electronic parts like resistors) and specify its intended use and operating characteristics, so that designers can judge whether their planned application is feasible.

## data-driven

Describes reasoning in which the focus of attention starts with the known data (facts or percepts) and proceeds forward toward conclusions, often without a specific query in mind. Forward chaining is the canonical example of data-driven reasoning.

## data-oriented parsing

A parsing approach that learns to parse by drawing on a corpus of already-parsed sentence/tree pairs (a treebank), reusing fragments of prior analyses rather than relying solely on a hand-written grammar.

## database semantics

A semantics for logical sentences based on three assumptions: the unique-names assumption (distinct constant symbols name distinct objects), the closed-world assumption (atomic sentences not known to be true are false), and domain closure (the only objects are those named by the constants). It is popular in database and logic programming systems and differs from standard first-order semantics.

## Datalog

A language consisting of first-order definite clauses with no function symbols. It represents the kinds of statements typically made in relational databases, and the absence of function symbols makes inference much easier (and decidable).

## dataset augmentation

A technique for improving learning (especially in object recognition) in which existing training examples are copied and slightly modified to create additional training data.

## Davis-Putnam algorithm

A complete backtracking algorithm for deciding the satisfiability of a propositional sentence in conjunctive normal form. Named after Davis and Putnam's 1960 paper, though the commonly used version (with the additions of Davis, Logemann, and Loveland) is called DPLL; it improves on simple model enumeration with early termination, pure-symbol, and unit-clause heuristics.

## de-identification

The practice of sharing records, such as health data, only after stripping out personally identifying information (e.g., names and social security numbers) so the data can be used by researchers without exposing individuals.

## decayed MCMC

An MCMC filtering algorithm for temporal models that, to avoid growing update times as the network is unrolled, prefers to sample more recent state variables, with a sampling probability that decreases for variables further in the past.

## decentralized planning

The situation in a multibody problem where communication constraints prevent the bodies from pooling sensor data into a common world estimate. The planning phase is still centralized, but execution is decoupled, so each body's subplan may need to include explicit communicative actions with the others.

## decision analysis

A field, drawing on decision theory, that emerged to make policy decisions more rational in areas such as military strategy, medical diagnosis, and engineering design. The process involves a decision maker who states preferences among outcomes and a decision analyst who enumerates actions and outcomes and elicits preferences to determine the best course of action.

## decision analyst

In decision analysis, the person who enumerates the possible actions and outcomes and elicits the decision maker's preferences among them, in order to determine the best course of action.

## decision boundary

A line (or, in higher dimensions, a surface) that separates one class of examples from another in the input space. When the boundary is a straight line (a hyperplane) it is called a linear separator, and data admitting such a separator are linearly separable.

## decision lists

A hypothesis space consisting of a series of tests, each a conjunction of literals; each test is checked in order, and the first one that succeeds determines the classification.

## decision maker

In decision analysis, the person or agent who states preferences among outcomes; distinguished from the decision analyst, who structures the problem by enumerating possible actions and outcomes and eliciting the decision maker's preferences.

## decision network

A formalism, also called an influence diagram, that extends Bayesian networks by incorporating action (decision) nodes and a utility node, so that rational decisions can be computed by maximizing expected utility.

## Decision nodes

Nodes in a decision network (drawn as rectangles) that represent the points at which the decision maker chooses among the available actions.

## decision theory

The framework that combines probability theory with utility theory to provide a formal and complete method for making individual decisions under uncertainty. A rational agent is one that chooses the action that maximizes its expected utility.

## decision tree

A representation of a function that maps a vector of attribute values to a single output value ("a decision"). It reaches its decision by a sequence of tests: each internal node tests one input attribute, branches are labeled with attribute values, and each leaf specifies the value to return.

## decision tree pruning

A technique that combats overfitting in decision trees by eliminating nodes that are not clearly relevant: test nodes whose subtrees only detect noise are replaced with leaf nodes.

## decision-theoretic metareasoning

Reasoning about one's own computations by treating them as actions whose costs and benefits can be evaluated, choosing which computations to perform so as to maximize expected utility; a specific form of reflective reasoning.

## declarative

Describes the approach to system building in which the agent operates by knowing things—knowledge is expressed as explicit sentences in a representation language and used by a general reasoning process. It contrasts with the procedural approach, which encodes desired behaviors directly as program code.

## declarative bias

Prior knowledge in inductive learning that is stated explicitly (for example, as restrictions on the allowable form of hypotheses) so as to constrain the hypothesis space and guide the learner, rather than being built implicitly into the algorithm.

## decoder

In a sequence-to-sequence model, the RNN (or network) that takes an internal representation—typically the final hidden state produced by an encoder—and generates the output sequence from it.

## decomposition

The technique of breaking a problem into smaller subproblems whose solutions can be combined. In hierarchical planning, a high-level action is decomposed into a sequence of lower-level actions.

## deduction theorem

The classical result that, for any sentences alpha and beta, alpha entails beta if and only if the implication (alpha implies beta) is valid. It connects the notions of entailment and logical validity, so that every valid implication describes a legitimate inference.

## deductive databases

An approach, arising from the merging of logic-based knowledge representation with database systems, that stores facts and rules and answers queries by logical deduction over them.

## deductive learning

Learning that goes from a known general rule to a new rule that is logically entailed by it. The result contains nothing logically new, but is useful because it allows more efficient processing (as in explanation-based learning).

## deductive synthesis

The construction of a program by logically deducing it from a formal specification, so that the resulting program is provably correct with respect to that specification.

## deep belief networks

Generative neural network models built from stacked layers of stochastic latent variables (often restricted Boltzmann machines) that can be trained one layer at a time in an unsupervised fashion; they were influential in the early revival of deep learning.

## deep learning

A broad family of machine learning techniques in which hypotheses take the form of complex algebraic circuits with tunable connection strengths. "Deep" refers to the many layers into which the circuits are organized, so that computation paths from inputs to outputs have many steps. It is currently the dominant approach for tasks such as visual object recognition, machine translation, and speech recognition.

## default logic

A formalism for nonmonotonic reasoning in which default rules generate contingent conclusions of the form "if the premise holds and it is consistent to assume the conclusion, then conclude it," so that plausible conclusions may be retracted when new information arrives.

## default rules

Rules in default logic that license contingent, nonmonotonic conclusions—conclusions that hold by default in the absence of contrary information but may be withdrawn when new facts are learned.

## definite clause

A disjunction of literals of which exactly one is positive; equivalently, an implication whose premise is a conjunction of positive literals and whose conclusion is a single positive literal. Forward and backward chaining operate on knowledge bases of definite clauses.

## definite clause grammar

A grammar formalism in which grammar rules are written as definite (Horn) clauses, typically within a logic programming system, so that parsing and generation can be carried out by logical inference.

## definition of a rational agent

The characterization of a rational agent as one that acts so as to achieve the best outcome, or, when there is uncertainty, the best expected outcome—selecting for each percept sequence an action expected to maximize its performance measure given the evidence and its built-in knowledge.

## deformable template

A model for object recognition in which an object's shape is represented by a template that can be flexibly stretched or bent to match the variations in appearance of different instances.

## degree heuristic

A variable-ordering heuristic for CSPs that selects the variable involved in the largest number of constraints on other unassigned variables, aiming to reduce future branching; often used as a tie-breaker for the minimum-remaining-values heuristic.

## degree of belief

The subjective probability an agent assigns to a proposition, expressing its confidence (on a scale from 0 to 1) that the proposition is true. It should not be confused with degree of truth in fuzzy logic.

## degree of freedom

Each independent direction in which a system's configuration can be varied. For a robot, the number of degrees of freedom is the number of independently movable joints; more generally, the number of parameters that can be varied subject to the problem's constraints.

## delete list

In a planning action described in the STRIPS style, the set of fluents that are removed from the state when the action is executed (the negative literals of the action's effects), written DEL(a); the complementary set that is added is the add list.

## deliberative layer

The higher-level component of a layered agent architecture that plans and reasons about goals over longer horizons. This deliberative view of robotics—defining or learning a reward and optimizing it—contrasts with a reactive view that maps percepts directly to actions.

## delta rule

Another name for the Widrow--Hoff rule, a gradient-descent weight-update rule for training a linear unit that adjusts weights in proportion to the error between predicted and target output.

## demonic nondeterminism

Nondeterminism in which an adversary chooses the outcomes of the agent's actions in the worst possible way, so a plan must succeed no matter which outcome occurs. It contrasts with angelic nondeterminism, in which the agent itself gets to choose the outcomes.

## Dempster-Shafer theory

A theory of evidence, originating with Dempster's generalization of probability to interval values and Shafer's later work, that represents uncertainty using belief functions and distinguishes an agent's degree of belief in a proposition from its plausibility, thereby modeling ignorance explicitly.

## density estimation

The general task of learning a probability model from data assumed to be generated by that model; a form of unsupervised learning that applies to both continuous densities and discrete distributions.

## dependency grammar

A syntactic formalism that represents sentence structure through binary relations (dependencies) between individual words, rather than through hierarchical phrase-structure categories.

## dependency-directed backtracking

A backtracking technique that combines backjumping with no-good learning, recording the reasons for conflicts so the search can jump back to a relevant decision; it led to the development of truth maintenance systems.

## depth

For a node in a search tree, the number of actions along the path from the root node to that node.

## depth of field

In imaging, the range of distances from the camera over which objects appear acceptably sharp; objects outside this range are blurred. The larger the lens aperture, the smaller the depth of field.

## depth-first search

A search strategy that always expands the deepest node in the frontier first. It is usually implemented as a tree-like search that does not keep a table of reached states, giving it low memory cost but the risk of descending infinite paths.

## depth-limited search

A version of depth-first search in which a depth limit is supplied and all nodes at that depth are treated as if they had no successors. It avoids infinite descent, but may fail to find solutions that lie deeper than the limit.

## descriptive theory

A theory that describes how agents (such as people) actually behave, in contrast to a normative theory that prescribes how a rational agent ideally should behave.

## detailed balance

A condition on a Markov chain stating that the expected flow of probability between any pair of states is equal in both directions. Detailed balance implies stationarity of the distribution and is exploited by MCMC methods such as Metropolis-Hastings.

## Deterministic

Describes an environment (or transition model) in which the next state is completely determined by the current state and the action executed by the agent, with no randomness; it contrasts with a stochastic environment.

## deterministic nodes

The simplest kind of node with a canonical distribution in a Bayesian network: a node whose value is specified exactly (with no uncertainty) by the values of its parents, via a logical or numerical relationship.

## deterministic parser

A parser that makes an irrevocable choice at each step rather than exploring alternatives; equivalently, a beam-search parser with beam width b = 1, as in shift-reduce parsing.

## detour index

A multiplier applied to straight-line distance to account for the typical curvature of roads; used by road engineers to obtain more accurate distance estimates, and generalizable as a way to strengthen heuristics.

## diachronic

Pertaining to relationships that hold across time. A diachronic sentence links variables at different time steps, as in the transition model of a temporal probability model; it contrasts with a synchronic (same-time) relationship.

## diagnostic

Describes reasoning or a probabilistic relationship that runs from observed effects (evidence, symptoms) back to their causes—the diagnostic direction—as opposed to the causal direction that runs from causes to effects. Diagnostic knowledge is often more fragile than causal knowledge.

## diameter

The maximum, over all pairs of states, of the shortest-path distance between them in a state-space graph. It bounds how far apart any two states can be and can supply a good depth limit for depth-limited search.

## dictionary

A structured word model, usually built by hand, that lists words and information about them; for example, WordNet is an open-source, hand-curated machine-readable dictionary widely used in NLP.

## differentiable programming

A proposed generalization of deep learning (a term suggested by Yann LeCun) in which ordinary programming languages and machine learning models are merged, so that programs built from differentiable components can be optimized by gradient methods.

## Differential drive

A common mobile-robot drive arrangement using two independently controlled wheels: driving them at equal speeds moves the robot straight, while driving them at different speeds turns it, giving control of both translation and rotation.

## Differential GPS

An enhancement to satellite positioning in which a second ground receiver at a known location provides corrections, improving localization accuracy from a few meters (ordinary GPS) to millimeters under ideal conditions.

## differential heuristic

An admissible heuristic defined using precomputed distances to landmark points, computed as the maximum over landmarks of the absolute difference between the cost from the node to a landmark and the cost from the goal to that landmark (hence the subtraction).

## differential privacy

A strong privacy guarantee ensuring that an attacker cannot use database queries to re-identify any individual, even when making many queries and holding separate linking databases.

## diffuse albedo

The fraction of incoming light that a diffuse surface patch reflects (equally in all directions), determining its perceived matte brightness or color; for practical surfaces it lies roughly in the range 0.05 to 0.95.

## diffuse reflection

The way most surfaces reflect light: incoming light is scattered evenly in all directions so that the surface appears equally bright from any viewing angle, in contrast to the specular reflection of a mirror.

## Diophantine equations

Polynomial equations over integer domains, for which only integer solutions are sought. They are named after Diophantus, and determining whether general such equations have solutions is undecidable.

## direct utility estimation

A passive reinforcement learning method in which the utility of a state is estimated as the average observed expected reward-to-go (total reward from that state onward) over the trials that pass through it, treating utility learning as supervised regression.

## directional arc consistency

A weaker notion of arc consistency (abbreviated DAC) defined relative to a fixed variable ordering: every variable is arc-consistent only with the variables that come after it in the ordering.

## Dirichlet process

A distribution over probability distributions—describable as a distribution over Dirichlet distributions—used as a prior in nonparametric Bayesian methods, allowing the number of components (e.g., clusters in a mixture) to be unknown and inferred from data.

## disambiguation

In natural language understanding, the process of recovering the most probable intended meaning of an utterance from among its possible interpretations, using context and world knowledge.

## discount factor

A number gamma between 0 and 1 that describes an agent's preference for current rewards over future ones in a sequential decision problem. Values near 0 make distant rewards insignificant; a value of 1 reduces discounted rewards to purely additive rewards, and it keeps infinite-horizon utilities finite.

## discrete

Describes an environment, time, percepts, or actions that take on a finite (or countable) set of distinct values, as opposed to sweeping smoothly through a continuous range. Chess is discrete; taxi driving is continuous.

## discrete events

Events that have a definite, bounded structure (a clear beginning and end), as opposed to processes or liquid events that hold over intervals and can be arbitrarily subdivided.

## discretization

The technique of handling a continuous quantity by dividing its possible values into a fixed set of intervals, for example so that continuous variables can be given explicit conditional probability tables in a Bayesian network.

## discriminator

In a generative adversarial network (GAN), a second neural network trained as a classifier to distinguish real inputs (drawn from the training set) from fake inputs produced by the generator.

## disjoint

Describes sets or events that share no elements or cannot occur together; mutually exclusive events are disjoint.

## disjoint pattern databases

A set of pattern databases that partition a problem so that each move affects only one subproblem; because their costs count disjoint sets of moves, the databases' values can be added while still yielding an admissible heuristic, greatly reducing search effort.

## disjunction

A logical sentence whose main connective is "or"; it is true when at least one of its disjuncts is true.

## disjunctive constraint

A constraint satisfied if any one of several alternative conditions holds. For example, to keep two tasks that share a tool from overlapping in time, a disjunctive constraint requires that one task come entirely before or entirely after the other.

## disparity

In binocular stereo vision, the shift in the image position of a scene point between the left and right views. The size of the disparity is the reciprocal of the point's depth, so disparity serves as a cue to distance.

## distant point light source

An idealized model of illumination, such as the sun, whose rays travel parallel in a known direction because the source is so far away. The energy a surface patch collects depends on the angle between the illumination direction and the surface normal.

## distortion

A systematic geometric deviation introduced by a lens or imaging process that causes straight lines in the scene to appear curved or otherwise displaced in the image.

## distributed constraint satisfaction

The solving of a constraint satisfaction problem by a collection of agents, each controlling a subset of the variables and communicating to satisfy the shared constraints, rather than by a single central solver.

## distributed representation

A representation in which each concept is spread across many memory locations and each location contributes to representing multiple concepts, in contrast to a localist representation with a one-to-one concept-to-location mapping.

## DL

An abbreviation for description logic, a family of formal languages designed to represent categories, their definitions, and relationships among objects, and to make it easy to express constraints such as "exactly two." Its principal inference tasks are subsumption (is one category a subset of another?) and classification (does an object belong to a category?).

## do the right thing

The guiding goal for a rational agent: to take the action that is expected to best achieve its objectives given what it knows; building agents that "do the right thing" is a central aim of AI.

## domain

The set of possible values a variable may take; in a constraint satisfaction problem, each variable has a domain of candidate values. In first-order logic, the domain of a model is the set of objects it contains.

## domain closure

The assumption (part of database semantics) that each model contains no more domain elements than those named by the constant symbols, so no unmentioned objects are considered.

## domain elements

The objects contained in the domain of a model; the set of things that terms can refer to and over which relations are defined.

## dominant strategy

A strategy for a player that does at least as well as, and in some cases better than, every other available strategy, no matter what the other players do; a rational player is assumed to choose it.

## dominant strategy equilibrium

An outcome of a game in which every player is playing a dominant strategy—one that yields an outcome at least as good as any alternative regardless of the other players' choices—so no player has any incentive to deviate.

## dominating policy

An optimal policy (for example, in a bandit problem) that is at least as good as every other policy in all relevant circumstances, so that following it can never be improved upon.

## doubly block circulant

The structured form of matrix that expresses a 2-D convolution as a matrix multiplication (analogous to a Toeplitz matrix for 1-D convolution); its regular block structure lets a small kernel be applied across an image.

## downward refinement property

A property of a hierarchical planning library guaranteeing that every high-level plan that appears (under its description) to achieve the goal has at least one refinement into primitive steps that actually achieves it, making abstract reasoning sound.

## dropout

A regularization technique for neural networks in which randomly selected units are temporarily removed (their outputs set to zero) during training, reducing overfitting at the cost of making the training data harder to fit.

## dropping conditions

A generalization operation in inductive learning that removes a condition from a rule's premise, thereby making the rule apply to more cases.

## DT

An abbreviation for decision theory, the framework that combines probability and utility to determine the expected-utility-maximizing action under uncertainty.

## dual graph

A transformation that converts a constraint satisfaction problem with n-ary constraints into an equivalent one with only binary constraints: it creates a new graph with one variable per original constraint and a binary constraint between each pair of original constraints that share variables.

## dual use

The property of a technology that has both peaceful and harmful (e.g., lethal) applications, which makes it difficult to regulate or restrict.

## dualism

The philosophical position, held by Descartes, that there is a part of the mind (or soul) that lies outside nature and is exempt from physical laws. It contrasts with materialism, which holds that the operation of the brain according to physical law constitutes the mind.

## duration

The length of time that an action or event takes. In scheduling and temporal planning, actions are annotated with durations so that plans respect timing and resource constraints.

## dynamic

Describes an environment that can change while the agent is deliberating, so that the passage of time matters and the agent must act in a timely way; it contrasts with a static environment. (An environment is semidynamic if it does not change with time but the agent's performance score does.)

## dynamic backtracking

A backtracking search technique that, when backtracking over an earlier variable choice, retains the successful partial assignments made to later variables that were not invalidated by that choice, rather than discarding them.

## dynamic Bayesian network

A Bayesian network that models a temporal probability process as a repeated structure of state and evidence variables across time slices, compactly encoding how the state evolves and how observations depend on it. Hidden Markov models and Kalman filters are special cases.

## dynamic programming

A method for solving a complex problem by breaking it into subproblems and combining their solutions, typically by working backward from the goal and recording the cost of each solution so that it can be reused. It underlies algorithms for pattern databases and for solving Markov decision processes.

## dynamic search problems

Search problems in which the agent must explore an unknown environment and interleave computation with action, so that the map or state space is discovered as the agent moves rather than known in advance.

## dynamic state

The part of a robot's state that includes rates of change such as velocity, in addition to configuration. It is needed to compute what torque to apply, in contrast to the kinematic state, which captures only positions (configuration).

## dynamically stable

Describes a robot that stays upright only while in motion (for example, by hopping or continually adjusting), in contrast to a statically stable robot that remains upright without moving its legs.

# E

## early goal test

A technique in graph search in which a node is checked to see if it is a solution as soon as it is generated, rather than waiting until it is popped off the frontier queue (the "late goal test" used by best-first search).

## early playout termination

A speedup for Monte Carlo simulations in which a playout that is running too long is stopped before reaching a terminal state, and its value is estimated with a heuristic evaluation function or simply declared a draw.

## early stopping

In decision tree learning, the idea of stopping the tree-growing algorithm from generating further nodes once there is no good attribute to split on, rather than growing the full tree and then pruning. The book warns this can fail, because sometimes no single attribute is informative even though a combination is (e.g., the XOR of two attributes), so generate-and-then-prune is preferred.

## economy

One of the stances toward a multiagent environment, appropriate when there are very many agents: they are treated in the aggregate as an economy, which lets one predict things such as "increasing demand causes prices to rise" without predicting any individual agent's action.

## effect

The part of a planning action schema (the EFFECT of an Action schema) that specifies how the state changes: a conjunction of literals, where positive literals are added to the resulting state and negated literals are removed from it.

## effective branching factor

A way to characterize a search heuristic's quality: the branching factor b* that a uniform tree with the same total number of generated nodes and the same solution depth would have. A well-designed heuristic gives a value close to 1.

## effective depth

The average depth of all paths explored during a search; A* pruning reduces the effective depth (as opposed to reducing the effective branching factor).

## efficient

Of an auction (mechanism), the property that the goods go to the agent who values them most, thereby helping to maximize global utility. The book notes the ascending-bid (English) auction is usually efficient, though a badly set reserve price can undermine this.

## electric motor

A common type of actuator/effector used in robots; a device that converts electrical energy into mechanical motion to move robot components. (The specific term is not defined in a margin note in the extracted text; this is standard AIMA usage.)

## eliminative materialism

A philosophical position holding that our ordinary talk of mental states (beliefs, desires, and other folk-psychology notions) does not correspond to anything real, and that such concepts will ultimately be replaced by a mature neuroscience. (Not found as a defined term in the extracted book text; the book does discuss materialism as an alternative to dualism.)

## elitism

A practice in genetic/evolutionary algorithms in which a few top-scoring parents from the previous generation are carried over into the makeup of the next generation, which guarantees that overall fitness will never decrease over time.

## embedding

A learned, meaningful vector-valued representation of a discrete symbol such as a word, mapping it to a low-dimensional dense vector that captures useful features for a neural network.

## embedding lookup layer

The first layer of a neural network applied to discrete tokens, which maps each token to its corresponding learned embedding vector.

## embodied cognition

The approach claiming that it makes no sense to consider the brain separately: cognition takes place within a body embedded in an environment, and the brain's functioning exploits regularities in that environment (including the body). Under this view, robotics, vision, and sensing become central rather than peripheral.

## emergent behavior

Overall behavior of a system that arises from the interactions of many simpler components or agents and is not exhibited by any single component on its own. (Not found as a defined margin-note term in the extracted book text; this is standard AIMA usage.)

## empirical gradient

A method that estimates progress toward an optimum by measuring the change in the objective function's value between two nearby points (rather than using an analytic gradient). Empirical gradient search is equivalent to steepest-ascent hill climbing in a discretized version of the state space.

## empirical loss

The estimate of a hypothesis's generalization loss computed on an actual set of examples of size N; since the true loss over the whole distribution is generally unknown, the learning agent minimizes empirical loss instead.

## empiricism

The view, associated with the empiricism movement beginning with Francis Bacon's Novum Organum and John Locke, that the source of knowledge is sensory experience, captured by the dictum "Nothing is in the understanding, which was not first in the senses."

## encoder

In a sequence-to-sequence model, the (typically RNN) component that reads through the input sequence, producing a hidden representation whose final state is passed to the decoder that generates the output.

## end-to-end learning

An approach in which a single deep learning model represents the entire function needed to solve an application, and all of that function can be learned together from data by backpropagation.

## English auction

The best-known auction mechanism, an ascending-bid auction: bidding starts low and continues upward by some increment until nobody will bid anymore, whereupon the last (highest) bidder wins the item and pays the price bid.

## ensemble learning

A method that selects a collection (ensemble) of hypotheses h1, h2, ..., hn and combines their individual predictions, for example by averaging or voting, to produce a more accurate overall prediction.

## entailment

The relation between sentences whereby one logically follows from another. A knowledge base entails a sentence exactly when the sentence is true in all models (possible worlds) in which the knowledge base is true.

## entropy

A measure of the uncertainty of a random variable; the more information, the less entropy. A variable with a single possible value has zero entropy, a fair coin flip has 1 bit, and a fair four-sided die has 2 bits; in general it is the expected number of bits needed to encode the variable's value.

## environment

Along with the agent, one of the two basic elements of the study of AI: everything outside the agent, which the agent perceives through its sensors (percepts) and affects through its actuators, and in which its performance is measured.

## environment class

A set of related environments from which individual environments are drawn for evaluation; an agent's performance is measured by its average performance across many environments sampled from the class (e.g. a taxi driver evaluated over varied traffic, lighting, and weather).

## environment generator

A component of an experimental setup that generates many task environments (instances) from a class, so that an agent can be evaluated on its average performance across a range of environments rather than on a single one. (Not found as a defined margin-note term in the extracted book text; this is standard AIMA usage.)

## episodic

A property of a task environment (episodic vs. sequential) in which the agent's experience is divided into independent atomic episodes, so that the action taken in one episode does not affect later episodes.

## epsilon-ball

In computational learning theory (PAC learning), the region in hypothesis space around the true function f containing all hypotheses whose error is at most epsilon; such a hypothesis is "approximately correct," and the space outside the ball is denoted H_bad.

## equal impact

A fairness criterion holding that the expected utility of the outcome, given a prediction, should be the same across all groups (e.g. races).

## equality symbol

A symbol in first-order logic used to state that two terms refer to the same object in the world (and, e.g., to assert facts about the values of functions).

## Equational unification

A form of unification in which two terms count as unifiable if they can be proven equal under some substitution using equality reasoning; it is carried out with specialized algorithms tailored to the relevant axioms (such as commutativity and associativity) rather than by explicit inference over those axioms.

## equilibrium

A strategy profile (solution concept) that is a stable point of a game. In a Nash equilibrium no player could unilaterally change their strategy and thereby obtain a higher payoff, given the others' choices, so no rational player has an incentive to deviate.

## ergodic

A property of a Markov chain's transition kernel meaning that every state is reachable from every other state (and there are no strictly periodic cycles), which guarantees the chain converges to a unique stationary distribution independent of the starting state.

## erratic vacuum world

A variant of the vacuum-world problem in which the Suck action behaves nondeterministically: on a dirty square it cleans that square and sometimes an adjacent square too, requiring contingency planning.

## error rate

A measure of a hypothesis's performance: the proportion (or probability) of examples on which it makes an incorrect prediction (its 0/1 generalization loss), i.e., the probability that it misclassifies a new example.

## evaluation function

A function f(n) that estimates the desirability of expanding a node n; best-first search selects the node with the minimum (best) evaluation-function value.

## event

In probability theory, a set of possible worlds (a subset of the sample space, often specified by an assignment to some variables) to which a probability is assigned; more generally, an occurrence located at a point or interval in time.

## event calculus

A formalism for representing time, change, and continuous processes whose basic objects are events, fluents, and time points; by reifying events it allows arbitrary information to be asserted about them.

## evidence

In probabilistic inference, the observed values of a set of variables that are conditioned upon; the prior probability of a query is revised in light of this evidence to yield a posterior (conditional) probability.

## evidence reversal

An improvement to particle filtering (from Kanazawa et al., 1995) in which the state at the next time step is sampled conditional on both the current state and the evidence at the next step, letting the evidence influence sample generation directly and thereby reducing approximation error.

## evolutionary algorithms

Variants of stochastic beam search, explicitly motivated by natural selection, in which a population of individuals (candidate solutions) is repeatedly evaluated by a fitness function, selected, recombined, and mutated to produce offspring over successive generations.

## evolutionary psychology

A field that seeks to explain human cognition and behavior as adaptations shaped by natural selection, interpreting mental traits (such as decision-making tendencies) in terms of their evolutionary function.

## evolutionary strategies

A form of evolutionary algorithm (evolution strategies) in which each individual is a sequence of real numbers rather than a string over a discrete alphabet, and mutation parameters may themselves be adapted during the search.

## exact cell decomposition

A robot motion-planning method that partitions the free space into a finite number of cells whose boundaries exactly match the free space, so that a path can be planned by moving within and between adjacent cells (in contrast to an approximate grid decomposition).

## execution

The phase in which an agent carries out the actions of a chosen plan in the environment, as distinct from the planning/search phase that produced the plan.

## execution monitoring

The process of watching the world while a plan is being carried out to detect when the actual situation diverges from what the plan expected; it is presupposed by replanning, which repairs or regenerates the plan when a discrepancy is found.

## executive layer

In a layered (three-layer) robot control architecture, the middle layer that sits between the deliberative/planning layer and the reactive layer, sequencing and coordinating the primitive activities needed to carry out higher-level plans. (Not found as a defined margin-note term in the extracted book text; this is standard AIMA usage.)

## exhaustive decomposition

A decomposition of a category into subcategories such that every member of the category belongs to at least one subcategory. When the subcategories are also mutually disjoint, the exhaustive decomposition is a partition.

## existence uncertainty

Uncertainty about what objects actually exist (e.g., which real books or customers there are), as opposed to uncertainty about the properties of known objects; it arises in open-universe probability models.

## existential graphs

A graphical logical notation developed by Charles Sanders Peirce (which he called "the logic of the future"), regarded as an early precursor of semantic-network formalisms.

## Existential Instantiation

An inference rule of first-order logic that replaces an existentially quantified variable with a single new constant symbol (a Skolem constant) that does not appear elsewhere in the knowledge base; unlike Universal Instantiation, it need be applied only once.

## expand

In search, to generate the successor nodes of a node by applying each applicable action to it, thereby extending the search tree/frontier.

## expectation

The expected value of a random variable: the average of its possible values, each weighted by its probability (summed for discrete variables, integrated for continuous ones).

## expectation-maximization

An iterative algorithm (EM) for maximum-likelihood learning in models with hidden variables. It alternates an E-step, which computes the expected values (posterior probabilities) of the hidden variables given the current parameters, with an M-step, which re-estimates the parameters to maximize likelihood as if those expected values were observed.

## expected monetary value

The probability-weighted average of the monetary outcomes of a gamble or decision (EMV); note that maximizing EMV is not always rational because utility is generally not proportional to money.

## expected utility

The average utility of the outcomes of an action given the evidence, EU(a), computed by weighting each outcome's utility by the probability that the outcome occurs.

## expected value

The probability-weighted average of the possible outcomes of a random variable or of an action's results; used, for example, to evaluate chance nodes and to make decisions under uncertainty.

## expectiminimax value

A generalization of the minimax value for games that include chance nodes. It is computed by taking maxima at MAX nodes, minima at MIN nodes, and, at chance nodes, the expectation (probability-weighted average) of the values of the successor states.

## expert systems

AI programs that apply specialized knowledge to solve problems in a specific domain of human expertise; a methodology pioneered by Stanford's Heuristic Programming Project.

## expit

The inverse-logit model, an alternative to the probit model used to map a linear input to a probability via the logistic function.

## explainable AI

A system (abbreviated XAI) that can explain the reasoning behind its decisions to humans.

## explanation-based learning

A method for extracting general rules from individual observations by constructing an explanation, using background knowledge, of why an example has the properties it does, and then generalizing that explanation; it can produce generalized plans from experience.

## explanatory gap

The apparent difficulty in explaining how and why subjective conscious experience arises from physical brain processes. (Not found as a defined margin-note term in the extracted book text; this is standard AIMA usage related to the discussion of consciousness.)

## exploitation

In sequential decision making (e.g., bandit and reinforcement learning problems), choosing the action that currently appears best (maximizing expected reward given present knowledge), as opposed to exploring.

## exploration

In sequential decision making, choosing actions to gather information about the environment (reducing uncertainty and improving future decisions) even at the expense of immediate reward.

## exploration bonus

An added incentive given to choices (e.g. bandit arms) that have been tried only a few times, encouraging the agent to gather more information about under-explored options.

## exploration function

A function used in reinforcement learning that trades off greed (preference for high estimated utility) against curiosity (preference for actions tried infrequently); it increases with estimated value and decreases with the visit count, so the agent is drawn to under-explored state-action pairs.

## exploration problem

The problem faced by an agent that must act in an environment whose states and actions it does not know in advance; it must decide when to explore to learn about the world and when to exploit what it has learned, balancing exploration against exploitation.

## exploratory data analysis

Examining data with various plots and statistical tests to get a feel for what the data looks like and gain insight into what hypothesis space might be appropriate.

## expressiveness

The range of facts and distinctions a representation language can capture, forming an axis along which representations can be ordered. A more expressive language can state more, but typically at the cost of more difficult inference.

## extended Kalman filter (EKF)

A version of the Kalman filter that copes with nonlinear transition or measurement models by locally linearizing them (via a Taylor expansion) around the current state estimate, so the standard Gaussian Kalman update equations can be applied.

## extension

In default logic, a maximal set of consequences of a default theory: it comprises the known facts plus a set of conclusions drawn from the default rules such that no further conclusions can be added and the justifications of every default conclusion remain consistent (e.g., the Nixon-diamond theory has two extensions).

## extensive form

A representation of a (typically sequential) game as a tree that makes explicit the order of moves, the information available to each player at each decision point, and the payoffs at the leaves; it can capture imperfect information.

## externalities

Effects on global utility that are not recognized in the individual agents' transactions (the economists' term for factors external to a deal); mechanisms may fail to account for them unless deliberately designed to do so.

## extrinsic

Of a property of an object, one (such as weight, length, or ownership) that is not retained when the object is divided into parts, in contrast to intrinsic properties; a category defined to include extrinsic properties behaves like a count noun.

# F

## fact

In a Horn/definite-clause knowledge base, a sentence consisting of a single positive literal (e.g., L1,1). It can be written in implication form as True ⇒ L1,1, but is usually written simply as the literal itself.

## factor

In the variable elimination algorithm for Bayesian-network inference, a matrix indexed by the values of its argument variables that represents part of the probability expression being evaluated. Factors are manipulated by pointwise product and by summing out variables.

## factored frontier

An approximate filtering algorithm for dynamic Bayesian networks (Murphy and Weiss, 2001) that, like the Boyen–Koller algorithm, assumes the posterior distribution over states can be approximated well by a product of small factors rather than a full joint distribution.

## factored representation

A representation that splits each state into a fixed set of variables (attributes), each with a value. Unlike atomic states, two factored states can share some attributes and differ on others; constraint satisfaction, propositional logic, planning, Bayesian networks, and many learning methods rest on such representations.

## factorial HMM

(Not defined in AIMA 4e.) A hidden Markov model whose hidden state is decomposed into several state variables that evolve independently, so the overall state is the combination of these separate chains.

## factoring

In resolution, the step of reducing a resulting clause so that it contains only one copy of each literal by removing redundant duplicate literals (e.g., resolving to get A ∨ A, then factoring to just A).

## false negative

In classification, an example that is actually positive but is labeled negative (e.g., labeling a piece of spam as legitimate mail). It is traded off against false positives.

## false positive

In classification, an example that is actually negative but is labeled positive (e.g., labeling a legitimate piece of mail as spam). It is traded off against false negatives, a tradeoff visualized by the ROC curve.

## feature extraction

An approach to perception that applies simple, direct computations to raw sensor responses to compute informative features that can be used directly by a program (for example, an animal computing time-to-contact from visual input to steer).

## feature map

In a convolutional neural network, an array of values produced by applying a kernel across an input, representing the presence of a detected feature at each spatial location. Feature maps are split into channels, and CNNs use multidimensional feature maps to represent layer activations.

## feature selection

The process of reducing the dimensionality of a model by discarding attributes that appear to be irrelevant; decision-tree pruning is one kind of feature selection.

## federated learning

A machine learning approach with no central database: each user keeps their training data in their own local database, so the raw data stays completely private while models are still learned collaboratively.

## feed-forward network

A neural network whose connections go in only one direction, forming a directed acyclic graph from input to output nodes with no loops; each node computes a function of its inputs and passes the result forward. Contrasted with a recurrent network, which feeds outputs back into its own inputs.

## feedback

The information about outcomes or correctness that accompanies an agent's inputs during learning. The type of feedback available (e.g., labeled outputs, rewards, or none) distinguishes the main kinds of learning: supervised, reinforcement, and unsupervised.

## FIFO queue

A first-in, first-out queue that first pops the node added earliest; it is used by breadth-first search. Contrasted with a LIFO queue (stack) and a priority queue.

## filtering

The task of computing the belief state—the posterior distribution over the most recent state given all evidence observed so far. Also called state estimation.

## fine-motion planning

A robust motion-planning method for robotic assembly tasks that plans precise, small-scale motions in the presence of uncertainty. A fine-motion plan consists of a series of guarded motions that continue until a sensed condition (such as contact) is met.

## finite horizon

In sequential decision problems, the assumption that there is a fixed time N after which nothing matters, so the utility of a state sequence is unaffected by what happens beyond N. Under a finite horizon the optimal action can depend on how much time remains (nonstationary policy). Contrasted with infinite horizon.

## first-choice hill climbing

A form of stochastic hill climbing that generates successors at random until one is found that is better than the current state, then moves to it—a good strategy when a state has very many (e.g., thousands of) successors.

## first-order logic

A representation language, more expressive than propositional logic, that models the world in terms of objects, the relations (and properties) that hold among them, and functions. Also called first-order predicate calculus (FOL or FOPC).

## first-order Markov process

The simplest Markov process, in which the current state depends only on the immediately preceding state and not on any earlier states; the transition model is then P(X_t | X_{t-1}).

## fixate

In stereopsis, to direct the two eyes so that their optical axes intersect at some point in the scene (the point of fixation), where the disparity is zero; used to compute depth from angular disparity.

## fixed point

In forward chaining, a state of the knowledge base reached when no new inferences are possible, so that everything derivable is already present; applying the inference process yields the same set of sentences. (For first-order clauses the fixed point can include universally quantified atomic sentences.)

## fixed-lag smoothing

Smoothing done in an online setting, computing the smoothed estimate for the time slice a fixed number of steps d behind the current time t; as t advances the smoothing must keep up, and in some cases it can be done in constant time per update.

## flaw

(Not defined in AIMA 4e.) In partial-order planning, an open precondition or a threatened causal link in a partial plan that must be resolved before the plan is complete and consistent.

## fluent

A noun (from Latin fluens, "flowing") for an aspect of the world that changes over time; a synonym for "state variable" in the factored-representation sense. Symbols for permanent aspects need no time superscript and are called atemporal variables.

## focal plane

In a lens system, the depth (center of the range where focus is sharpest) at which points in the scene are imaged sharply; the range of depths around it that stay acceptably sharp is the depth of field. In a camera the lens moves to change the focal plane; in the eye the lens shape changes.

## focus of expansion

In an optical flow field produced by an observer moving forward, the single point from which the apparent motion of scene features appears to radiate outward.

## foreshortening

The distortion by which a surface (or pattern) viewed at a slanting/glancing angle appears compressed in the image—for example, a book tilted away seems to shrink. Modeling it aids object recognition and provides cues for reconstructing geometry.

## forget gate

A component of an LSTM unit whose value controls how much of the memory cell's contents are retained; it can be used to quickly reset the cell state to zero.

## formal logic

The systematic, mathematical study of valid reasoning using precisely defined symbols and rules. Its roots trace to ancient Greek philosophers, with mathematical development beginning with George Boole's propositional (Boolean) logic.

## forward checking

A simple form of constraint propagation: whenever a variable is assigned a value, the algorithm removes from the domains of neighboring unassigned variables any values that are now inconsistent with the assignment.

## forward propagation

The process, in a neural network's computational graph, of traversing the graph in the direction of its directed edges to gradually compute all desired output values from the inputs.

## forward pruning

A game-tree search technique that immediately prunes moves appearing to be poor without full evaluation, even though some of those moves might actually turn out to be good.

## forward-backward algorithm

A dynamic-programming algorithm for smoothing in hidden Markov models: a forward pass computes state probabilities given evidence up to each point, and a backward pass computes the likelihood of subsequent evidence; combined, they give the posterior over each state given the entire evidence sequence.

## forward-chaining

An inference algorithm (for Horn/definite clauses) that starts from known facts and repeatedly applies rules whose premises are satisfied to derive new facts, adding them to the knowledge base, until the query is derived or no new facts can be added. Contrasted with backward chaining.

## frame problem

The problem, arising from effect axioms that state only what an action changes, of representing everything that remains unchanged when an action occurs—without having to write an explicit frame axiom for every unaffected proposition at every time step. (The name comes from a "frame of reference" in physics.)

## frames

A structured knowledge-representation formalism (from Minsky, 1975), a version of semantic networks in which a frame represents an object or category with attributes and relations (links) to other objects or categories.

## framing effect

A departure from rational decision making in which the exact wording of a decision problem sways choices; e.g., a procedure described as a "90% survival rate" is preferred to the logically identical "10% death rate."

## free space

In robot motion planning, the region of configuration space made up of the configurations that are not C-space obstacles—i.e., the configurations in which the robot does not collide with any obstacle. It is the complement of the C-space obstacle.

## frequentist

The position that probability numbers can come only from experiments: the probability of an event is the fraction of occurrences observed in the limit of infinitely many similar trials, estimated from finite samples. Contrasted with the objectivist and subjective (Bayesian) views.

## friendly AI

The design goal of building highly capable AI whose objectives remain beneficial to humans; AIMA frames this in terms of agents that are provably beneficial—machines that are uncertain about the human objectives they are meant to pursue and therefore defer to human control.

## front-to-end

In bidirectional search, a heuristic strategy that estimates the distance from a node to the goal (the end) rather than to the opposing search frontier.

## front-to-front

In bidirectional search, a strategy that estimates the distance from a node to the other search's frontier, rather than to the goal.

## frontier

In search, the set of nodes that have been generated (reached) but not yet expanded; it separates the explored interior region of the state-space graph from the unexplored region. Also called the open list.

## full joint probability distribution

The joint distribution over all the random variables in a domain, which completely determines a probability model. Any query about the domain can in principle be answered from it by summing the entries of the possible worlds that satisfy the query.

## fully observable

A task environment is fully observable if the agent's sensors give it access to the complete state of the environment at each point in time (effectively so if they detect all aspects relevant to the choice of action). Such environments are convenient because the agent need not maintain internal state; otherwise the environment is partially observable.

## function approximator

A parameterized function (using far fewer parameters than there are states) used in place of an exact table to estimate a target function, such as a reinforcement-learning agent approximating the true utility function with a small set of weights.

## functionalism

(Not defined in AIMA 4e.) The philosophical position that a mental state is defined by its functional/causal role—its relations to inputs, outputs, and other mental states—rather than by the physical substrate that realizes it, implying that suitably organized systems could have minds.

## futility pruning

A game-search pruning technique that decides in advance which moves are so unpromising that they will cause a beta cutoff in the successor nodes, so they need not be explored.

## fuzzy control

A methodology for constructing control systems in which the mapping between real-valued input and output parameters is represented by fuzzy rules; it has been very successful in commercial products such as automatic transmissions, video cameras, and electric shavers.

## fuzzy logic

A method for reasoning with logical expressions describing membership in fuzzy sets; it makes the ontological commitment that propositions have a degree of truth between 0 and 1 (e.g., "Vienna is a large city" true to degree 0.8). Rather than competing with probability theory, it addresses vagueness rather than uncertainty about well-defined propositions.

## fuzzy set theory

The theory of fuzzy sets (Zadeh, 1965), where membership in a set is a matter of degree rather than all-or-nothing; it was developed to cope with the difficulty of providing exact inputs to intelligent systems and underlies fuzzy logic and fuzzy control.

# G

## G-set

In version-space learning, the G-set (general boundary) is the set of the most general hypotheses consistent with the examples seen so far. Together with the S-set (specific boundary) it delimits the version space: every hypothesis consistent with the data lies between these two boundary sets.

## gain parameter

A scalar constant in a control law that scales a feedback term to set how strongly the controller responds. AIMA's PID controller uses proportional, derivative, and integral gain parameters; larger gains make the actuator react more forcefully but can produce oscillation.

## gain ratio

The information gain ratio normalizes an attribute's information gain by the intrinsic information of its split. In decision-tree learning it is used to discourage choosing attributes with many distinct values (such as Zipcode), which would otherwise split the data into many single-example subtrees.

## gait

For a legged robot, the chosen pattern of movement of the limbs used to locomote. AIMA gives the example of a statically stable hexapod gait that moves one set of three legs while the other three stay planted, then swaps.

## game theory

The branch of study that explicitly models adversarial (multi-agent) situations as games between rational players who are each actively trying to win. In AIMA it underlies adversarial search, where the optimal move is computed by minimax, and is developed further into non-cooperative and cooperative game theory for multi-agent decision making.

## game tree

A tree superimposed over the state-space graph of a game, whose nodes are game states and whose edges are moves, following every sequence of moves from the initial state to terminal states. The game tree may be infinite if the state space is unbounded or positions can repeat indefinitely.

## Gaussian distribution

A continuous probability distribution, also called the normal distribution, defined by its mean and variance (or covariance) and having the familiar bell-shaped density. In its multivariate form it is central to Kalman filtering and to linear-Gaussian models in Bayesian networks.

## Gaussian error model

A sensor or process model in which the noise is assumed to be Gaussian, centered on the true value. AIMA notes that a simple Gaussian error model does not accommodate transient failures (a sensor briefly sending nonsense), which can lead filtering to draw badly wrong conclusions.

## Gaussian filter

An image-smoothing filter that replaces each pixel's intensity with a weighted sum of nearby pixel intensities, where the weights come from a zero-mean Gaussian function. Nearby pixels receive more weight, so the filter suppresses noise while blurring the image.

## Gaussian process

A nonparametric Bayesian method that defines a prior distribution over the space of continuous functions, such that any finite set of function values is jointly Gaussian. It supports regression that yields calibrated uncertainty estimates over its predictions.

## generalization

The ability of a learned hypothesis to predict accurately on new inputs it did not see during training, rather than merely fitting the training examples.

## generalization hierarchy

A hierarchy of categories organized from more specific to more general, in which a general category subsumes its subcategories so that instances inherit the properties of every category above them, as in AIMA's semantic-network inheritance reasoning.

## generalization loss

The expected loss of a hypothesis over the full (usually unknown) distribution of input-output examples, with respect to a chosen loss function. The best hypothesis is the one minimizing expected generalization loss; because the distribution is unknown, it is estimated by the empirical loss on a set of examples.

## generalized modus ponens

An inference rule for first-order logic: given atomic sentences and an implication whose premises unify with them under some substitution, it concludes the substituted consequent. It combines unification with modus ponens in one sound step, requiring premises to match only up to a substitution.

## generating

In search, producing successor states by applying the available actions to a state. AIMA distinguishes early goal testing (checking a node as soon as it is generated) from late goal testing (checking only when the node is selected for expansion).

## generative adversarial networks

The most popular framework for implicit generative modeling, in which two neural networks are trained against each other: a generator that maps noise to samples resembling the data, and a discriminator (classifier) that tries to tell real training examples from generated ones.

## generative model

A model capable of estimating the probability of its inputs or of generating new examples drawn from the same distribution as its training data.

## generative modeling

A goal of unsupervised learning in which the aim is to learn a model of the data distribution that can produce new examples resembling the training data.

## generator

A component that proposes candidate outputs or examples for another component to evaluate. In AIMA's learning agent, the problem generator suggests exploratory actions that may be suboptimal in the short run but lead to informative new experiences.

## genetic algorithms

A form of stochastic local (evolutionary) search in which a population of individuals, each represented as a string over a finite alphabet, is evolved toward higher fitness through selection, crossover (recombination), and mutation.

## genetic programming

A subfield of genetic algorithms in which each individual in the population is a computer program (typically an expression tree) rather than a fixed-length string, so that evolution searches over programs that perform well on a task.

## Gibbs sampling

A Markov chain Monte Carlo algorithm for approximate inference in Bayesian networks. Starting from an arbitrary state, it repeatedly resamples each nonevidence variable in turn conditioned on the current values of its Markov blanket, and the resulting samples approximate the posterior.

## Gittins index

For an arm of a multi-armed bandit (each arm being a Markov reward process), a value assigned to that arm such that always playing the arm with the highest index is an optimal strategy; it captures the value of an arm accounting for both its reward and its information value.

## GLIE

Greedy in the Limit of Infinite Exploration: a property an exploration scheme must have to be guaranteed to converge to an optimal policy. A GLIE scheme tries each action in each state an unbounded number of times (so no optimal action is missed) while eventually behaving greedily.

## global constraint

In constraint satisfaction, a constraint involving an arbitrary number of variables (the name is traditional; it need not involve all variables). A common example is Alldiff, which requires that all variables it constrains take different values, as used in Sudoku rows, columns, and boxes.

## global maximum

In an optimization landscape, the highest point overall—the state with the greatest objective-function value—as opposed to a merely local peak.

## global minimum

The point at which a function attains its lowest value over its entire domain, as opposed to a local minimum that is lowest only within a neighborhood; gradient descent seeks to reach such a lowest-valued "valley."

## Global Positioning System

A satellite-based location-sensing system (GPS) that is the most common outdoor solution to the robot localization problem, determining position from signals of orbiting satellites.

## Go

An ancient two-player board game played by placing stones on a grid. Its very large branching factor long made it hard for AI, until programs such as ALPHAGO combined deep neural-network evaluation with Monte Carlo tree search to reach and exceed top human play.

## goal

A state, or set of states, that a goal-based agent is trying to reach. Goals organize behavior by limiting the objectives and hence the actions the agent needs to consider.

## goal clauses

Horn clauses that contain no positive literal. In resolution-based proving they encode the negated query; deriving the empty clause from them together with the knowledge base establishes the original query.

## goal formulation

The first phase of the problem-solving process, in which the agent adopts a goal (a set of goal states) based on the current situation and its performance measure. Adopting a goal organizes behavior by limiting the objectives and actions to be considered.

## goal monitoring

During plan execution, checking before each action whether there is a better set of goals the agent could be pursuing, and re-forming goals if so; contrasted with action monitoring and plan monitoring.

## goal states

The set of one or more states in a search problem that the agent is trying to reach; a solution is a path from the initial state to a goal state.

## goal test

A test in a search problem that determines whether a given state is a goal state, letting the search algorithm recognize when a solution has been found. It may be applied early (when a node is generated) or late (when a node is expanded).

## goal-directed reasoning

Reasoning that works backward from a goal, as in backward chaining, to find facts or actions that would achieve it. It is useful for answering specific questions and often touches only the relevant part of the knowledge base; contrasted with data-driven forward reasoning.

## gold standard

A reference set of correct answers, typically produced or verified by humans, against which a system's outputs are compared to measure accuracy.

## gorilla problem

AIMA's name for the concern that, just as gorillas ceded control of their future to the more intelligent human branch, humanity could lose control of its future by creating machines more intelligent than itself; it captures the essence of Turing's warning about controlling superintelligent machines.

## gradient

The vector giving the magnitude and direction of the steepest slope of an objective function with respect to its parameters. Optimization methods use the gradient to move toward a maximum (or, negated, a minimum).

## gradient clipping

A technique used to counter exploding gradients, especially in recurrent networks, by capping the magnitude of the gradient during training so updates remain bounded.

## gradient descent

An optimization method that iteratively adjusts parameters in the direction opposite to the gradient of a loss function, taking steps scaled by a learning rate, in order to descend toward a minimum of the function.

## grammar

A formal specification of the allowable structure of strings in a language, typically a set of rewrite rules over terminal and nonterminal symbols (for example, the BNF grammar of propositional logic in AIMA).

## graph

A structure of nodes (vertices) joined by edges, used to represent relations such as the state-space graph of a search problem, in which nodes are states and edges are actions.

## graph coloring

The problem of assigning colors to the nodes of a graph so that no two adjacent nodes share a color; it is a canonical constraint satisfaction problem, exemplified by map coloring.

## graph search

A search algorithm that checks for and avoids redundant paths (for example, by remembering already-reached states), in contrast to a tree-like search that may re-expand repeated states.

## grasping

The robotics problem of determining a stable way for a manipulator or hand to hold an object, which requires reasoning about contact points and forces (often using touch or haptic feedback to detect contact and slip).

## greedy agent

In reinforcement learning, an agent that at each step takes the action it currently believes to be optimal according to its learned model. Because the learned model differs from the true environment, greedy action selection often converges to a suboptimal policy, since it ignores the informational value of actions.

## greedy best-first search

A form of best-first search that expands first the node judged closest to the goal by a heuristic function h(n), ignoring the path cost already incurred to reach the node.

## greedy local search

Another name for hill climbing: it repeatedly moves to a good neighboring state without looking ahead to plan where to go beyond the immediate next step.

## grid search

A hyperparameter optimization method that systematically tries combinations of values laid out on a grid across the multidimensional space of hyperparameter settings.

## grid world

A two-dimensional rectangular array of square cells through which an agent moves; a standard setting used to illustrate search, planning, and reinforcement-learning problems (for example, vacuum-world or sokoban-style tasks).

## ground resolution theorem

The completeness result for resolution in propositional logic: if a set of clauses is unsatisfiable, then the resolution closure of those clauses contains the empty clause.

## ground term

A term in first-order logic that contains no variables, and thus denotes a specific object in the domain.

## grounding

The connection between an agent's internal logical reasoning (its syntax) and the real environment it inhabits. AIMA explains that an agent's sensors provide this connection for percept sentences, while general rules gain their grounding, fallibly, through learning.

## guaranteed checkmate

In Kriegspiel (a partially observable chess variant), a winning strategy that, for every possible percept sequence, forces an actual checkmate for every board state in the current belief state, no matter how the opponent moves.

# H

## Hamming distance

For two points described by Boolean attributes, the number of attributes on which the two points differ. AIMA presents it as the Minkowski/Manhattan-style distance measure used for Boolean-valued features in nearest-neighbor methods.

## hand-eye machines

Robots that combine visual perception with physical manipulation, coordinating a camera (eye) with an arm or gripper (hand); the first major example was Heinrich Ernst's MH-1 (1961), which linked vision to grasping.

## Hansard

The official published record of parliamentary debates, notably the Canadian Hansard produced in both English and French. Such paired bilingual documents form a parallel corpus used to train and evaluate machine translation systems.

## haptic feedback

Sensing of contact forces through touch, used in robotics to determine the forces a manipulator is exerting and to detect slip during grasping and manipulation.

## head

In a Horn (or definite) clause written as an implication, the single positive literal that forms the conclusion, as opposed to the body, which is the premise. A clause with just a head and no body is a fact.

## heavy-tailed distribution

A run-time (or other) distribution in which the probability of a very long value is much higher than an exponential distribution would predict. AIMA notes that systematic backtracking search often has heavy-tailed run times, which is why random restarts can find solutions faster on average.

## Hebbian learning

A simple neural learning rule, due to Donald Hebb (1949), that modifies the connection strength between two neurons based on their joint activity, strengthening the link between neurons that are active together; it remains an influential model.

## Herbrand's theorem

A theorem in first-order logic stating that if a set S of clauses is unsatisfiable, then some finite subset of its Herbrand base (the ground instantiations of S over the Herbrand universe) is also unsatisfiable. It justifies reducing first-order inference to propositional-style reasoning over ground terms.

## Hessian

The square matrix of second-order partial derivatives of a function, describing its local curvature. AIMA notes that computing and inverting the Hessian is often expensive, so second-order (Newton-style) methods that use it can be costly.

## heuristic function

A function h(n) that estimates the cost of the cheapest path from the state at node n to a goal, providing problem-specific hints that guide informed (heuristic) search algorithms toward the goal.

## heuristic search

Informed search that uses a heuristic function to prioritize which states to explore, so that it typically finds solutions more efficiently than uninformed search; its performance depends on the quality of the heuristic.

## hidden

Describes the intermediate variables or computations in a neural network whose values are not specified by the training data, in contrast to the observed input and output variables.

## hidden Markov model

A hidden Markov model (HMM) is a temporal probabilistic model in which the state of the process is described by a single, discrete random variable (any other model can be forced into this form by combining variables into one "megavariable"). Its restricted structure yields a simple and elegant matrix formulation of the filtering and smoothing algorithms.

## hidden representation

The set of internal features h(x) that a neural network extracts from an input x to summarize it, forming an intermediate encoding used by later layers to produce the output.

## hierarchical decomposition

The strategy of managing complexity by breaking a problem down into levels, in which high-level actions are recursively refined into sequences of lower-level actions; central to hierarchical task network (HTN) planning.

## hierarchical lookahead

Planning that reasons at multiple levels of abstraction, examining the effects of high-level actions before committing to their detailed refinements so that infeasible plans can be pruned early; in AIMA this rests on high-level-action descriptions (e.g., angelic semantics) satisfying the downward refinement property.

## hierarchical reinforcement learning

An approach (HRL) that copes with very long action sequences by breaking a task into a hierarchy of smaller subtasks, and those into still smaller pieces, until the action sequences are short enough to learn easily; it has much in common with HTN planning.

## high-level action

In hierarchical (HTN-style) planning, an abstract action (an HLA), such as "Go to San Francisco airport," that has one or more refinements into sequences of lower-level actions, each of which may itself be an HLA or a primitive action.

## Higher-order logic

A logic that extends first-order logic by treating the relations and functions of first-order logic as objects that can themselves be quantified over and reasoned about.

## Hinton diagrams

A way of visualizing the weights of a neural network in which each weight is drawn as a shape (typically a square) whose size encodes the weight's magnitude and whose color or shading encodes its sign.

## HOG feature

A Histogram Of Gradient orientations feature: an image descriptor, computed over a sliding window, that summarizes the distribution of local intensity-gradient directions, widely used for object detection.

## holdout cross-validation

A validation method in which part of the data is held out as a separate validation (or test) set while the model is trained on the remaining examples, then evaluated on the held-out data to estimate its error rate.

## homeostatic

Describing a device or system that uses feedback loops to keep its internal variables within preferred ranges, maintaining stable adaptive behavior; the term is associated with Ashby's early cybernetic idea that intelligence might arise from such homeostatic devices.

## homophones

Words that are pronounced the same but differ in meaning or spelling, which are a source of ambiguity for speech recognition systems.

## horizon effect

A pitfall of depth-limited game-playing search in which the program faces serious, ultimately unavoidable damage but uses delaying tactics to push the damaging event just beyond its search horizon, and so fails to account for it.

## Horn clause

A disjunction of literals of which at most one is positive. Horn clauses can be written as implications and are closed under resolution; they include definite clauses (exactly one positive literal) and goal clauses (no positive literal), and they support efficient inference such as forward and backward chaining.

## hub

In web link analysis, a page that points to many good (authoritative) pages. AIMA's HITS algorithm computes mutually reinforcing hub and authority scores to rank pages.

## human-level AI

The goal, also called HLAI, of building AI that matches human ability across the full breadth of cognitive tasks (machines "that think, that learn and that create"), rather than excelling only in narrow, specific applications.

## Hungarian algorithm

A polynomial-time combinatorial method for the assignment problem. In AIMA's data-association setting it efficiently chooses the assignment of observations to predicted objects that maximizes joint probability, despite the factorially many possible assignments.

## hybrid A*

A variant of the A* search algorithm for continuous robot motion planning in which each grid cell records the continuous state (position and heading) at which it was reached, and the robot's continuous motion model is used to expand to nearby cells, guaranteeing a smooth, executable trajectory.

## hybrid agent

An agent that combines logical inference with problem-solving/planning components. AIMA's wumpus-world hybrid agent maintains a propositional knowledge base together with a current plan, using inference to deduce the state of the world and planning to decide what to do.

## hybrid architecture

An agent design that integrates several control paradigms, typically layering reactive, executive, and deliberative levels so that fast reflexes and slower deliberation coexist within one system.

## hybrid Bayesian network

A Bayesian network containing both discrete and continuous random variables, which requires conditional distributions (such as linear-Gaussian distributions) that connect the two kinds of variables.

## hydraulic actuation

Driving a robot's effectors using pressurized hydraulic fluid (such as oil or water) to generate mechanical motion; one of the actuator types AIMA contrasts with electric and pneumatic actuators.

## hyperparameters

Parameters that define the hypothesis class or learning procedure rather than the individual model, such as a polynomial's degree, a decision tree's pruning threshold, or a learning rate; they are chosen during model selection rather than fit by ordinary training.

## hypothesis

A candidate function that approximates the unknown target function, drawn from a hypothesis space, proposed to explain the training data and to predict future examples.

## hypothesis prior

In Bayesian learning, the prior probability P(h) assigned to each hypothesis before any data is observed. Combined with the likelihood of the data under each hypothesis, it yields the posterior over hypotheses via Bayes' rule.

## hypothesis space

The set of candidate hypotheses (functions) from which a learning algorithm chooses when fitting the data, for example the set of degree-3 polynomials, of JavaScript functions, or of 3-SAT formulas; also called a model class or function class.

# I

## i.i.d.

Short for **independent and identically distributed**: a set of examples (or random variables) that are drawn independently from the same fixed prior probability distribution. This is the standard "stationarity" assumption underlying much of learning theory.

## identification in the limit

A criterion for learning success, due to Gold (1967), in which a learner succeeds if it eventually converges on the correct hypothesis after some finite amount of data and never changes thereafter, even if it can never be sure it has converged. This early approach to learning theory was applied mainly to learning grammars from example sentences.

## identity matrix

A matrix **I** whose entries are 1 on the main diagonal (where the row and column index are equal) and 0 elsewhere, and which satisfies **IA** = **A** for any conformable matrix **A**.

## identity uncertainty

Uncertainty about which logical terms actually refer to the same object in the world; that is, not knowing whether two distinct observations or references denote the same underlying entity. It arises together with existence uncertainty in problems such as record linkage and data association.

## ignore delete lists

A relaxation used to derive planning heuristics: assuming goals and preconditions contain only positive literals, the negative (delete-list) effects of every action are removed. Once a fluent becomes true it stays true, so progress toward the goal is monotonic and the relaxed problem is easier to solve.

## ignore preconditions heuristic

A planning heuristic that drops all preconditions from actions, making every action applicable in every state. Any single goal fluent can then be achieved in one step, so the number of steps to solve the relaxed problem roughly equals the number of unsatisfied goals.

## image

A rectangular grid of pixels forming the projection of a scene onto the image plane of a sensor. Each pixel is a tiny sensor whose output reports a weighted average of the intensity of the light arriving at that point, over wavelength, direction, time, and area.

## image captioning

The computer vision task of automatically generating a natural-language description of the contents of an image, a step beyond simple object recognition.

## ImageNet

A large standard image database containing over 14 million labeled photos spanning about 20,000 categories, widely used as a benchmark for training and evaluating object-recognition systems.

## imitation learning

A learning approach in which the agent learns to perform a task by observing an expert's behavior and reproducing it, effectively told to "do it like that."

## imperfect information

The condition in a game where players lack full knowledge of the state, for example because some information is private (as in poker, where players hold cards their opponents cannot see).

## implementation

The concrete realization of an abstract agent design as a program that runs on some architecture; the agent program takes a percept as input and returns an action.

## implementation level

The lowest level at which a system can be described, concerned with the actual physical or programmatic mechanisms that carry out the computation, as opposed to the knowledge level (what the agent knows) or higher levels of abstraction.

## implication

A sentence formed with the "implies" connective, also called a conditional or if-then statement (and sometimes a rule). Its premise (antecedent) is on the left and its conclusion (consequent) on the right; it is false only when the antecedent is true and the consequent is false.

## implicit models

A category of generative models that can produce samples from a distribution without explicitly defining its probability density; generative adversarial networks (GANs) are the most popular example.

## importance sampling

A general statistical technique that emulates sampling from a hard-to-sample target distribution by drawing samples from a different, more convenient distribution and reweighting them to correct for the difference.

## inadmissible heuristic

A heuristic that may overestimate the true cost to reach the goal; using one in A* search risks losing optimality but can be more accurate and expand fewer nodes.

## inclusion-exclusion principle

A counting rule for the probability (or size) of a union of sets: sum the individual probabilities, subtract the pairwise intersections, add back the triple intersections, and so on, so that overlapping regions are not counted more than once.

## income inequality

The uneven distribution of income across a population, which technology and automation tend to magnify, raising economic and ethical concerns about AI's societal impact.

## incompleteness theorem

Gödel's 1931 result that in any formal theory strong enough to express arithmetic there exist true statements that have no proof within the theory; it establishes fundamental limits on what deduction can achieve.

## incremental belief-state search

An approach to searching over belief states in which a solution is built up one physical state at a time: find a plan that works for one state in the belief state, then check whether it also works for the others, extending or repairing it as needed.

## independence

A relationship between two variables (or events) in which knowing one gives no information about the other, so their joint distribution factors into the product of the separate distributions. In a constraint graph, independence corresponds to separate connected components.

## independent subproblems

Portions of a problem that share no variables, so each can be solved separately and the solutions combined into a solution of the whole. In a CSP they correspond to the connected components of the constraint graph, giving large computational savings.

## index

A data structure that stores, for each key or term, pointers to the sentences or records that contain it, so that items matching a query can be retrieved without scanning the whole knowledge base.

## indexed random variable

A random variable that is one member of a family distinguished by an index (such as an object identifier or time step), used in probabilistic modeling languages to write down repeated or object-specific quantities compactly.

## indexing

Organizing a knowledge base so relevant facts can be fetched efficiently; for example, predicate indexing groups all clauses by their predicate symbol so a query can be answered by examining only the relevant group.

## individuation

The division of the world into distinct objects (things). Some parts of reality, such as substances like butter (stuff), resist individuation because any part of the substance is still the same substance, unlike a countable object such as an aardvark.

## induction

Hume's principle that general rules are acquired by exposure to repeated instances; reasoning from specific observations to general conclusions, which, unlike deduction, yields conclusions that are probable rather than guaranteed.

## inductive learning

Learning a general rule (function) from a collection of specific input-output pairs; the learned rule may be only an approximation and is not guaranteed to be correct.

## inductive logic

An attempt (associated with Carnap) to build a formal theory of reasoning under uncertainty analogous to deductive logic, in which evidence confers degrees of belief on hypotheses rather than entailing them with certainty.

## inductive logic programming

A machine learning method that combines inductive learning with first-order logic, producing hypotheses expressed as logic programs (rules) that, together with background knowledge, explain a set of examples.

## inference

The process by which a knowledge-based agent derives new sentences from those already in its knowledge base; done properly, it produces only conclusions that follow from what has been told, without "making things up."

## inference rules

Patterns of sound inference (such as Modus Ponens or resolution) that can be applied to given sentences to derive new sentences and thereby construct proofs, while preserving truth.

## inferential frame problem

The problem of reasoning efficiently about the many aspects of the world that an action leaves unchanged, so that the agent does not have to re-derive every unaffected fact at each time step.

## infinite

Describing a state space, sequence, or quantity that is unbounded; infinite state spaces arise frequently and require extra care to ensure search algorithms behave correctly.

## infinite horizon

A sequential decision setting with no fixed bound on the number of time steps, so the agent may act indefinitely. Handling it correctly (for example with discounting) is more subtle than the finite-horizon case.

## infix

Notation in which an operator symbol is written between its operands (for example, a plus sign between two numbers, or "and" between two sentences), as opposed to prefix or postfix notation.

## influence diagram

A graphical notation, also called a decision network, that augments a Bayesian network with decision nodes and utility nodes to represent and solve decision problems under uncertainty.

## information extraction

The process of acquiring knowledge by skimming text and extracting structured facts, such as the entities mentioned and the relations that hold among them, from unstructured natural language.

## information gain

The expected reduction in entropy about the classification produced by testing a given attribute; it measures how informative an attribute is and is used to choose which attribute to split on when building a decision tree.

## information gathering

Taking actions whose purpose is to acquire knowledge (such as sensing or experimenting) in order to reduce uncertainty; this is an important part of rational behavior.

## information gathering action

An action taken specifically to modify future percepts and reduce uncertainty about the state of the world, such as a robot sensing a critical unknown variable before acting; part of rational behavior.

## Information retrieval

The task of finding documents that are relevant and important for a given query.

## information sets

The game-theoretic name for the belief states of players in a game with imperfect information: a set of states a player cannot tell apart given their observations, across which the player must act identically. The extensive form makes these explicit and enables computing equilibrium solutions.

## information value theory

A theory that enables an agent to decide which information to acquire by quantifying the expected improvement in decision quality gained from observing a piece of evidence, weighed against the cost of obtaining it.

## informed search

A search strategy that uses problem-specific knowledge, typically a heuristic function estimating how close a state is to a goal, to find solutions more efficiently than an uninformed search, which is given no such clue.

## inheritance

The organizing principle of categories by which properties asserted of a category are automatically taken to hold for all its instances and subcategories; semantic networks make such inheritance reasoning convenient.

## initial state

The state in which the agent begins a search problem, from which a solution is a path leading to a goal state.

## input resolution

A resolution strategy in which every resolution step must combine one of the original input sentences (from the KB or query) with some other sentence. Its proofs have a characteristic single "spine" shape; forward and backward chaining with Modus Ponens are kinds of input resolution.

## inside-outside algorithm

A dynamic-programming algorithm that learns the rule probabilities of a probabilistic context-free grammar from example sentences, analogous to the forward-backward algorithm used for hidden Markov models.

## instance-based learning

A form of nonparametric learning, also called memory-based learning, that makes predictions directly from the stored training instances rather than from a fixed set of fitted parameters; table lookup and nearest-neighbors are examples.

## insurance premium

The difference between the expected monetary value of a lottery (gamble) and its lower certainty equivalent; because risk-averse agents will pay this amount to avoid risk, insurance premiums are positive, which is what makes the insurance industry possible.

## intelligence

The capacity for rational action: perceiving an environment and choosing actions that can be expected to achieve one's objectives. Building this capacity into machines is the aim of artificial intelligence.

## intended interpretation

The interpretation a knowledge-base designer has in mind that assigns the intended real-world objects, relations, and functions to the constant, predicate, and function symbols of a logical language.

## interleaving

Alternating computation, sensing, and acting rather than computing a complete solution before acting; for example, online search interleaves planning with execution so the agent can respond to what it observes.

## interlingua

A language-neutral intermediate representation of meaning used in machine translation: a source sentence is analyzed into the interlingua, from which a sentence in the target language is generated.

## internal state

Information about unobserved aspects of the world that a model-based agent maintains, updated from the percept history, so it can track parts of the current state it cannot directly perceive.

## interpretation

The part of a logical model that fixes the meaning of the symbols by mapping constant symbols to objects, function symbols to functions on those objects, and predicate symbols to relations. Many interpretations are possible; the intended interpretation is the one the user has in mind.

## interreflections

In vision and rendering, light that bounces from one surface onto another so that surfaces illuminate each other; these indirect contributions can significantly affect a pixel's brightness and complicate recovery of scene properties.

## intrinsic

Describing a property that belongs to the very substance of an object and is retained by any portion of it (such as density, color, or boiling point), as opposed to extrinsic properties like mass or shape that pertain to the object as a whole.

## intuition pump

A thought experiment, such as Searle's Chinese Room, constructed to elicit and shape intuitions about a philosophical question, for instance whether a machine that manipulates symbols can truly understand or think.

## inverse

The element or operation that undoes another; for a nonsingular square matrix, the inverse is the matrix that, multiplied by the original, yields the identity matrix (a singular matrix has no inverse).

## inverse entailment

An operation used in inductive logic programming that runs entailment backwards: given background knowledge and examples, it generates candidate hypotheses that would, together with the background, entail the observed examples.

## inverse kinematics

The problem of computing the joint configuration (such as joint angles) that places a robot's end effector (gripper) at a desired position and orientation in space; the inverse of the forward kinematics mapping.

## inverse logit

A model (also called the expit model) that maps a linear score to a probability using the logistic function 1/(1+e^-x), providing a soft threshold; an alternative to the probit model.

## inverse reinforcement learning

Learning the reward function by observing an agent's behavior: rather than being given the rewards, the learner infers what reward function would make the observed policy optimal.

## inverted pendulum

A classic control benchmark in which a pole hinged on a moving cart must be kept balanced upright by moving the cart; harder variants (such as a triple inverted pendulum) stress an agent's ability to stabilize an unstable system.

## inverted spectrum

A philosophical scenario in which one person's subjective color experiences are systematically swapped relative to another's while all their behavior remains identical, used to argue about qualia and the nature of subjective experience.

## IR

Abbreviation for **information retrieval**, the task of finding, within a large collection, the documents that are relevant to a user's query.

## irreversible

Describing an action (or an environment containing such actions) whose effects cannot be undone, so that the agent must act cautiously because some mistakes are permanent.

## iterative deepening search

An uninformed search that repeatedly runs depth-limited search with a steadily increasing depth limit, solving the problem of choosing a good limit. It combines the modest memory use of depth-first search with the completeness and optimality of breadth-first search.

## iterative expansion

A search technique, also known as IE, that repeatedly expands and re-expands parts of the search space under growing bounds; it is closely related to recursive best-first search (RBFS) and trades recomputation for reduced memory use.

## iterative-deepening A*

A memory-bounded variant of A* (IDA*) that is to A* what iterative deepening is to depth-first search: it performs successive searches each bounded by an increasing limit on the f-cost, giving A*'s benefits without storing all reached states, at the cost of revisiting some states.

# J

## job-shop scheduling problem

A scheduling problem consisting of a set of jobs, each made up of a collection of actions with ordering constraints among them, where each action has a duration and requires certain resources; the goal is typically to minimize the total plan duration (makespan).

## join tree

Another name for the clustering algorithms used for exact inference in Bayesian networks. Individual nodes of a multiply connected network are joined into cluster (mega)nodes so that the resulting network is a polytree, allowing posterior probabilities for all variables to be computed in O(n) time; such algorithms are widely used in commercial Bayes net tools.

## joint action

In a multiactor system with several actors, the combined action that replaces the single action of the one-actor case: it is the tuple of individual actions, one taken by each actor. Because there are exponentially many joint actions, describing the transition model and planning over them yields a huge branching factor, which motivates decoupling the actors.

## joint plan

A plan for a group of cooperating agents that specifies what action each agent must take to achieve a shared goal. It is constructed for a loosely coupled multiactor problem (e.g., a doubles tennis team), typically by pretending the actors act independently and then fixing up the interactions.

## joint probability distribution

A table (or function) giving the probability of every combination of values for a set of random variables. The full joint probability distribution covers all the variables in the model and completely specifies the probability model, since any query can be answered from it.

## JTMS

Justification-based truth maintenance system: a reason-maintenance system in which each sentence in the knowledge base is annotated with a justification—the set of sentences from which it was inferred. When a premise is retracted, the JTMS deletes exactly those sentences whose every justification included it, and it retains inference chains so retracted sentences can be efficiently restored (e.g., re-colored back in) if the premise returns.

## justification

The annotation attached to a sentence in a JTMS listing the other sentences from which it was inferred. A sentence may have any number of justifications; it is removed only when every one of its justifications is invalidated, which makes retraction and restoration of beliefs efficient.

# K

## k-anonymity

A privacy property of a data set: the data are k-anonymized if every record is indistinguishable from at least k-1 other records, so that no individual can be singled out from the group.

## k-d tree

A balanced binary tree over data with an arbitrary number of dimensions (a "k-dimensional tree"). Built like a balanced binary tree, at each node it splits the examples along one dimension at the median value, so half go to each branch, recursing until fewer than two examples remain; it is used to speed up nearest-neighbor lookup in instance-based learning.

## K-means clustering

An unsupervised clustering algorithm that partitions data into k groups by alternately assigning each point to the nearest cluster center and recomputing each center as the mean of its assigned points. (Not defined under this name in AIMA 4e; the book instead treats unsupervised clustering as learning mixtures of Gaussians via the EM algorithm. This entry gives the standard meaning consistent with that discussion.)

## Kalman filtering

An algorithm, named after Rudolf Kalman, for filtering continuous state variables (such as an object's position and velocity) from a stream of noisy observations over time. It assumes linear-Gaussian transition and sensor models, and because filtering preserves the Gaussian nature of the state distribution, the belief state remains a Gaussian summarized by its mean and covariance.

## Kalman gain matrix

The matrix in the Kalman filter's mean-update equation that weights the prediction error—the difference between the actual observation and the predicted observation—to determine how much the new observation adjusts the predicted state estimate.

## kernel

In locally weighted regression, a function of the distance between a query point and an example that determines how heavily that example is weighted: it decreases with distance and peaks at zero, so closer examples get higher weight. Its integral over the input space must be finite (choosing 1 simplifies calculations).

## kernel function

A function, usually written K(x_j, x_k), that computes the dot product of two inputs in some corresponding higher-dimensional feature space. By replacing the dot product in a linear-separator algorithm with a kernel function, one can learn in that feature space while computing only kernel values rather than the full feature vectors. By Mercer's theorem, any reasonable kernel function corresponds to some feature space.

## kernel trick

The technique of plugging kernel functions in place of dot products so that optimal linear separators can be found efficiently in feature spaces with billions of, or even infinitely many, dimensions. Mapped back to the original input space, these separators can correspond to arbitrarily nonlinear decision boundaries.

## kinematic state

A robot state described only by configuration (the positions of its parts), as opposed to a dynamic state that also includes velocity. The kinematic state is not sufficient for computing exactly what torque to apply, since torque depends on velocity and acceleration.

## kinematics

The relationship between a robot's configuration and the locations of points on the robot. Forward kinematics maps a configuration to the location of a point on the robot; inverse kinematics maps a desired location (and sometimes orientation) for a point back to the configuration(s) that achieve it.

## King Midas problem

The failure that arises when a machine is given a fixed objective that it optimizes literally but which does not capture what the designer really wanted—like King Midas, who wished everything he touched would turn to gold and then regretted it. AIMA uses it to motivate moving away from the standard model of putting definite objectives into machines toward agents that are uncertain about the true human objectives.

## knowledge acquisition

The step in knowledge engineering in which the knowledge engineer—possibly working with domain experts—extracts and assembles the relevant knowledge of a domain. At this stage the knowledge is not yet represented formally; the goal is to understand the scope of the knowledge base and how the domain actually works.

## knowledge base

The central component of a knowledge-based agent: a set of sentences, each expressed in a knowledge representation language and asserting something about the world. Sentences taken as given (not derived) are axioms; sentences are added with TELL and queried with ASK, both of which may involve inference.

## knowledge engineering

The general process of knowledge-base construction, carried out by a knowledge engineer who investigates a domain, learns which concepts matter, and creates a formal representation of the objects and relations in it. It is suited to special-purpose knowledge bases whose domain is circumscribed and whose range of queries is known in advance.

## knowledge level

The level of description at which an agent is characterized purely by what it knows and what its goals are, allowing its behavior to be predicted independently of how that knowledge is implemented internally.

## knowledge representation

The subfield of AI concerned with how to store what an agent knows or perceives, expressing assertions about the world so they can be used to answer questions and draw conclusions.

## knowledge representation language

The formal language in which the sentences of a knowledge base are expressed, each sentence stating some assertion about the world.

## knowledge-based agents

Agents that use a process of reasoning over an internal representation of knowledge to decide what actions to take. Their central component is a knowledge base plus an inference mechanism, so that (like humans who "know things") what they know helps them decide what to do.

## Known

A property of an environment referring not to the environment itself but to the agent's or designer's state of knowledge about its "laws of physics." In a known environment the outcomes (or outcome probabilities, if nondeterministic) of all actions are given; in an unknown one the agent must learn how the environment works. This distinction is separate from full versus partial observability.

## Kolmogorov complexity

Also called algorithmic complexity; a formal measure of the simplicity of an object, defined as the length of the shortest program (for a universal Turing machine) that produces it. It gives a mathematical basis for the notion of simplicity used in Ockham's razor.

## Kolmogorov's axioms

The basic axioms of probability theory (attributed to Andrei Kolmogorov): probabilities are non-negative, the probability of the certain event is 1, and the probability of a disjunction of mutually exclusive events is the sum of their probabilities. All of probability theory can be built up from these.

## Kriegspiel

A partially observable, deterministic variant of chess in which each player sees only their own pieces, the opponent's pieces being completely invisible. A referee who sees all pieces adjudicates and announces information such as illegal moves, captures, and checks, from which a player must infer the opponent's positions by maintaining a belief state.

## Kullback-Leibler divergence

A nonnegative measure of the difference between two probability distributions, defined as the expected log ratio between them (an average, under one distribution, of log(Q/P)). It equals zero exactly when the distributions coincide; minimizing cross-entropy for a fixed distribution is equivalent to minimizing the KL divergence, and it underlies the variational lower bound (ELBO) in variational inference.

# L

## label

In supervised learning, the output that accompanies an input example; for example, an image accompanied by an output saying "bus" or "pedestrian." The agent learns a function that, given a new input, predicts the appropriate label.

## Lambert's cosine law

In image formation, the rule that the brightness of a diffuse surface patch is given by I = rho*I0*cos(theta), where rho is the diffuse albedo, I0 the source intensity, and theta the angle between the incoming light and the surface normal.

## landmark points

A small set of fixed locations (also called pivots or anchors) whose optimal path costs to all other nodes are precomputed, so that these stored distances can be combined into an admissible heuristic that speeds up route-finding search.

## landmarks

Stable, recognizable features of the environment that a robot's sensors can detect; for each landmark the sensor reports its range and bearing, which the robot uses for localization and mapping. (AIMA also uses "landmark points" as reference nodes for computing differential heuristics in search.)

## language

A formal language is defined as a set of strings, where each string is a sequence of symbols; because the languages of interest are typically infinite, the set is characterized concisely by a grammar (in AIMA, a context-free grammar written in Backus-Naur form). AIMA defines the languages of propositional logic, first-order logic, and a subset of English this way, alongside the natural languages used by humans.

## language generation

The task of producing natural-language text (or speech) from an internal representation of meaning or intent.

## language identification

The task of determining which natural language a given piece of text or speech is written or spoken in.

## language model

A model that assigns scores or probabilities to strings in a language, used for example to evaluate how plausible a proposed sentence is in the target language.

## large-scale learning

Learning in settings where the amount of training data is so large that the main limitation is computation time rather than the availability of examples.

## late move reduction

A game-tree search technique that assumes move ordering is good, so moves appearing later in the ordered move list are less promising and are searched to reduced depth rather than pruned outright.

## laws of robotics

Isaac Asimov's proposed ethical rules governing robot behavior, the first attempt to address robot ethics through built-in constraints on how robots may act toward humans and themselves.

## layers

The groups of units in a neural network, arranged so that the outputs of one layer feed as inputs to the next, from the input layer through hidden layers to the output layer.

## leak node

In a noisy-OR model, an extra parent added to cover "miscellaneous causes" when not all of an effect's true causes have been explicitly listed.

## learn

To improve performance or adapt behavior on the basis of experience, allowing an agent to cope with new circumstances and to detect and extrapolate patterns in its inputs.

## learning

An agent is learning if it improves its performance after making observations about the world.

## learning curve

A plot used to evaluate a learning algorithm, showing how the accuracy of the learned hypothesis on a test set changes as the size of the training set grows; also called a happy graph.

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

## Lexical ambiguity

Ambiguity that arises when a single word has more than one meaning, such as "back" being usable as an adverb, adjective, noun, or verb.

## lexical category

A class of words that share syntactic behavior, such as noun, verb, adjective, or preposition; also called a part of speech.

## lexicalized PCFG

A type of augmented probabilistic context-free grammar that assigns probabilities based on the actual words in a phrase (for example, its head word), not just on the syntactic categories.

## lexicon

The list of words in a language together with information about their categories and properties, used by a grammar.

## LIFO queue

A last-in, first-out queue (a stack), in which the most recently added element is removed first; used to implement depth-first search.

## lifting lemma

The result in first-order theorem proving guaranteeing that any proof using ground (variable-free) sentences can be reproduced at the level of general first-order sentences via unification.

## likelihood

The probability of the observed data under a given hypothesis, P(data | hypothesis). In Bayesian learning it is combined with the hypothesis prior via Bayes' rule to obtain the posterior probability of each hypothesis.

## likelihood weighting

An approximate inference method for Bayesian networks that samples only the non-evidence variables while weighting each sample by the probability of the evidence given its sampled parents.

## limited rationality

Acting as appropriately as possible when there is not enough time or computational resources to perform all the reasoning that perfect rationality would require.

## line search

A procedure in continuous optimization that, given a search direction, finds a good step size by searching along that direction.

## linear constraints

Constraints in which each variable appears only in linear form; special-purpose algorithms exist to solve constraint problems with linear constraints on integer variables, unlike the general nonlinear case, which is undecidable.

## linear Gaussian

A distribution in which a variable is normally distributed with a mean that is a linear function of its parents and a fixed variance, used in continuous Bayesian networks.

## linear interpolation smoothing

A language-model smoothing technique that estimates n-gram probabilities as a weighted combination of unigram, bigram, trigram, and higher-order estimates.

## linear planning

Planning that treats a plan as a strictly ordered (total-order) sequence of actions and works on one subgoal completely before moving to the next, without interleaving steps from different subgoals.

## linear programming

A constrained optimization method in which the objective function is linear and the constraints are linear inequalities forming a convex set. Its time complexity is polynomial in the number of variables, and it is a special case of convex optimization.

## linear regression

A supervised learning method that fits a linear function of the input variables to predict a continuous output, typically by minimizing squared error.

## linear resolution

A resolution strategy in which each resolution step uses the most recently derived clause together with another clause, forming a linear chain of inferences.

## linear separator

A hyperplane that divides the input space into two regions, used to classify examples into two classes when the data are linearly separable.

## linear temporal logic

A modal logic for reasoning about time that adds operators such as X ("next"), F ("eventually"), G ("always"), and U ("until") to express how propositions hold over a sequence of time steps.

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

## local beam search

A search algorithm that keeps track of k states at once rather than a single state, generating successors of all k and retaining the k best at each step.

## local consistency

A property of a constraint satisfaction problem in which subsets of variables (single variables, pairs, or larger groups) satisfy their constraints, enforced by algorithms that prune inconsistent values.

## local search

A search technique that operates by moving from a start state to neighboring states without keeping track of the paths taken or the set of states already reached. It uses very little memory and can find reasonable solutions in large or infinite state spaces, but is not systematic. It is also used to solve optimization problems.

## localist representation

A representation in which there is a one-to-one mapping between concepts and memory locations, so each concept is stored in a single distinct location (as opposed to a distributed representation).

## locality

The property that each action typically changes only a small number of fluents rather than the whole world state; exploiting locality lets the transition model be described compactly. (AIMA also uses "locality" for the property of algorithms like depth-first and hill-climbing search that expand nodes near the current one.)

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

## logical connectives

Operators such as AND, OR, NOT, implication, and biconditional used to build complex sentences from simpler ones in logic.

## logical equivalence

The relation between two sentences that are true in exactly the same set of models.

## logical inference

The process of deriving new conclusions that are entailed by a knowledge base; for example, model checking carries out inference by enumerating models to confirm that a sentence holds in every model where the knowledge base is true.

## logical minimization

A technique for defining an object as the smallest one satisfying certain conditions; used, for example, to define the "bunch" composed of a set of parts as the smallest object having all of them as parts.

## logical omniscience

The idealized assumption that a reasoning agent instantly knows all the logical consequences of everything it believes.

## logical positivism

The doctrine, developed by the Vienna Circle, holding that all knowledge can be characterized by logical theories ultimately connected to observation sentences corresponding to sensory input, thereby combining rationalism and empiricism.

## logicist

Pertaining to the approach to AI that builds intelligent systems on formal logic, representing knowledge in logical sentences and acting by logical inference.

## logistic function

The S-shaped function 1/(1 + e^(-x)) that maps any real number to the interval (0, 1), widely used as a soft threshold in classification and neural networks.

## logistic regression

The process of fitting, by minimizing loss, the weights of a hypothesis that passes a linear combination of the inputs through the logistic function. The output, a number between 0 and 1, forms a soft decision boundary and can be interpreted as the probability of belonging to a class. It has no closed-form solution but is fit by gradient descent.

## long-distance dependencies

Grammatical relationships between words that can be separated by arbitrarily much intervening text, such as the link between a question word and the gap it fills.

## LOOCV

An abbreviation for leave-one-out cross-validation, in which each of the n examples in turn serves as the single test case while the rest are used for training.

## loopy belief propagation

An approximate inference method that applies the belief-propagation message-passing algorithm to a graphical model containing loops (cycles), for which exact convergence is not guaranteed but which often works well in practice.

## loopy path

A path in a search that revisits a state already on the path, forming a cycle.

## loosely coupled

Describing a system whose components interact through limited, well-defined connections, so that each can operate largely independently of the others.

## loss function

A function measuring the cost of a prediction, giving the amount of utility lost when a learner outputs one value while the true value is another; learning aims to minimize expected loss.

## lottery

In utility theory, the set of possible outcomes of an action together with their probabilities, thought of as a ticket; lotteries are the objects over which an agent expresses its preferences.

## low impact

A design principle for aligning an agent's objectives whereby, instead of purely maximizing utility, the agent maximizes utility minus a penalty summarizing all changes it makes to the state of the world, to avoid unintended side effects.

## low-dimensional embedding

A representation that maps high-dimensional data into a space with far fewer dimensions while preserving important structure such as distances or neighborhoods.

## LSTM

Long short-term memory; a special recurrent-network layer design that uses gating mechanisms (such as a forget gate) to control its internal memory, helping avoid the vanishing- and exploding-gradient problems when learning over long sequences.

# M

## machine evolution

An early approach (now called genetic programming) that attempts to generate a well-performing program by applying random small mutations to machine-code programs and using a selection process to preserve the useful ones.

## machine learning

The process by which a computer agent observes data, builds a model based on that data, and uses the model (a hypothesis about the world, executable as a function) to guide its future actions.

## machine reading

A form of information extraction that operates at Web scale with no human input, learning to read text on any topic and build its own knowledge base without hand-labeled training examples.

## Machine translation

The task of automatically transforming text (or speech) from one natural language into another.

## macrops

Macro-operators: generalized operators, each consisting of a sequence of primitive planning steps, that are learned and stored so an entire sequence can later be reused as a single step; introduced in the STRIPS planner and a precursor to hierarchical planning.

## magic set

A deductive-database technique for making forward chaining relevant to a query: the rule set is rewritten using information from the goal so that forward inference considers only the relevant variable bindings—those in the "magic set"—effectively blending forward inference with backward-directed preprocessing.

## Mahalanobis distance

A distance metric, more complex than simple normalized Euclidean distance, that takes into account the covariance between dimensions when measuring the distance between points.

## Maintaining Arc Consistency

A constraint-satisfaction algorithm (MAC) that, after each variable assignment during search, enforces arc consistency to detect and prune inconsistencies early.

## Maintaining Arc Consistency (MAC)

A CSP inference algorithm that, after each variable X is assigned during backtracking search, runs AC-3 starting from the arcs of X's unassigned neighbors to prune inconsistent values, backtracking if any domain becomes empty; it is strictly more powerful than forward checking.

## makespan

The total duration of a plan or schedule; often used as the cost function to minimize in scheduling problems such as job-shop scheduling, where actions may run in parallel.

## Manhattan distance

The distance between two points measured as the sum of the absolute differences of their coordinates; the Minkowski (L^p) distance with p = 1.

## margin

The width of the region bounded by the lines around a linear separator—twice the distance from the separator to the nearest example points; a support vector machine seeks the maximum margin separator, the boundary that lies as far as possible from the seen examples.

## marginal probability

An unconditional probability for a variable or set of variables, obtained by summing (marginalizing) a joint distribution over all values of the other variables.

## marginalization

The process (also called summing out) of obtaining the distribution over a subset of variables by summing the joint probabilities over all possible values of the other variables, thereby removing them.

## Markov assumption

The assumption that the current state depends on only a finite, fixed number of previous states; in a first-order Markov process it depends only on the immediately preceding state.

## Markov blanket

A node's parents, its children, and its children's other parents; given its Markov blanket, a variable is conditionally independent of all other nodes in the network—a property exploited by inference methods such as Gibbs sampling.

## Markov chain

A random process that generates a sequence of states in which each next state is produced by making a random change to the current state; it underlies Markov chain Monte Carlo methods such as Gibbs sampling.

## Markov chain Monte Carlo

A family of approximate-inference algorithms (MCMC) that generate a sequence of samples by making random transitions between states (a Markov chain), used to estimate posterior probabilities in models such as Bayes nets.

## Markov decision process

A sequential decision problem for a fully observable, stochastic environment with a Markovian transition model P(s'|s,a) and additive rewards, defined by a set of states (with an initial state), the actions available in each state, the transition model, and a reward function; its solution is a policy.

## Markov localization

The general probabilistic formulation of robot localization: the robot's pose is estimated by maintaining a probability distribution over locations that is updated by a Markov (Bayesian) filter as the robot moves and senses.

## Markov network

An undirected form of graphical model (also called a Markov random field) that represents a joint distribution through potentials over cliques; a Bayesian network can be converted to an equivalent undirected Markov network.

## Markov property

The assumption that the future is independent of the past given the present—that is, the current state depends only on a fixed finite number of previous states (usually just the immediately preceding one). It is violated when there is systematic dependence on earlier history (such as cumulative battery drain), which can be repaired by enlarging the state.

## Markov reward process

A Markov decision process (MRP) with only a single available action, so that it produces a fixed sequence of state transitions and rewards; used, for example, to model each arm of a bandit problem.

## material value

In a game-playing evaluation function (as in chess), the value contributed by the pieces each side holds—for example, pawn = 1, knight or bishop = 3, rook = 5, queen = 9—summed as features in a weighted linear evaluation of the position.

## materialism

The view, opposed to dualism, that the brain's operation entirely according to the laws of physics constitutes the mind; also called physicalism or naturalism. On this view free will is just the way the perception of available choices appears to the chooser.

## matrix

A rectangular array of numbers used throughout the book to represent linear operators and factors—for example, the Hessian matrix of second derivatives in optimization, or the transition matrix of an HMM whose entries are indexed by the values of its variables.

## max norm

A way of measuring the length of a vector as the absolute value of its largest component, so that the distance between two utility vectors is the maximum difference between corresponding elements; it is used to prove that the Bellman update is a contraction in value iteration.

## max-level

A planning-graph heuristic that estimates the cost of a conjunctive goal as the first level in the graph at which all the goal literals appear together without a mutex among them; it is admissible. (This label is not used in AIMA 4th ed., which instead presents ignore-preconditions, ignore-delete-lists, and subgoal-independence heuristics.)

## max-pooling

A pooling operation in neural networks that summarizes each group of inputs by reporting only the maximum value in the group, reducing the size of the representation.

## maximin

Von Neumann's criterion for zero-sum games: a player chooses the (mixed) strategy that maximizes its minimum guaranteed payoff, assuming the opponent will respond so as to minimize it. Every two-player zero-sum game has a maximin equilibrium under mixed strategies, and the maximin strategy guarantees at least the game's value.

## maximin equilibrium

The mixed-strategy solution that von Neumann's theorem guarantees for every two-player zero-sum game when mixed strategies are allowed; it is also a Nash equilibrium, and a player using the maximin strategy is guaranteed at least the value of the game even if the strategy is revealed to the opponent.

## maximum a posteriori

Making predictions based on the single most probable hypothesis (MAP)—the one maximizing the posterior P(h|d)—rather than averaging over all hypotheses as in full Bayesian learning; it turns learning into an optimization, and its prior naturally embodies Ockham's razor by penalizing complexity.

## maximum expected utility

The fundamental principle of decision theory (MEU): a rational agent should choose the action that yields the highest expected utility, averaging the utilities of the possible outcomes weighted by their probabilities.

## maximum margin separator

The decision boundary chosen by a support vector machine: the linear separator positioned to have the largest possible distance to the nearest training examples, which helps generalization.

## maximum-likelihood

Choosing the hypothesis that maximizes the likelihood of the data P(d|h), ignoring any prior over hypotheses; reasonable when no hypothesis is favored a priori and a good approximation to Bayesian/MAP learning with large data sets, but unreliable on small ones.

## mechanism

The "right game" designed for a collection of agents to play, consisting of a language of allowable strategies, a distinguished agent called the center that collects the agents' strategy choices (e.g., an auctioneer), and an outcome rule, known to all, that the center uses to determine each agent's payoff.

## mechanism design

The problem of defining the rules of the game that agents must play so that the collective good is maximized when each agent selfishly maximizes its own utility—for example, designing routing protocols so each router is incentivized to act in a way that maximizes global throughput.

## mel frequency cepstral coefficient (MFCC)

A compact set of acoustic features extracted from a short window of a speech signal, obtained by applying a perceptually spaced (mel-scale) filter bank to the signal's spectrum and taking the discrete cosine transform of the log filter-bank energies; the standard input features for speech-recognition acoustic models. (Not defined in AIMA 4th ed., which dropped the dedicated speech chapter.)

## memoization

A technique for alleviating the redundant subcomputations and infinite loops that afflict backward chaining and logic programming, by caching previously computed results so they can be reused rather than recomputed (developed extensively as tabled logic programming).

## memoized

Describing a computation whose results have been cached (memoized) so that the same subcomputation is reused rather than repeated on a later call with the same inputs.

## mental states

In Searle's biological naturalism, high-level emergent features of the brain that are caused by low-level physical processes in the neurons, where it is the physical properties of the neurons that determine whether there is genuine understanding.

## Mercer's theorem

A result in mathematics stating that any "reasonable" kernel function (one whose kernel matrix is positive definite) corresponds to an inner product in some feature space, justifying the kernel trick.

## mereology

Leśniewski's theory (Greek for "part") that uses the part-whole relation as a substitute for mathematical set theory, aiming to eliminate abstract entities such as sets—for example, treating "butter" as a single object consisting of all buttery objects in the universe.

## metadata

Descriptive data about content—such as information accompanying a piece of music, video, or text—used (alongside content analysis and user history) by recommender systems.

## metalevel learning

Learning from experience in the metalevel state space—for example, from past search missteps—so as to avoid exploring unpromising subtrees in future searches, with the aim of minimizing the total cost of problem solving.

## metalevel state space

A state space whose states capture the internal computational state of a program searching in an ordinary object-level state space (for example, A*'s current search tree), where each action is a computation step—such as expanding a leaf node—that alters that internal state.

## metaphor

A figure of speech in which a phrase with one literal meaning is used to suggest a different meaning by way of an analogy; it can be viewed as a kind of metonymy in which the relation between the two things is one of similarity.

## metareasoning

Reasoning about what computations to do—"reasoning about reasoning." Since computations have costs and some likelihood of improving the quality of a decision, metareasoning selects high-utility computations (such as node expansions) and stops when no computation's expected utility exceeds its cost.

## metonymy

A figure of speech in which one object is used to stand for another related one—for example, "Chrysler announced a new model" meaning that a spokesperson did so—including producer-for-product ("I drive a Honda") and part-for-whole ("the team needs a strong arm") cases.

## Metropolis-Hastings

A general form of MCMC sampling that allows flexible proposal of new samples, accepting or rejecting them so that the chain converges to the desired stationary distribution.

## micromort

A unit representing a one-in-a-million chance of death, used in medical and safety analysis to elicit and measure the value people place on their own lives and on small risks (for example, driving about 230 miles in the UK incurs roughly one micromort).

## min-conflicts

A local-search heuristic for constraint satisfaction that repeatedly picks a conflicted variable and reassigns it the value resulting in the minimum number of conflicts with other variables; it is remarkably effective, solving problems such as n-queens in roughly constant time regardless of size.

## mind-body problem

The philosophical question of how the mind relates to matter: on the dualist view part of the mind lies outside physical law, whereas on the materialist (physicalist) view the brain's operation under the laws of physics simply constitutes the mind.

## minibatch

A small group of m examples selected out of the full training set of N examples, used to compute the gradient estimate at each step of stochastic gradient descent.

## minimax

The approach to optimal play in deterministic, two-player, zero-sum games: the minimax value of each state, computed under the assumption that both players play optimally to the end of the game, is used to determine MAX's best strategy.

## minimax decision

The optimal move for MAX at the root of a game tree: the action leading to the successor state with the highest minimax value, assuming MIN plays optimally in reply.

## minimax search

A depth-first game-tree search algorithm that recurses to the leaves, applies the utility function there, and backs up minimax values as the recursion unwinds—MAX nodes taking the maximum and MIN nodes the minimum of their children's values—to select MAX's optimal move.

## minimax value

The utility (for MAX) of being in a given state, assuming both players play optimally from that state to the end of the game: a terminal state's minimax value equals its utility, while at non-terminal states MAX takes the maximum and MIN the minimum over the successors' minimax values.

## minimum description length

A hypothesis-selection principle (MDL) that chooses the hypothesis minimizing the total number of bits needed to encode both the hypothesis and the data given that hypothesis; it expresses a preference for simplicity directly by bit-counting, rather than through a prior as in MAP learning.

## minimum slack

A greedy scheduling heuristic that, on each iteration, selects among the actions whose predecessors are all scheduled the one with the least slack (smallest gap between earliest and latest start) and schedules it at its earliest possible start, then updates the times and repeats.

## minimum-remaining-values

A CSP variable-ordering heuristic (MRV, also called "most constrained variable" or "fail-first") that selects next the unassigned variable with the fewest remaining legal values, so that likely failures are detected early and the search tree is pruned.

## Minkowski distance

The L^p norm commonly used to measure the distance between a query point and an example point: the p-th root of the sum of the p-th powers of the coordinate differences. With p=2 it is Euclidean distance and with p=1 it is Manhattan distance.

## missing effect

A flaw in an action model in which one of the action's actual consequences is not represented (e.g., painting an object also getting paint on the floor).

## missing precondition

A flaw in an agent's action model in which a precondition truly required for an action to succeed is absent from the model (for example, not knowing that opening a paint can requires a screwdriver); such errors can cause the plan to fail and trigger replanning.

## missing state variable

A relevant state variable (fluent) that is entirely absent from an agent's representation—for example, the amount of paint left in a can when the model has no notion of it—leaving the model incomplete and its behavior potentially fragile. The book discusses this as a missing fluent.

## mixed strategy

A randomized policy in a game in which the agent selects among its available actions according to a probability distribution, rather than always playing one fixed action.

## mixing rate

The rate at which a Markov chain (such as one defined by Gibbs sampling) converges toward its stationary distribution; it depends on the properties of the conditional distributions involved.

## mixing time

Roughly, the number of steps a Markov chain takes to reach (approach) its stationary/fixed-point distribution.

## mixture density

A neural network output layer that models continuous outputs using a mixture of Gaussian distributions.

## mixture distribution

A distribution built from several components, each a distribution in its own right: a data point is generated by first choosing a component (via a component random variable) and then sampling from that component's distribution.

## mixture of Gaussians

A mixture distribution whose components are multivariate Gaussians—a natural choice for continuous data—with the mixture weights and component parameters typically learned via the EM algorithm.

## mobile manipulator

A robot that combines a mobile base (using wheels, legs, or rotors to move about the environment) with one or more manipulator arms, so that it can both navigate and physically manipulate objects.

## mobile robot

A robot that moves about its environment, in contrast to a fixed manipulator; examples include unmanned ground vehicles (UGVs).

## modal logic

A logic designed to go beyond the single modality of truth by adding modal operators that take whole sentences (rather than terms) as arguments—for example, a "knows" operator such as Knows_a(sentence)—while otherwise sharing the syntax of first-order logic.

## modal operators

Special operators in modal logic that take whole sentences (rather than terms) as arguments, allowing statements about, for example, knowledge, belief, or time.

## model

In logic, a mathematical abstraction taking the place of a "possible world" that fixes a truth value for every relevant sentence; if a sentence α is true in model m, then m satisfies α, or m is a model of α.

## model checking

An inference approach that determines entailment (KB ⊨ α) by enumerating all possible models and checking that α is true in every model in which the knowledge base is true; it applies when the space of models is finite.

## model preference

A form of nonmonotonic logic in which a sentence is entailed (by default) if it holds in all preferred models of the knowledge base, rather than in all models; e.g., circumscription prefers models with fewer abnormal objects.

## model selection

The learning subtask of choosing a good hypothesis space (as opposed to optimization/training, which finds the best hypothesis within a given space); a partly qualitative choice for which "hypothesis space selection" would be a more accurate name.

## model-based agent

An agent that maintains an internal state, based on its percept history and a model of how the world works, to track aspects of the world it cannot currently observe.

## modified policy iteration

A variant of policy iteration in which the policy-evaluation step is only approximated (e.g., by a few value-update iterations), often making it more efficient than standard policy iteration or value iteration.

## modus ponens

A sound rule of inference (Latin, "mode that affirms"): whenever sentences of the form α ⇒ β and α are given, the sentence β can be inferred.

## momentum

An optimization technique that keeps a running average of past gradients (across minibatches) to smooth updates and compensate for noisy or small-minibatch gradient estimates.

## monitoring

Maintaining one's belief state over time in a partially observable environment by recursively updating the previous belief state from each new percept rather than reprocessing the whole percept sequence; also called filtering or state estimation. (In planning, execution monitoring instead checks during execution whether replanning is needed.)

## monotonic

A property of path costs in search whereby the g cost never decreases as a path is extended (strictly monotonic if it always increases), since action costs are nonnegative.

## monotonic preference

An agent has a monotonic preference for more of a good (such as money) when, all else equal, it always prefers more to less; on its own this does not make the quantity a utility function, since it says nothing about preferences among lotteries.

## monotonicity

In logic, the property that the set of entailed sentences can only grow as information is added to the knowledge base, so a previously drawn conclusion is never invalidated by new premises. The term is also used of search path costs (which increase monotonically along a path when action costs are positive) and of the f-cost in A* (monotonic exactly when the heuristic is consistent).

## Monte Carlo

Referring to randomized sampling algorithms that give approximate answers whose accuracy improves with the number of samples: random events are generated according to the probabilities in a model and the outcomes are counted, converging to the true distribution as the number of samples grows.

## Monte Carlo localization

A robot-localization algorithm (MCL) that is an instance of the particle-filtering algorithm supplied with the appropriate motion and sensor models: the belief over the robot's pose is represented by a set of particles that are updated as the robot moves and senses.

## Monte Carlo planning

An approach that computes an approximately optimal policy by sampling many possible future trajectories rather than exhaustively solving the decision problem.

## Monte Carlo simulation

An estimation technique that runs many randomized simulations of a process and aggregates their outcomes to approximate probabilities or expected values.

## Monte Carlo tree search

A search strategy (MCTS) that estimates the value of a state not with a heuristic evaluation function but as the average utility over many simulated playouts (rollouts) of complete games played from that state to a terminal position, growing a search tree toward the most promising moves.

## most general unifier

The unifier (MGU) of a pair of unifiable expressions that makes the fewest commitments, unique up to renaming and substitution of variables.

## most likely state

The single state extracted from a probability distribution (produced by state estimation) that has the highest probability, often used as a point estimate for planning.

## most probable explanation

The most likely joint assignment to the nonevidence variables of a probabilistic model given the observed evidence (abbreviated MPE).

## motion blur

The blurring effect that occurs when an object moves a significant distance during the camera's exposure interval, so that its image is smeared across the sensor rather than sharply focused.

## motion model

For a robot, the transition model P(X_{t+1} | x_t, a_t) giving the probability distribution over the next state as a function of the current state and the action taken; together with the sensor model it drives probabilistic localization and filtering.

## move

In the context of games, a synonym for "action" (just as "position" is used as a synonym for "state").

## multiactor

A generic label covering multieffector, multibody, and multiagent settings, where "actor" is the generic term for effectors, bodies, and agents; the aim is to define transition models, correct plans, and planning algorithms for such settings.

## multiagent

Describing an environment (as opposed to single-agent) in which another entity is best modeled as an agent whose behavior maximizes a performance measure that depends on our agent's behavior—for example, a chess opponent—rather than as a mere object obeying physical laws.

## multiagent planning problem

The decision-making problem faced by an agent in a multiagent system; its precise nature and the techniques appropriate for solving it depend on the relationships (cooperative, competitive, or mixed) among the agents in the environment.

## multiagent systems

Environments that contain multiple actors (agents), in which each agent must make its own decisions while taking the presence and behavior of the other agents into account.

## multiattribute utility theory

The theory used to handle decision problems whose outcomes are characterized by two or more attributes, combining these attributes into an overall utility.

## multibody planning

Planning for multiple effectors that are physically decoupled into separate units (such as a fleet of delivery robots), while still coordinating them centrally.

## multieffector planning

Planning for a single agent with multiple effectors that operate concurrently (e.g., typing and speaking at once), managing positive and negative interactions among the effectors.

## multiple inheritance

A situation in a knowledge representation in which an object belongs to more than one category, or a category is a subset of more than one other category, so it inherits properties from several sources.

## multiplexer

A conditional-distribution structure in which one designated selector parent determines which of the other parent variables actually governs the value of the child variable.

## multiplicative utility function

The form of multiattribute utility function implied when a set of attributes is mutually utility independent (MUI): the overall utility is built by multiplicatively combining the single-attribute utility functions.

## multiply connected

Describing a Bayesian network that is not singly connected (not a polytree)—that is, one in which some pair of nodes is joined by more than one undirected path; exact inference in such networks can be exponential in the worst case, since Bayes net inference is NP-hard.

## multitasking

The ordinary term for a bandit superprocess: performing several full decision processes (tasks) at once rather than a single one.

## multivariate Gaussian

The generalization of the normal distribution to a vector-valued random variable in d dimensions, parameterized by a mean vector and a covariance matrix.

## multivariate linear regression

Linear regression in which each input example is an n-element vector, giving hypotheses of the form h_w(x) = w_0 + w_1 x_1 + ... + w_n x_n (one weight per feature plus an intercept). The book calls this multivariable linear regression, reserving "multivariate" for vector-valued outputs, though it notes the terms are often used interchangeably.

## mutation

In genetic algorithms, a variation operator that randomly alters an offspring's representation—for example, flipping each bit independently with probability equal to a small mutation rate—introducing new diversity into the population.

## mutation rate

A parameter of an evolutionary/genetic algorithm that determines how often offspring undergo random mutations to their representation.

## mutex

A mutual-exclusion link in a planning graph marking that two actions or two literals at the same level cannot co-occur—for actions because of inconsistent effects, interference, or competing needs, and for literals because of inconsistent support.

## mutual preferential independence

A condition (MPI) holding when each attribute may matter to preferences but no attribute affects the way the others are traded off against each other; MPI justifies representing preferences with an additive value function.

## mutually utility independent

A condition (MUI) on a set of attributes holding when every subset of the attributes is utility-independent of the remaining ones; MUI implies that preferences can be represented by a multiplicative utility function.

## myopic

Describing a shortsighted information-gathering strategy that applies the value-of-information formula as if only a single evidence variable will be acquired; because it ignores the value of combining observations, it can behave suboptimally.

# N

## n-armed bandit

By analogy with a one-armed bandit (slot machine), a formal decision model with n levers, each hiding a fixed but unknown probability distribution of winnings that is sampled each time the lever is pulled. On each turn the agent must choose a lever, embodying the ubiquitous tradeoff between exploitation of the current best action and exploration to gain information. It models many real problems, such as which new treatment, investment, research project, or advertisement to try.

## n-gram model

A Markov chain model of language that considers only the dependence among adjacent words: the probability of each word depends on just the n-1 previous words. A sequence of symbols of length n is an n-gram, with special cases unigram (1-gram), bigram (2-gram), and trigram (3-gram). Such models work well for text classification tasks like spam detection, sentiment analysis, and author attribution.

## naive Bayes

A probability model that assumes the effect variables are conditionally independent given the cause variable. It is called "naive" because this independence assumption is used as a simplification even in cases where the effects are not actually independent.

## natural kind

A category of real-world things (the standard example is tomatoes) that has no clear-cut definition, unlike categories with strict definitions such as triangle. Rather than a complete definition, a natural kind is identified by a set of typical features, so most knowledge about it concerns its typical instances. Natural kinds cannot be completely defined in logic, though their properties can be described.

## natural language processing

The capability, needed by an agent, to communicate successfully in a human language such as English.

## natural numbers

The nonnegative integers, definable in logic with a predicate NatNum, the constant 0, and a successor function S. The Peano axioms define natural numbers recursively (NatNum(0), and every successor of a natural number is a natural number) and define addition, showing how a large theory can be built from a tiny kernel of axioms.

## naturalism

The materialist view, opposed to dualism, holding that the mind is constituted by the brain operating according to the laws of physics.

## nearest-neighbor filter

An approximate data-association method for multiple-object tracking that, at each time step, repeatedly chooses the closest pairing of a predicted object position and an observation and adds that pairing to the assignment. It works well when objects are well separated and prediction and observation errors are small, so there is little possibility of confusion.

## nearest-neighbors regression

A nonparametric regression method: given a query point, find the k training examples nearest to it and predict the output by taking the mean or median of their outputs, or by solving a local linear regression on those neighbors.

## negation

The logical connective not (¬); a sentence such as ¬P is called the negation of P. A negated atomic sentence is a negative literal (as opposed to a positive literal, which is an atomic sentence).

## negative

In Boolean classification, an example whose output is false (a negative example), as opposed to a positive example whose output is true.

## Netflix Prize

A machine learning competition in which de-identified individual movie ratings were released and competitors were challenged to build an algorithm that could accurately predict which movies a person would like.

## neuroscience

The study of the nervous system, particularly the brain, which informs AI by revealing how biological systems process information with networks of neurons.

## Newton-Raphson

A general iterative technique for finding roots of a function (solving f(x)=0) by repeatedly updating the estimate via Newton's formula using the derivative. For optimization it is applied to the gradient, so finding a maximum or minimum requires the Hessian (matrix of second derivatives) in the update.

## no-good

In constraint satisfaction, a set of variables together with values shown to be inconsistent (leading to no valid solution). Recording the no-good—as a new constraint or in a separate cache—lets forward checking or backjumping avoid re-exploring that combination of assignments.

## no-regret learning

Online learning in which, by choosing the learning parameter appropriately, the agent's performance asymptotically approaches that of the best fixed expert or strategy, so the average amount of regret per trial tends to 0 as the number of trials increases.

## node

A data structure representing a point in a search tree, holding the corresponding state, a pointer to the parent node, the action that generated it from the parent, and the total path cost from the initial state.

## noise

Errors or unpredictable variation in data—for example when two examples have the same description but different classifications because of an error, a nondeterministic domain, or an unobserved distinguishing attribute. By definition noise cannot be predicted, only characterized, and trying to fit it exactly leads to overfitting.

## noisy channel model

A probabilistic model, used in tasks like spelling correction and machine translation, that treats an observed signal as a correct message passed through a channel that corrupts it, and recovers the intended message by combining a prior over messages with a model of the channel's distortions.

## noisy-OR

A noisy logical relation generalizing the logical OR, used to compactly specify the CPT of a Boolean child with several possible causes. It allows uncertainty about each parent's ability to cause the child: each cause may be independently inhibited, so the child is false if and only if all its true parents are inhibited, with probability equal to the product of the individual inhibition probabilities.

## nondeterministic

Said of an environment whose next state is not completely determined by the current state and the action executed. (A deterministic but partially observable environment can appear nondeterministic to the agent.)

## nonholonomic

Describing a system, such as a car, whose controllable degrees of freedom are fewer than its total degrees of freedom, so that it cannot move arbitrarily in configuration space (e.g., it cannot slide sideways) and its motion is subject to differential constraints.

## nonlinear

Said of a system whose transition model cannot be described as a matrix multiplication of the state vector; more generally, a function whose output is not a linear (weighted-sum-plus-constant) function of its inputs. The extended Kalman filter handles such systems by modeling them as locally linear.

## nonlinear constraints

Constraints that cannot be expressed in linear form. Solving general nonlinear constraints over integer variables is undecidable, as no algorithm exists for the problem.

## nonlinear regression

Regression in which the hypothesis relating inputs to the continuous output is a nonlinear function (for example, a neural network) rather than a linear function of the inputs, typically fit by minimizing a loss such as squared error.

## nonmonotonicity

The property, exhibited by default reasoning, in which the set of beliefs does not grow monotonically as new evidence arrives: a conclusion reached by default (e.g., that a parked car has four wheels) can later be retracted when new evidence contradicts it. Nonmonotonic logics such as circumscription and default logic use modified notions of truth and entailment to capture such reasoning.

## nonparametric

A representation that, rather than summarizing data with a fixed set of parameters, defines a distribution or hypothesis implicitly using a collection of the data instances themselves. In contrast to a parametric model, its complexity can grow with the number of training examples, which are retained to make predictions.

## nonparametric density estimation

Density estimation—learning a probability model from data assumed to be generated by it—without fixing the model to a bounded set of parameters, for example using kernel or k-nearest-neighbor methods whose form adapts to the observed samples.

## nonparametric model

A learning model that cannot be characterized by a bounded set of parameters, in contrast to a parametric model that summarizes the data with a fixed-size parameter set. It retains the training examples (as in k-nearest-neighbors), so its representational capacity grows with the amount of data, which is useful when data are plentiful.

## nonserial dynamic programming

A dynamic-programming technique that is essentially identical to the variable elimination method used in probabilistic inference.

## normalization

Multiplying the entries of a probability distribution by a normalization constant so that they sum to 1, often used as a shortcut (e.g., in applying Bayes' rule) to avoid computing a denominator directly.

## normalized form

A standardized syntactic form to which sentences are converted to simplify processing, such as conjunctive normal form for resolution or a canonical form for probabilistic or grammatical representations.

## Normalized utilities

Utilities placed on a scale where the worst possible catastrophe is assigned a utility of 0 and the best possible outcome a utility of 1.

## normative theory

A theory that describes how a rational agent should act (decision theory is a normative theory), as opposed to a descriptive theory that describes how actual agents such as humans really do act—which experiments suggest can be "predictably irrational."

## NP-complete

The class of the hardest problems in NP: decision problems in NP to which every other NP problem can be reduced in polynomial time, so that a polynomial-time algorithm for any one would solve all of NP efficiently.

## NP-completeness

The theory, pioneered by Cook (1971) and Karp (1972), that provides a basis for analyzing the tractability of problems: any problem class to which the NP-complete problems can be reduced is likely to be intractable. Showing a problem is NP-complete is thus strong evidence that no efficient algorithm exists for it in the worst case.

## null hypothesis

In a statistical significance test, the assumption that there is no underlying pattern in the data. The test measures how far the data deviate from a perfect absence of pattern; if that deviation is statistically unlikely (typically 5% or less) under the null hypothesis, it is taken as evidence of a real pattern. In decision-tree pruning, the null hypothesis is that an attribute is irrelevant.

## null move

A search-pruning heuristic used in game-playing programs that obtains a good lower bound on a position's value via a shallow search in which the opponent is allowed to move twice at the start.

# O

## object model

In the model-based approach to vision, a model of what an object is like—either a precise geometric model (as from a CAD system) or a vaguer statement of general object properties—used together with a rendering model to interpret images.

## object-level state space

The ordinary problem-solving state space (such as the map of Romania) that an agent searches over, as distinguished from the metalevel state space that captures the internal computational state of the search program itself.

## objective function

The function whose value a local search or optimization procedure tries to find the best state for; in the state-space landscape metaphor, elevation corresponds to the objective function, and the goal is to find its highest peak (global maximum) or lowest valley.

## objectivist

The view that probabilities are real aspects of the universe—propensities of objects to behave in certain ways (e.g., a fair coin's 0.5 propensity for heads)—rather than mere descriptions of an observer's degree of belief; contrasted with the subjectivist view.

## observation sentences

In logical positivism, the sentences corresponding to sensory inputs, to which all knowledge is ultimately connected via logical theories.

## observed

In a learning model, the variables (such as the input x and output y) whose values are specified exactly by the training data, in contrast to hidden intermediate computations.

## occupancy grid

A representation used for probabilistic robotic mapping that specifies, for each location in the grid, the probability that it is occupied by an obstacle.

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

## offline search

Search in which an agent computes a complete solution before ever acting in the real world, as opposed to online search, which interleaves computation and action.

## offset strategy

A defense approach, long used by the U.S. Department of Defense, in which superior technology is relied upon to provide a military advantage.

## omniscience

Complete knowledge of the actual outcome of every action; a distinction is drawn between rationality, which requires making the best decision given available information, and omniscience, which no real agent can achieve.

## on-policy

Describing a reinforcement learning method that evaluates and improves the same policy that is being used to select actions and generate experience, as in SARSA.

## one-armed bandit

A decision problem in which a single arm can be pulled, where the arm is a Markov reward process (an MDP with only one available action).

## one-hot vector

An encoding of an integer code i as a vector that is all zeros except for a single 1 at position i.

## online gradient descent

Another name for stochastic gradient descent (SGD), which updates parameters using a minibatch of examples.

## online learning

A learning setting in which an agent receives an input, must predict the corresponding output, and only afterward is told the correct answer, so the timing of predictions matters.

## online replanning

Interleaving execution with planning so that the agent revises its plan during execution in response to new observations or unexpected outcomes, rather than committing to a complete plan in advance.

## online search

Search in which an agent interleaves computation and action—first taking an action, then observing the environment and computing the next action—rather than computing a complete solution offline before acting; useful in dynamic or nondeterministic and unknown environments.

## ontological commitment

What a representation language assumes exists in the world—for example, propositional logic commits only to facts, while first-order logic additionally commits to objects and relations among them.

## ontological engineering

The activity of representing general, abstract concepts—such as Events, Time, Physical Objects, and Beliefs—that recur across many different domains, leaving placeholders where domain-specific knowledge can later be filled in.

## ontology

The vocabulary of predicates, functions, and constants chosen to represent a domain; the term also means a particular theory of the nature of being or existence, and the ontology determines what kinds of things exist for the representation.

## open list

An alternative name some authors use for the frontier in graph search—the set of nodes that have been generated but not yet expanded.

## open-code

To implement a construct by inserting its code inline at each point of use (for example, expanding a routine in place) rather than invoking it through a general mechanism, often to gain efficiency.

## open-loop

Describing execution that ignores percepts while carrying out a fixed sequence of actions; the loop between agent and environment is broken. In a fully observable, deterministic, known environment the agent can safely close its eyes, whereas uncertainty calls for a closed-loop approach.

## operationality

The property of a learned or derived concept description being expressed in terms that the agent can efficiently evaluate and use directly during problem solving, a key criterion in explanation-based learning.

## operations research

A field that applies mathematical modeling and optimization to decision-making problems such as scheduling, routing, and resource allocation, sharing many techniques with AI.

## opportunity cost

The amount of utility given up per time step by devoting that step to one option rather than to a better alternative.

## optical flow

The apparent motion of brightness patterns in an image sequence, arising when an object moves or when the camera moves relative to an object.

## optimal brain damage

A neural network pruning technique that removes weights judged to have the least impact on the training error, using second-derivative (curvature) information, to produce a smaller network that generalizes better.

## optimal controllers

Controllers derived by optimizing a cost functional over the trajectories of a dynamical system; in the linear-quadratic case these yield closed-form optimal control laws.

## optimal policy

A policy that yields the highest expected utility of all policies.

## optimal solution

A solution that has the lowest path cost among all solutions to a problem.

## optimally efficient

Describing a search algorithm that expands no more nodes than necessary—no other algorithm using the same heuristic information is guaranteed to expand fewer—as A* is among algorithms using a given admissible heuristic.

## optimism under uncertainty

The tendency, built into agents such as LRTA*, to assume unexplored options are promising, which encourages the agent to explore new and possibly rewarding paths.

## optimistic description

A description of an action's outcomes that lists all the effects the action might possibly have, providing an upper bound on what can be achieved (a standard-AIMA definition; term not located in the source text).

## optimization

The problem of finding an input that maximizes or minimizes an objective function, possibly subject to constraints, using methods such as gradient descent, linear programming, or local search.

## optimizer's curse

The tendency for the estimated expected utility of the choice that appears best to be systematically too high, because choosing the maximum of noisy utility estimates favors options whose errors happen to be optimistic; it can be avoided with a Bayesian approach that models the estimation errors.

## optogenetics

An experimental neuroscience technique that uses light to control neurons that have been genetically modified to be light-sensitive, allowing precise activation or silencing of specific cells.

## OR-parallelism

A form of parallelism in logic programming in which the different clauses that could satisfy a goal (the alternative ways to prove it) are explored simultaneously.

## order statistic

A particular ranked element of a sample; computing the distribution of such a ranked element generalizes calculations like taking the maximum.

## ordering constraints

In planning, constraints on a set of actions specifying that one action must occur before another, without committing to a fully sequenced (total) order; in the "plan first, schedule later" approach these are selected in the planning phase before temporal and resource information is added.

## ordinal utility function

A function that captures only the ordering of preferences (also called a value function), without committing to meaningful numerical magnitudes.

## orientation

The direction a robot or object is facing, part of its pose along with position; in configuration space it is represented by angular degrees of freedom.

## origin function

In probabilistic models with objects of uncertain provenance, a function that maps an observed percept or track to the object in the world from which it originated, used in data association.

## Othello

A two-player, deterministic, fully observable board game (also known as Reversi) that has served as a test bed for game-playing programs, in which computer players surpassed the best humans.

## out of vocabulary

Describing a word that never appeared in the training corpus of a language model; such unknown words are typically handled by mapping rare and unseen words to a special symbol, traditionally <UNK>, whose n-gram probabilities are then used.

## out-of-bag error

An error estimate for an ensemble such as a random forest, computed as the mean error on each example using only the trees whose training set did not include that example, serving as an alternative to cross-validation.

## outcome

A specific resulting state or event that may follow from an action or a random process; in decision theory, agents assign utilities to outcomes and choose actions to maximize expected utility.

## overall intensity

In computer vision, the intensity of the ambient light in a scene—one of the three factors (along with whether a surface point faces the light or is in shadow, and how much light it reflects) that together determine how much light reaches the image from a point on an object.

## overfitting

The phenomenon in which a model fits the training data too closely—including its noise and idiosyncrasies—so that it captures spurious patterns and generalizes poorly to new data.

# P

## PAC learning

Probably approximately correct (PAC) learning: a framework in computational learning theory in which any algorithm that returns hypotheses that are probably approximately correct is a PAC learning algorithm. PAC theorems bound the number of examples needed to guarantee, with high probability, a hypothesis with low error, relying on the stationarity assumption that training and future examples come from the same distribution.

## pace of change

The rate at which technology-driven automation transforms the economy; even when automation has a large net-positive impact, rapid transition can create serious problems because workers and institutions cannot adapt quickly enough.

## PageRank

An algorithm (due to Brin and Page) used in web search that ranks the importance of pages by taking into account the link structure between pages rather than page content alone.

## parameter independence

An assumption used in Bayesian parameter learning that the prior over a model's multiple parameters factors into independent distributions over each parameter, so the parameters can be treated separately.

## parameter learning

The task of finding the numerical parameters (such as the probabilities in a probability model) for a model whose structure is fixed. It can be done by maximum-likelihood estimation or by Bayesian methods.

## parametric model

A learning model that summarizes the training data with a set of parameters of fixed size, independent of the number of training examples; once the parameters are fit, the training data can be discarded. Contrast with a nonparametric model, which cannot be characterized by a bounded set of parameters.

## parent node

In a search tree, the node from which a given node was generated. Expanding a node produces child nodes for each applicable action, and each such child records the expanded node as its parent.

## Pareto dominated

An outcome is Pareto dominated by another outcome if that other outcome makes at least one player better off without making any player worse off; equivalently, an outcome fails to be Pareto optimal precisely when it is Pareto dominated by some other outcome.

## Pareto optimal

An outcome is Pareto optimal if there is no other outcome that all players would prefer; that is, you cannot make any player better off without making another worse off. Named after economist Vilfredo Pareto.

## parse tree

A tree representing the phrase structure of a string of words according to a grammar, with the sentence symbol at the root and the words at the leaves; interior nodes correspond to grammatical constituents.

## parsing

The process of analyzing a string of words to uncover its phrase structure according to the rules of a grammar; it can be viewed as a search for a valid parse tree that covers the words.

## partial assignment

In a constraint satisfaction problem, an assignment that gives values to only some of the variables, leaving others unassigned. A partial assignment that is consistent is a partial solution.

## partial information

Incomplete knowledge about the state of the world, such as knowing that a fact holds in one of several places without knowing which. Simple factored data structures cannot express it, whereas propositional logic can, using disjunction and negation.

## partial program

In hierarchical reinforcement learning, a program that outlines the overall structure or scaffolding of the agent's behavior while leaving some choices of action open to be filled in by learning; it can be arbitrarily complicated as long as it terminates.

## partial solution

In a constraint satisfaction problem, a partial assignment (one that leaves some variables unassigned) that is nonetheless consistent, i.e., violates no constraints.

## partial-order planning

A planning approach that represents a plan as a graph of actions with ordering constraints between them, rather than as a strictly linear sequence, leaving some steps unordered relative to one another.

## partially observable

A property of a task environment in which the agent's sensors do not give access to the complete state of the environment at each point in time (for example, due to noisy, inaccurate, or limited sensors, or missing data). Contrast with a fully observable environment.

## particle filtering

A family of approximate filtering algorithms that represent a belief state by a population of sample states ("particles"); the particles are propagated through the transition model, weighted by the evidence likelihood, and resampled at each step. It is consistent and often efficient for approximating the true posterior.

## particle MCMC

A family of algorithms that combines Markov chain Monte Carlo over the unrolled temporal model with particle filtering to generate the MCMC proposals; it provably converges to the correct posterior even with both static and dynamic variables, but is an offline method.

## partition

A division of a set into disjoint, exhaustive subsets, so that every element belongs to exactly one subset.

## passive learning

A reinforcement learning setting in which the agent has a fixed policy and its task is to learn the utility (expected total discounted reward) of states under that policy. This resembles the policy-evaluation step of policy iteration, but done without a model of the environment. Contrast with an active learning agent.

## passive learning agent

A reinforcement learning agent that follows a fixed policy and, by executing trials in the environment, learns the value (expected utility) of states under that policy, without trying to improve or change the policy.

## path

A sequence of actions leading from one state to another; a solution to a search problem is a path from the initial state to a goal state, and the cost of a path is the sum of its individual action costs.

## path planning

In robotics, the problem of finding a route for the robot (once it has a map and can localize itself) from its current location to a goal; one of the stages in the map-building-and-planning approach to navigation.

## paths

(Plural of path.) Sequences of actions connecting states in a search problem; a cost-optimal solution is the path with the lowest total action cost among all solution paths.

## pattern matching

The operation, central to forward chaining and rule application, of matching a rule's premises against a set of known facts to find satisfying variable bindings. It is closely connected to constraint satisfaction and is NP-hard in general.

## payoff function

A function that gives the utility (payoff) to each player for each combination of the players' actions. For a two-player game it can be represented as a matrix.

## PD controller

A proportional-derivative controller: it extends a proportional controller by adding a term proportional to the derivative (rate of change) of the error, which improves stability by damping oscillations and can achieve stability where a pure proportional controller cannot.

## PDDL

The Planning Domain Definition Language, a factored representation language (with Lisp-like syntax) for expressing planning problems. It lets many actions be described with a single action schema; basic PDDL handles classical planning, and extensions cover continuous, partially observable, concurrent, and multi-agent domains.

## Peano axioms

A small set of axioms (using the constant 0, the successor function S, and the predicate NatNum) that recursively define the natural numbers and addition. They include the principle of induction, which is a second-order-logic sentence.

## PEAS

An acronym for Performance, Environment, Actuators, Sensors, used to specify a task environment. The PEAS description groups these four elements and is the first step in designing an agent.

## pedigree analysis

The analysis of genetic inheritance in family trees, which is a special case of reasoning with Bayesian networks.

## peeking

An error in evaluating a learning algorithm in which information from the test set improperly influences the choice of hypothesis or hyperparameters, producing over-optimistic performance estimates. It can happen even when no single hypothesis directly sees the test data, if the test set is reused across many trials.

## percept

The content that an agent's sensors are perceiving at a given instant; the perceptual input the agent receives about its environment.

## percept schema

An extension to PDDL that models the agent's sensors for planning purposes, specifying under what preconditions the agent will perceive the value of a given fluent. A fully observable environment has a percept schema with no preconditions for each fluent; a sensorless agent has no percept schemas at all.

## percept sequence

The complete history of everything an agent has ever perceived. An agent's action choice at any instant can depend on its entire percept sequence to date, and the agent function maps every possible percept sequence to an action.

## perception

The process by which an agent interprets the input from its sensors to form an understanding of the state of the world (for a robot, converting sensor data into a usable model of the environment).

## perception layer

The input layer of a neural network: the first layer, to which the raw percepts (for example, image pixels) are fed, followed by hidden layers and an output layer. (Not a distinct margin term in AIMA; defined per its layer terminology.)

## perceptron

A linear classifier consisting of a one-layer neural network (a single threshold unit) that computes a weighted sum of its inputs and applies a hard threshold. Trained by the perceptron learning rule, it can fit any linearly separable data (with guaranteed convergence) but only linearly separable functions.

## perceptron learning rule

The learning rule that adjusts the weights of a linear threshold unit (which outputs 1 when the weighted sum of inputs meets a threshold and 0 otherwise) to fit training data.

## perceptron network

A network built from perceptron units. In its classic single-layer form each output is a threshold unit computing a weighted sum of the same inputs; the outputs are computed independently.

## perfect information

A property of a game in which the state is fully observable to all players at all times; "perfect information" is essentially a synonym for "fully observable" (e.g., chess), as opposed to games with hidden private information.

## perfect rationality

Always taking the exactly optimal action. AIMA notes perfect rationality is not feasible in complex environments because the computational demands are too high, motivating notions of limited or bounded rationality, though it remains a useful theoretical starting point.

## performance element

In a learning agent, the component responsible for selecting external actions; it corresponds to what would otherwise be considered the entire (non-learning) agent. The learning element works to improve it based on feedback from the critic.

## performance measure

A criterion that evaluates any given sequence of environment states to determine how desirable the agent's behavior is. Reflecting a consequentialist view, it is used to judge whether an agent has acted well and is typically defined by the designer or user rather than the agent itself.

## perpetual punishment

In repeated games, a strategy in which a player, using knowledge of all players' past choices, responds to a defection by punishing the offending player forever afterward.

## perplexity

A measure used to compare language models, defined as 2 raised to the entropy of the probability distribution. A model with lower perplexity is, all else equal, a better model, though the book cautions that measuring performance on a real task is usually more informative.

## persistence arc

In a dynamic Bayesian network, an arc linking a variable at one time step to the same variable at the next. Its conditional probability table can assign a small probability of transitioning to failure each step while specifying that once a component fails it stays failed, modeling the persistence of a condition.

## persistent failure model

A sensor model describing how a sensor behaves both normally and after it has failed, capturing failures that persist over time rather than transient blips. It augments the state with a status variable connected across time by a persistence arc, letting the system infer that a sensor is broken rather than trusting anomalous readings.

## perspective projection

The image-formation process in which a 3D scene point is projected onto the image plane through a pinhole/focal point, with coordinates scaled by focal length divided by depth. Consequences include farther objects appearing smaller and parallel lines converging to a vanishing point.

## pessimistic description

In hierarchical planning, an approximate description of the reachable set of a high-level action that may understate that set (guaranteeing only states known to be reachable), complementing the optimistic description that may overstate it.

## phone model

A probabilistic model (typically a hidden Markov model) of the acoustic realization of a single speech sound (phone), used within the acoustic model to relate audio waveforms to phonemes. (Not a distinct margin term in AIMA; defined per its speech-recognition usage.)

## phoneme

A basic unit of speech sound in a language. In AIMA's account of speech recognition the problem breaks down compositionally from waveforms to phonemes to words to sentences, so phonemes are the sound-level building blocks between the raw waveform and words.

## phrase structure

The hierarchical syntactic organization of a sentence into nested constituents (such as noun phrases and verb phrases). Syntactic categories constrain likely words and provide a framework for meaning; phrase-structure grammars (notably PCFGs) model this structure.

## Physical games

Games that involve real-world physical interaction, such as robotic soccer, billiards, and ping-pong, which have attracted attention as AI challenges.

## physical symbol system

From Newell and Simon's physical symbol system hypothesis, which states that a physical symbol system has the necessary and sufficient means for general intelligent action; that is, any system exhibiting intelligence must operate by manipulating data structures composed of symbols.

## physicalism

The view (also called naturalism, and here equivalent to materialism) that the brain's operation according to the laws of physics constitutes the mind, treating mental phenomena as physical; it stands in contrast to dualism.

## piano movers

The class of deterministic, fully observable robot motion-planning problems, named after the problem of moving a piano through a space among obstacles. Robot motion planning of this kind was shown to be PSPACE-hard.

## pictorial structure model

A model that represents an object as a collection of parts connected by spring-like deformable relationships, so recognition scores both the appearance of individual parts and their relative spatial configuration. (Not a distinct margin term in AIMA; defined per its computer-vision usage.)

## PID controller

A proportional-integral-derivative controller: it combines a proportional term (in proportion to the current error), an integral term (correcting for accumulated error over time), and a derivative term (reacting to how fast the error is changing). The integral term eliminates systematic long-term error left by a PD controller.

## pinhole camera

The simplest image-forming device, consisting of a small pinhole opening at the front of a box and an image plane at the back, onto which stationary objects are projected to form a focused image.

## pipeline architecture

A robot software architecture in which data flows through a sequence of stages: entering at the sensor interface layer, passing through the perception layer that updates internal models, then the planning and control layer, and finally out through the vehicle interface layer.

## plan monitoring

An execution-monitoring strategy in which, before executing each action, the agent verifies that the entire remaining plan will still succeed (by checking the preconditions for the rest of the plan). This lets the agent detect failures early and also exploit serendipity by abandoning the rest of a plan when the goal is already achieved.

## plan recognition

An approach to multi-agent coordination in which one agent observes another's action (or short sequence of actions) and infers the joint plan being followed, then acts accordingly. It works when an early action determines the joint plan unambiguously.

## planetary rover

A mobile robot designed to explore the surface of a planet, a representative application domain for autonomous robotics.

## planning and control layer

In a robot's pipeline architecture, the stage that takes the perception layer's models of the environment, adjusts the robot's internal plans, and turns them into the actual control commands for the robot.

## planning graph

A specialized data structure (introduced by Graphplan) that alternates levels of literals and levels of actions and encodes constraints on preconditions, effects, and mutual exclusions. It provides useful information and admissible heuristic estimates for planning problems.

## PlanSAT

The decision problem of whether there exists any plan that solves a given planning problem. Bounded PlanSAT asks whether a solution of length k or less exists. Both are decidable and PSPACE for propositionalized classical planning, but PlanSAT becomes only semidecidable if function symbols make the state space infinite.

## playout

In Monte Carlo tree search, a simulation (rollout) of a complete game from a given state, choosing moves alternately for each player until a terminal position is reached. The value of a state is estimated as the average utility over many such playouts, with move choices biased by a playout policy.

## playout policy

In Monte Carlo tree search, a policy that guides the moves made during a simulated playout (a game played out to a terminal position) so that the moves are biased toward good ones rather than chosen purely at random.

## ply

One move by a single player, corresponding to one level deeper in the game tree. The term disambiguates "move" in games; a two-ply game tree involves one action by MAX and one by MIN.

## pneumatic actuation

Actuation produced by a pneumatic actuator, which uses compressed air to generate mechanical motion, contrasted with electric actuators driven by motors and hydraulic actuators driven by pressurized fluid.

## point-to-point motion

A motion task in which the goal is to move the robot (or its end effector) from a specified start configuration to a specified goal configuration, without constraining the detailed path between them. (Not a distinct margin term in AIMA; defined per its robotics usage.)

## pointwise product

An element-by-element multiplication operation (distinct from ordinary matrix multiplication) used, for example, when combining factors in probabilistic inference such as variable elimination.

## poker

A card game of imperfect (partial, stochastic) information in which players hold private cards and bluffing is a core strategy. AIMA uses it as an example of imperfect-information games; the program Libratus reached superhuman play in no-limit Texas hold 'em using abstraction to reduce the enormous state space.

## policy

A solution to a Markov decision process: a function, denoted pi, that specifies an action pi(s) for every state s the agent might reach. Executing a policy makes the agent behave like a simple reflex agent; an optimal policy (pi*) yields the highest expected utility.

## policy evaluation

One of the two steps of policy iteration: given a fixed policy, compute the utility of each state that would result from executing that policy. It can be done exactly by solving linear equations for small state spaces, or approximately via simplified value-update iterations.

## policy gradient

A policy-search method that adjusts a parameterized policy by following the gradient of the policy value (the expected reward-to-go) with respect to its parameters. When the value is not available in closed form it is estimated empirically by executing the policy, converging to a local optimum in policy space.

## policy improvement

A step in policy iteration that computes a new maximum-expected-utility policy by one-step look-ahead based on the utilities of the current policy. It is alternated with policy evaluation until it yields no change, at which point the policy is optimal.

## policy iteration

An algorithm for finding an optimal policy that, starting from an initial policy, alternates policy evaluation (computing the utility of each state under the current policy) and policy improvement (computing a new MEU policy from those utilities). Because each iteration yields a strictly better policy over a finite policy space, it terminates at an optimal policy.

## policy loss

The most utility an agent can lose by executing a given policy (typically one derived from imperfect utility estimates) instead of the optimal policy; it is bounded by the error in the utility estimates. In practice a policy often becomes optimal well before its utility estimates fully converge.

## policy search

A reinforcement learning approach in which the agent directly learns a policy mapping states to actions, adjusting the parameters of a parameterized policy representation as long as performance keeps improving. Even when the policy is represented via Q-functions, this differs from Q-learning.

## policy value

Denoted rho(theta), the expected reward-to-go obtained when a parameterized policy pi-theta is executed; policy search improves the policy by following the gradient of this quantity (or an empirical estimate of it) in policy space.

## polite convention

Turing's suggested convention, in the context of the Turing Test, that rather than endlessly debating whether machines can think, one simply assumes that everyone (and by extension a suitably behaving machine) thinks.

## polynomial kernel

A kernel function of the form K(x_j, x_k) = (1 + x_j . x_k)^d, corresponding to a feature space whose dimension is exponential in d, which allows linear separators to be found efficiently in that high-dimensional space (the kernel trick).

## POMCP

Partially Observable Monte Carlo Planning: an online planning algorithm for POMDPs that combines particle filtering with UCT-based Monte Carlo tree search.

## pooling

A neural network operation, common in convolutional networks, that summarizes a group of values into a single value to add invariance to small translations; for example, max-pooling reports the maximum value within each group.

## portfolio

A system that maintains a collection of different algorithms which can each be applied to a given problem, allowing the best-suited one to be selected or run.

## pose

The position and orientation of an object (or the camera or robot) in space.

## position

In the terminology of games, a synonym for "state" (just as "move" is used as a synonym for "action").

## positive

In classification, a positive example is an input whose target output is true, as opposed to a negative example whose output is false; the resulting task is Boolean classification.

## possibility axiom

In AIMA the basic axioms of probability (Kolmogorov's axioms) state that every possible world has a probability between 0 and 1 and that the probabilities of the set of all possible worlds sum to 1. (Not located as a distinct margin term.)

## possibility theory

A formalism introduced by Zadeh (1978) to handle uncertainty in fuzzy systems, sharing much in common with probability theory, used for reasoning about vague or imprecise information.

## possible world

An environment or situation with respect to which the truth of each sentence is evaluated; in standard logics every sentence is either true or false in each possible world. When precision is needed, AIMA uses the term model, a mathematical abstraction with a fixed truth value for every relevant sentence.

## post-decision disappointment

A name (Harrison and March, 1984) for the same phenomenon as the optimizer's curse: the tendency for the actual value of a chosen "optimal" action to fall short of its estimated value, because selecting the highest-estimated option systematically favors options whose values were overestimated. Related to the winner's curse.

## potential field

A construct used in robot motion planning that defines a field over the configuration space (typically attracting the robot toward the goal and repelling it from obstacles) to help solve motion-planning problems such as those tackled by hybrid A*.

## pragmatics

The aspect of language interpretation that adds context-dependent information about the current situation to complete a sentence's meaning, including resolving indexicals (phrases like "I" or "today") and inferring the speaker's intent.

## precedence constraints

Constraints specifying that one task must occur before another; when task A must precede task B and A takes duration d, this is expressed as an arithmetic constraint such as start_A + d <= start_B, as used in scheduling.

## precision

In object-detection evaluation, the measure of not reporting objects that are not actually present, balanced against recall (finding all the objects that are there). It is given as True Positives / (True Positives + False Positives).

## precomputation

The technique of computing certain values, such as optimal path costs, in advance so they can be reused during later search, improving the efficiency of heuristic search.

## precondition

A condition that must hold for an action to be executable. Precondition axioms are logical sentences stating that an action occurrence requires its preconditions to be satisfied, preventing plans that contain illegal actions.

## precondition axioms

Logical axioms added to a planning encoding that state that an action occurrence is legal only if the action's preconditions are satisfied, preventing the generation of plans containing illegal actions.

## predicate indexing

A simple scheme for organizing a knowledge base so that facts are stored in buckets according to their predicate symbol (e.g., all Knows facts in one bucket, all Brother facts in another), speeding retrieval.

## prediction

In temporal probabilistic reasoning, the task of computing the posterior distribution over a future state given all evidence to date, P(X_{t+k} | e_{1:t}) for k > 0; useful for evaluating possible courses of action by their expected outcomes.

## predictive learning

LeCun's term for an unsupervised learning system that builds a model of the world and learns to predict aspects of future world states, rather than merely predicting labels for i.i.d. inputs.

## preference constraints

Constraints in a CSP that indicate which solutions are preferred rather than which are strictly allowed; a problem including such constraints is a constrained optimization problem (COP).

## preference elicitation

The process of working out a human's utility function by presenting choices and using the observed preferences to pin down the underlying utility function.

## preference independence

A regularity in deterministic preference structures: two attributes are preferentially independent of a third if the preference between outcomes differing only in the first two does not depend on the value of the third. This allows a value function to be represented concisely.

## preferred action

In planning, an action recommended by first defining a relaxed version of the problem and solving it to obtain a relaxed plan; actions appearing in that relaxed plan are treated as preferred during search.

## prefix

A notation in which the operator is written before its arguments (for example, +(n, S(n))), as used for the standard syntax of first-order logic, in contrast to the infix notation of ordinary mathematics.

## premise

The antecedent of an implication; in a sentence of the form P => Q, P is the premise (antecedent) and Q is the conclusion (consequent).

## presentation

In learning contexts, a presentation of the examples refers to one pass of showing the training examples to a learning algorithm (as in perceptron or gradient training). (Not located as a distinct margin term.)

## principle of indifference

The rule of assigning equal probabilities to cases when there is no knowledge that would lead us to consider one more probable than another. Attributed to Laplace but named by Keynes (1921); also known as the principle of insufficient reason.

## principle of insufficient reason

Another name for the principle of indifference (used by Boole and Venn): assign equal probabilities to alternatives when there is no reason to consider any one more probable than another.

## principle of trichromacy

The principle (Thomas Young, 1802) that a human observer can match the appearance of any spectral energy density by mixing appropriate amounts of just three primary colored lights, so color images can be represented with three numbers per pixel (for example, RGB).

## prioritized circumscription

A variant of circumscription that lets one assert precedence among the predicates being minimized (e.g., religious beliefs over political beliefs), thereby preferring models in which higher-priority abnormality predicates are minimized first.

## prioritized sweeping

A heuristic for speeding convergence of value/policy iteration in approximate dynamic programming that prefers to update states whose likely successors have just undergone a large adjustment, using a priority queue to focus computation on the most significant changes.

## priority queue

A queue that first pops the element with the best cost according to a given evaluation function; used, for example, to order nodes in best-first search.

## prisoner's dilemma

The most famous game in game theory: two separately interrogated suspects each choose to testify or refuse, with a payoff matrix in which testifying (defecting) is the dominant strategy for both, yet mutual refusal (cooperation) would leave both better off, illustrating the conflict between individually rational choices and collectively optimal outcomes.

## probabilistic checkmate

A concept in partially observable games (such as Kriegspiel): a checkmate that works in every board state of the belief state and is guaranteed with probability 1 through randomization of the winning player's moves, since the opponent cannot keep guessing correct evasive moves indefinitely.

## probabilistic context-free grammar

A grammar (PCFG) in which each rewrite rule is assigned a probability, defining a probability distribution over the strings the grammar can generate; a popular hierarchical language model.

## probabilistic Horn abduction

A framework combining logic-programming-style Horn clauses with probabilities, used for abductive reasoning under uncertainty and closely related to Bayesian networks (Poole, 1993).

## probabilistic inference

The computation of posterior probabilities for query propositions given observed evidence. In its simplest form it uses the full joint distribution as the knowledge base from which answers to all questions can be derived.

## probabilistic language model

A probability distribution over strings that assigns a likelihood to any sequence of words, so that more plausible sentences receive higher probability than implausible ones.

## probabilistic principal components analysis

A simple generative model (pPCA) in which a latent variable is drawn from a standard Gaussian and the observed data is generated as a linear transformation of that latent variable plus isotropic Gaussian noise.

## probabilistic roadmap

A motion-planning algorithm (PRM) that uses a fast, incomplete simple planner and a collision checker to connect sampled configurations into a graph (roadmap), which is then searched to find a path between the start and goal configurations.

## probability

A theory that fills the gap left by ignorance of exact facts, allowing rigorous reasoning with uncertain information; it assigns degrees of belief between 0 and 1 that support the construction of models of rational thought.

## probability density function

A parameterized function of a value x giving the probability that a continuous random variable takes on that value. A density expresses probability per unit width; the density at a point is the probability of falling in an arbitrarily small region beginning at that point divided by the region's width.

## probability distribution

An assignment of a probability to each possible value of a random variable. For a finite discrete range it can be written as a vector of numbers (a categorical distribution), and the notation extends to conditional distributions.

## probability model

A model that associates a numerical probability P(omega) with each possible world in the sample space, where every world's probability lies between 0 and 1 and all probabilities sum to 1. A fully specified probability model determines the probability of every possible world.

## probability theory

The mathematical framework for representing and reasoning about uncertainty, in which quantities of interest are modeled as random variables that take on values with associated probabilities.

## probably approximately correct

A criterion (PAC) requiring that a learned hypothesis be, with high probability, only slightly wrong; an algorithm that returns such hypotheses is a PAC learning algorithm, providing performance bounds for learning.

## probit distribution

The probit ("probability unit") model, based on the integral of the standard normal distribution, which acts as a soft threshold function. It is justified by supposing an underlying hard decision threshold whose location is subject to random Gaussian noise; it is an alternative to the logit model.

## probit model

A model in which a discrete outcome's probability is obtained by thresholding an underlying multivariate Gaussian variable ("probit" is short for "probability unit"), giving an S-shaped response curve; an alternative to the logit model.

## problem

A search problem defined formally by its components: a set of possible states (the state space), an initial state, goal states, the actions available, a transition (result) model, and an action cost function.

## problem formulation

The phase in which an agent devises a description of the states and actions needed to reach a goal, an abstract model of the relevant part of the world. Every problem formulation necessarily involves abstraction, leaving out details irrelevant to reaching the goal.

## problem generator

The component of a learning agent responsible for suggesting exploratory actions that will lead to new and informative experiences, even if those actions may be suboptimal in the short run, so the agent can discover better behavior over time.

## problem-solving agent

An agent that plans ahead by considering a sequence of actions forming a path to a goal state, using search. Problem-solving agents use atomic state representations, treating states of the world as wholes with no internally visible structure.

## procedural attachment

A technique used in semantic-network and similar systems whereby a query about or assertion of a certain relation triggers a call to a special-purpose procedure written for that relation, rather than invoking a general inference algorithm.

## process

A category of temporal entity (a "liquid" event), such as raining, whose subintervals are also occurrences of the same process, in contrast to discrete events that have definite start and end points.

## product rule

An alternative form of the definition of conditional probability, stating that P(a ^ b) = P(a | b) P(b): for a and b both to hold, b must be true and a must be true given b.

## progression planning

Forward search for a plan through the space of ground states, starting from the initial state and applying the problem's actions to search forward until a goal state is reached (as opposed to backward or regression search).

## Prolog

The most widely used logic programming language, employed mainly for rapid prototyping and symbol-manipulation tasks such as writing compilers and parsing natural language. Prolog programs are sets of definite clauses, computed by backward-chaining resolution.

## pronunciation model

A component of a speech-processing system that specifies how words map to sequences of sounds (phonemes), bridging the acoustic model and the word/language models in the pipeline from waveforms to sentences. (Not located as a distinct margin term.)

## proof

A chain of conclusions, derived by applying inference rules, that leads to a desired goal.

## proof-checker

A program that verifies the correctness of a purported proof by confirming that each inference step follows the allowed rules, as opposed to a theorem prover that searches for a proof. (Not located as a distinct margin term.)

## proper policy

In a Markov decision process, a policy that is guaranteed to eventually reach a terminal state; with proper policies one can use additive undiscounted rewards without worrying about comparing infinite reward sequences.

## proposal distribution

In Metropolis-Hastings sampling, the distribution used to generate a candidate next state from the current state, after which the candidate is probabilistically accepted or rejected according to the acceptance probability.

## proposition symbol

An atomic sentence of propositional logic consisting of a single symbol that stands for a proposition that can be true or false. Two symbols have fixed meanings: True (always true) and False (always false).

## propositional logic

A simple but powerful logic whose atomic sentences are individual proposition symbols, combined with logical connectives to form more complex sentences.

## propositionalize

To convert a first-order inference problem into propositional form by applying instantiation rules to ground the sentences, so that propositional inference can be applied. This approach is typically slow unless the domain is small.

## protein design

An assembly-type search problem whose goal is to find a sequence of amino acids that will fold into a three-dimensional protein having desired properties.

## provably beneficial

A property of AI agents that we ultimately want: agents that can be proven to act in humans' interest. It is approached by building machines that pursue human objectives while remaining uncertain about exactly what those objectives are, giving them incentives to act cautiously, ask permission, learn preferences, and defer to human control.

## pruning

Eliminating possibilities from consideration without having to examine them, an important technique across many areas of AI; for example, cutting off branches of a game tree that cannot affect the final result (as in alpha-beta pruning).

## psychological reasoning

The development of a working psychology for artificial agents to use in reasoning about themselves and other agents, often based on folk psychology. It is currently most useful in natural language understanding, where inferring a speaker's intentions is paramount.

## PUMA

The Programmable Universal Machine for Assembly, a robot developed by Unimation in 1978 that became the de facto standard for robotic manipulation for the following two decades.

## pure Monte Carlo search

The simplest Monte Carlo game-playing method: run N simulated playouts from the current state and choose the move whose simulations yield the highest win percentage.

## pure strategy

A deterministic policy in game theory; for a single-move game a pure strategy is just a single action. It contrasts with a mixed strategy, a randomized policy that selects actions according to a probability distribution.

## pure symbol

In the DPLL satisfiability algorithm, a symbol that always appears with the same sign (always positive, or always negative) across all clauses. If a sentence has a model, it has one in which pure symbols are assigned so as to make their literals true, since doing so can never falsify a clause.

# Q

## Q-learning

A temporal-difference reinforcement learning method that avoids the need for a transition model by learning an action-utility function Q(s, a), the expected total discounted reward if the agent takes action a in state s and then acts optimally. It is an off-policy learner: a Q-learning agent does not need a model to either learn or act, in contrast to its close relative SARSA.

## QALY

Quality-adjusted life year, a measure used in medical and public-policy decisions that accounts for the fact that patients will accept a shorter life expectancy to avoid disability (for example, being indifferent between two years on dialysis and one year at full health). QALYs correspond to utilities only under strong assumptions such as risk neutrality.

## quadratic programming

An optimization problem with a quadratic objective subject to linear constraints, for which good software packages exist. In AIMA it arises when solving the dual representation for support vector machines to find the maximum-margin separator.

## qualia

The technical term (from Latin, roughly "of what kind") for the intrinsic nature of subjective experiences, such as how the color red actually looks. Whether machines can have qualia is the central question running through debates about consciousness and strong AI.

## qualification problem

The problem of confirming that all the necessary preconditions of an action hold for it to have its intended effect, given the countless unusual exceptions that could make it fail. It has no complete solution within logic; probability theory lets us summarize all the exceptions without explicitly naming them.

## qualitative physics

A subfield of AI (and knowledge representation) that investigates how to reason about physical systems using monotonic relationships among measures, without plunging into detailed equations and numerical simulations.

## qualitative probabilistic networks

Networks that propagate qualitative information about how uncertain variables influence one another, allowing a system to reason about stochastic dominance and make rational decisions without relying on any numeric probability values.

## quantification

The use of quantifiers to express properties of entire collections of objects instead of enumerating them by name. First-order logic contains two standard quantifiers, the universal (for all) and the existential (there exists), and the order in which they are nested is significant.

## quantization factor

A parameter governing quantization, the preprocessing step of forcing a continuous-valued input (such as a wait time) into a fixed set of discrete bins; domain knowledge can suggest which thresholds matter. (Note: AIMA discusses "quantization" but does not use the exact phrase "quantization factor"; this definition follows the book's usage of quantization.)

## quasi-logical form

In natural language semantics, an intermediate representation produced by the grammar in place of an actual logical sentence; algorithms outside the parsing process (for example, using preference rules for quantifier scope) then turn it into a logical sentence.

## querying

The act of asking questions of a knowledge base to retrieve what follows from what it has been told; the questions posed (e.g., with an Ask operation) are called queries or goals. In data privacy contexts it also refers to aggregate querying, where only summary statistics rather than individual records are returned.

## question answering

A task in which the query is a genuine question (e.g., "Who founded the U.S. Coast Guard?") and the response is an actual answer ("Alexander Hamilton") rather than a ranked list of documents. Since about 2001 such systems have used Web information retrieval to greatly increase their coverage.

## queue

The data structure used to store the frontier in search algorithms, supporting the operations IS-EMPTY, POP, TOP, and ADD. Three kinds are used: a priority queue (pops the minimum-cost node, for best-first search), a FIFO queue (pops the earliest-added node, for breadth-first search), and a LIFO queue or stack (pops the most recently added node, for depth-first search).

## quiescence search

An extra search applied so that the evaluation function is used only on quiescent positions, that is, positions with no pending move (such as capturing the queen) that would wildly swing the evaluation. It continues searching nonquiescent positions, often restricted to moves like captures that quickly resolve the uncertainty.

# R

## radial basis function

A function whose value depends only on the distance from a center point, often a Gaussian. Radial basis functions are used as features or kernels so that a weighted combination of them can approximate arbitrary functions.

## radiometry

The study of the measurement of light and other electromagnetic radiation, providing the physical basis for modeling how much light reaches a camera sensor in computer vision.

## random forest

An ensemble method that extends decision-tree bagging by introducing extra randomness when growing each of the M trees, which further reduces variance and improves generalization.

## random surfer model

A model of web browsing underlying the PageRank algorithm, in which a hypothetical user follows links at random and occasionally jumps to a random page. A page's rank reflects the long-run probability of the surfer being there.

## random walk

A stochastic process in which each successive state is reached by taking a randomly chosen step, used for example as a baseline exploration strategy or a model of undirected movement through a state space.

## random-restart hill climbing

A variant of hill climbing that conducts a series of hill-climbing searches from randomly generated initial states until a goal is found. It is complete with probability 1, because it will eventually generate a goal state as its starting point, and it works especially well when the state-space landscape has few local maxima and plateaus.

## randomized weighted majority algorithm

An online learning method for pooling the predictions of a panel of experts. It keeps a weight for each expert, randomly chooses an expert to trust in proportion to its weight, and after seeing the correct answer penalizes the weight of every expert that erred. Its performance is measured by regret; the number of mistakes it makes is bounded relative to the best expert in hindsight.

## range

The set of possible values that a random variable can take on; formally, a random variable is a function mapping possible worlds to this range.

## Rao-Blackwellization

A statistical technique that improves the accuracy of sampling-based inference by performing exact inference over a subset of the variables while sampling the rest, exploiting the fact that exact computation is always at least as accurate as sampling.

## Rao-Blackwellized particle filter

A particle filter that applies Rao-Blackwellization by combining particle filtering (e.g., for a robot's location) with exact filtering for the remaining variables (e.g., map building), working well for both stochastic and deterministic domains.

## rational agent

An agent that does the right thing. AIMA's formal definition: for each possible percept sequence, a rational agent should select an action that is expected to maximize its performance measure, given the evidence provided by the percept sequence and whatever built-in knowledge the agent has.

## rationalism

One of the two dimensions along which AIMA characterizes AI: defining and pursuing intelligence in terms of an abstract, formal standard of rationality ('doing the right thing') rather than fidelity to human performance. A rationalist approach combines mathematics and engineering and connects to statistics, control theory, and economics.

## rationality

An abstract, formal standard for intelligence, loosely 'doing the right thing.' It concerns selecting actions expected to maximize a performance measure given what the agent knows; it is about making good decisions, not about being infallible in an unpredictable world.

## reachable set

In the angelic semantics for high-level actions (HLAs) used in hierarchical planning, the reachable set of an HLA from a given state is the set of states reachable by any of the HLA's implementations. A high-level plan achieves the goal if its reachable set intersects the set of goal states.

## reached

Describes any state for which a search node has been generated, whether or not that node has yet been expanded.

## reactive control

A control approach in which actions are chosen as direct responses to current sensory input, without building or consulting an internal model of the world.

## reactive layer

The lowest level in a layered robot architecture, providing fast, reflex-like responses to sensor input while higher layers handle deliberative planning.

## reactive planning

An approach, associated with reflex-agent architectures, in which agents respond directly to the current situation rather than deliberating over a plan; universal plans developed for reactive planning turned out to be a rediscovery of policies from Markov decision processes.

## real-time AI

The study of methods for controlling an agent's own deliberation so that it uses the available time well and produces acceptable decisions under time pressure, ceasing to deliberate when action is demanded. As AI moves into more complex domains, essentially all problems become real-time because the agent never has long enough to solve the decision problem exactly.

## real-time dynamic programming

A method (RTDP) that carries out value-iteration-style updates only on the states actually encountered during execution, interleaving computation with action; it is closely analogous to the LRTA* real-time search algorithm.

## real-time search

Search for environments where the agent must commit to an action after only a fixed amount of computation, interleaving search and action rather than fully planning in advance (as introduced with the LRTA* algorithm).

## real-world problem

A problem, such as robot navigation, whose solutions are actually used by people and whose formulation is idiosyncratic and non-standardized because each instance has its own peculiarities (e.g., different sensors producing different data).

## realizable

A property of a learning problem: it is realizable if the hypothesis space actually contains the true function that generated the data. If the problem is not realizable (e.g., fitting a quadratic with only linear hypotheses), no amount of data will recover the true function.

## reasoning

The process by which a knowledge-based agent operates over an internal representation of knowledge to derive new representations about the world and decide what actions to take.

## recall

A performance measure paired with precision: recall captures the ability to find all the relevant items that are there (for example, all the objects present in an image), whereas precision captures the ability to avoid reporting items that are not. It is the fraction of true positives among all actual positives (true positives plus false negatives).

## reciprocal rank

An information retrieval metric equal to one divided by the rank position of the first relevant result. Averaging it over queries gives the mean reciprocal rank.

## recognition

In computer vision, one of the two core problems (alongside reconstruction): drawing distinctions among the objects an agent encounters based on visual and other information, interpreted broadly. Attaching names to objects in an image is recognition, as is answering questions such as whether a thing is asleep or eats meat.

## recombination

In evolutionary algorithms, the process by which the fittest individuals (states) in a population produce offspring (successor states) that make up the next generation, echoing natural selection.

## recombine

In genetic algorithms, the recombination step of the crossover operation: a crossover point is chosen to split each parent string, and the parts are recombined to form two children, each inheriting one segment from each parent.

## reconstruction

One of the two core problems of computer vision, in which an agent builds a model of the world from an image or set of images. It is interpreted broadly, covering building geometric models and mapping surface textures, and is the counterpart to recognition.

## record linkage

A problem arising in statistics when data records lack standard unique identifiers, so it must be determined which records refer to the same real-world entity (for example, differently spelled citations of the same author). It is an instance of identity uncertainty, commonly addressed with probabilistic models such as the Fellegi-Sunter model.

## rectangular grid

One way to discretize a continuous state space for local search: instead of allowing points anywhere in continuous space, locations are limited to fixed points on a grid with spacing delta, giving each state a finite number of successors (obtained by incrementing a variable by delta).

## rectified linear unit

An activation function widely used in modern neural networks, abbreviated ReLU, which outputs its input when positive and zero otherwise.

## recurrent network

A neural network that, unlike a feedforward network, feeds its intermediate or final outputs back into its own inputs. The signal values thus form a dynamical system with internal state or memory, making recurrent networks effective for sequence-processing tasks such as language, text, and audio.

## recursive

Describing a procedure or definition that refers to itself, as in a recursive, depth-first search algorithm or a recursively defined object. In such an algorithm each recursive call unwinds back to an earlier point when a condition is met.

## recursive best-first search

A best-first search algorithm (RBFS) that mimics standard best-first search using only linear space. It resembles a recursive depth-first search but keeps an f_limit variable tracking the f-value of the best alternative path from any ancestor; when the current node exceeds this limit, the recursion unwinds, backing up the best f-value of the forgotten subtree so it can decide whether to reexpand it later.

## reduct

In default and nonmonotonic reasoning, a simplified version of a theory obtained relative to a candidate set of assumptions, used to test whether that set forms a consistent extension.

## redundant path

A path to a state for which the search has an alternative (usually better) path; a cycle is a special case. A search that checks for redundant paths is called a graph search, while one that ignores them is a tree-like search.

## reference class

The class of similar cases with known outcome frequencies in which a frequentist places a particular experiment in order to estimate its outcome probability. The reference class problem is the difficulty of choosing the right class: too specific a class becomes empty (every real event being unique), a vexing problem in the philosophy of science.

## reference controller

In robotics, a controller that turns a motion plan into a policy which tries to follow the reference path, correcting the robot back toward the path whenever it drifts away because of imperfect dynamics or perturbations.

## reference counts

Counts kept of the number of times a state has been reached, so that a state can be removed from the reached table once there are no remaining ways to reach it.

## reference path

In robotics, the path produced by motion planning that a controller then tries to track. Because it is created as a sequence of points without regard to velocities and accelerations, it is usually retimed into a proper trajectory before tracking.

## referential transparency

A property whereby it does not matter which term a logic uses to refer to an object; only the object named by the term matters, so equal terms can be freely substituted.

## reflect

For an agent to deliberate about the computational entities and actions occurring within its own architecture, that is, to reason about and direct its own internal computations rather than only about the external world.

## reflective architecture

An architecture that enables deliberation about the computational entities and actions occurring within the architecture itself (metareasoning being one example). It can be founded on a joint state space combining the environment state and the agent's own computational state, over which decision-making and learning algorithms improve the agent's computations.

## refutation

A proof technique, also called proof by contradiction and corresponding to reductio ad absurdum, that proves a sentence by assuming it is false and showing this leads to a contradiction with the known axioms (that is, showing the negated sentence together with the axioms is unsatisfiable). Resolution is refutation-complete.

## regions

In computer vision, the groups of similar pixels that result from segmentation, which partitions an image based on attributes such as brightness, color, and texture. A finer partition yields more regions and a coarser partition fewer.

## regression

In machine learning, the (admittedly obscurely named) supervised learning problem in which the output is a number, such as tomorrow's temperature, rather than one of a finite set of values (classification). In planning, regression also names computing, for a goal and an action applied backward, the state description from which that action would achieve the goal.

## regression planning

Planning by backward (regression) search: starting from the goal and applying relevant actions backward, computing at each step the regressed state description, until a sequence of steps reaching the initial state is found. Considering only relevant actions reduces the branching factor compared with forward search.

## regression to the mean

The statistical phenomenon, named by Francis Galton in 1886, that extreme cases tend to be followed by more average ones (for example, the children of tall parents are likely to be taller than average but not as tall as their parents). AIMA notes this topic is distinct from the machine-learning task of "regression," which really concerns function approximation; the name stuck only because Galton labeled his plots "regression lines."

## regression tree

A decision-tree variant used when the output attribute is a numerical value (such as an apartment price); each leaf predicts a number rather than a class. Trees covering both classification and regression are known as CART.

## regret

How much worse a policy does compared to the best available alternative measured in hindsight. In the bandit setting, regret is defined relative to a clairvoyant policy that always picks the best arm; a UCB policy's choice determines its regret, and Lai and Robbins showed no algorithm can have regret growing more slowly than logarithmically. In online learning with experts, regret is the number of additional mistakes made compared to the expert that had the best prediction record in hindsight.

## regular expression

A pattern-matching notation for text. AIMA cites regular expressions as a simple technique for information extraction: when source text is well structured (for example, laid out in a table), regular expressions can pull out the desired fields such as street, city, and zip code.

## regularization

An alternative to cross-validation for model selection that searches directly for a hypothesis minimizing a weighted sum of empirical loss and hypothesis complexity (total cost = loss + lambda * complexity). AIMA calls this "explicitly penalizing complex hypotheses"; the name reflects looking for functions that are more regular. The complexity term is the regularization function (for polynomials, the sum of squared coefficients is a good choice).

## regularization function

The complexity measure used within regularization; combined with a chosen loss function, it explicitly penalizes complex hypotheses so the learner prefers simpler, more regular functions.

## reinforcement

The reward or punishment feedback an agent receives in reinforcement learning. For example, at the end of a chess game the agent is told it has won (a reward) or lost (a punishment); it must then decide which prior actions were most responsible and adjust to aim for more rewards in the future.

## reinforcement learning

A form of learning in which the agent learns from a series of reinforcements: rewards and punishments. Unlike supervised learning, the agent is not told which actions to take; it must discover from the reinforcement signal which of its actions were responsible for the outcome and alter its behavior to gain more reward over time.

## rejection sampling

A general method for producing samples from a hard-to-sample distribution given an easy-to-sample one; in Bayesian networks it estimates a conditional probability P(X|e) by first generating samples from the prior distribution specified by the network, then rejecting all samples inconsistent with the evidence, and finally counting the outcomes among the samples that remain. It yields a consistent estimate but becomes inefficient as evidence grows unlikely.

## relation

The set of tuples of objects that stand in a given relationship. In constraint satisfaction, a constraint pairs a scope of variables with a relation defining which value combinations those variables may take.

## relational extraction

The extraction of relations (structured facts linking entities) from unstructured text. AIMA discusses this within information extraction; Brin's DIPRE system (Dual Iterative Pattern Relation Extraction) and later systems such as TEXTRUNNER performed extraction over open, expanding sets of relations, bootstrapping templates and examples from a few labeled instances.

## relational reinforcement learning

Reinforcement learning applied to structured (relational) representations of states and actions rather than atomic ones.

## relational uncertainty

Uncertainty about the relations among objects rather than about attribute values, arising in relational probability models. In AIMA's book-recommendation example, not knowing the value of a relation such as Author(b) makes the very dependency structure of the Bayes net uncertain, which is an instance of relational uncertainty.

## relative error

An error measure expressed as a fraction of the true or reference value, so that errors of differing magnitudes can be compared on a common scale (in contrast to absolute error, which AIMA uses as a bound on how far computed utilities may deviate). Note: AIMA does not define "relative error" as a distinct margin term.

## relative likelihood

The ratio of the likelihoods that two competing hypotheses assign to the same observed data, used to judge which hypothesis better explains the evidence. Note: AIMA does not define "relative likelihood" as a distinct margin term; the idea underlies maximum-likelihood and MAP hypothesis selection.

## relaxed problem

A problem with fewer restrictions on the actions than the original. Because removing restrictions only adds edges to the state-space graph (making it a supergraph of the original), any optimal solution to the original is also a solution to the relaxed problem, so the exact cost of an optimal solution to the relaxed problem is an admissible (and, being exact, consistent) heuristic for the original. If the problem is stated in a formal language, relaxed problems can be constructed automatically.

## relevance

The property of bearing on the choice of action or the answer to a query. AIMA notes that a task environment is effectively fully observable if the sensors detect all aspects that are relevant to the choice of action, and that relevance in turn depends on the performance measure.

## relevance feedback

An information-retrieval technique that improves query results by using a user's judgments of which returned documents are relevant, then refining the query accordingly. Note: AIMA discusses relevance in IR but does not define "relevance feedback" as a distinct margin term.

## relevant

Describing information, sensor data, actions, or facts that bear on the current choice of action or query, where what counts as relevant depends on the performance measure. Note: AIMA uses "relevant"/"relevance" descriptively rather than as a distinct defined margin term.

## relevant-states

In backward (regression) search for planning, the set of states from which applying a given action or plan would lead toward the goal, used to guide the backward search. Note: AIMA does not define "relevant-states" as a distinct margin term; the concept appears in its treatment of regression and backward chaining.

## ReLU

Abbreviation for the rectified linear unit, an activation function common in modern neural networks that passes positive inputs unchanged and outputs zero for negative inputs.

## renaming

A step in first-order inference in which the variables of a sentence are replaced with fresh, previously unused names. AIMA notes that a fact is not considered "new" during forward chaining if it is merely a renaming of a known fact (identical except for variable names)-for example, Likes(x, IceCream) and Likes(y, IceCream) are renamings of each other. The STANDARDIZE-VARIABLES function performs this renaming.

## rendering

In computer graphics, generating a two-dimensional image from a model of a three-dimensional scene. It is essentially the inverse of the vision problem: AIMA describes a rendering model as one that captures the physical, geometric, and statistical processes that produce the image (stimulus) from the world.

## rendering model

In the model-based approach to vision, a model that describes the physical, geometric, and statistical processes by which the world produces the visual stimulus. Although modern rendering models are sophisticated and exact, the stimulus they produce is usually ambiguous (for example, a white object under low light may look like a black object under intense light).

## repeated game

The simplest kind of multiple-move game, in which players face the same one-shot choice repeatedly, each time with knowledge of the history of all players' previous moves.

## repeated state

A state that is generated more than once in a search tree, typically produced by a cycle (a loopy path). Because there is no limit to how often a loop can be traversed, undetected repeated states can make the search tree infinite even when the state space is finite.

## representation

An encoding of an input; in a deep network, each intermediate computation can be viewed as a different representation of the input.

## representation learning

The goal of learning useful internal representations (features) of the input from data, rather than relying on hand-engineered features.

## representational frame problem

The specific manifestation of the frame problem concerned with how to represent, compactly, the many aspects of the world that remain unchanged by an action.

## Residual networks

A popular and successful approach to building very deep networks, developed for convolutional models, in which layers learn residual functions relative to their inputs, easing the training of deep architectures.

## resolution

A single inference rule that, coupled with any complete search algorithm, yields a complete inference procedure for propositional and first-order logic. Given two clauses containing complementary literals, it produces a new clause called the resolvent; theorem proving proceeds by refutation, deriving the empty clause from the negated goal.

## resolution closure

The set of all clauses derivable by repeatedly applying the resolution rule to a set of clauses and their derivatives; by the ground resolution theorem, if the original set is unsatisfiable this closure contains the empty clause.

## resolvent

The new clause produced by applying the resolution rule to two clauses that contain complementary literals (for first-order logic, literals that unify with each other's negation after standardizing apart). The resolvent contains all the literals of the two original clauses except the complementary pair that was eliminated, with the unifier applied.

## resource constraint

A limitation the world imposes on the resources available for carrying out actions (e.g., a fixed number of staff who cannot be in two places at once); each action in a scheduling problem has a duration and an associated set of resource constraints.

## resource constraints

Limitations the world imposes on the resources available for actions (such as a limited number of staff); in scheduling, each action has a duration together with the set of resource constraints it requires.

## result set

In information retrieval, the collection of documents that a system returns in response to a query. (This exact term is not defined in AIMA; the definition follows standard information-retrieval usage consistent with the book's treatment of IR.)

## rete

A matching algorithm used in production systems whose cost does not scale linearly with the number of ground facts in the knowledge base, making rule-condition matching efficient. (Rete is Latin for "net.")

## Rete algorithm

An efficient pattern-matching algorithm used in forward-chaining production systems that shares computation across rules so the cost of finding which rules are triggered does not grow linearly with the number of rules on each cycle. (This exact term is not defined in AIMA; the definition follows standard usage consistent with the book's discussion of production systems.)

## retrograde

Describing analysis that works backward from terminal (solved) positions. In retrograde minimax search for endgames, one marks won positions, reverses the rules of the game to make backward moves, and propagates win/loss/draw values back toward earlier positions, building a complete lookup table (as done for all chess endgames with seven or fewer pieces).

## reusable

Describing a resource that is occupied while an action is using it but becomes available again once the action is over (for example, a pilot occupied during a flight), as opposed to a consumable resource that is used up.

## revelation principle

A result in mechanism design stating that any mechanism can be transformed into an equivalent truth-revealing (incentive-compatible) mechanism, in which each participant's dominant strategy is to reveal its true value. Part of mechanism design consists of finding these equivalent truthful mechanisms.

## revenue equivalence theorem

A result in auction theory stating that, with a few minor caveats, any auction mechanism in which bidders have values known only to themselves (but drawn from a commonly known probability distribution) yields the same expected revenue to the seller. It implies auction formats compete on qualities other than revenue.

## reward

In a Markov decision process or reinforcement learning, the feedback R(s, a, s') the agent receives on each transition, distinguished by the performance standard as direct feedback on the quality of behavior. Rewards may be positive (reward) or negative (penalty), and the agent seeks to maximize their total over time.

## reward shaping

A technique, originally used in animal training, for helping an agent learn in environments with sparse rewards by supplying additional rewards, called pseudorewards, for making progress. It can speed up learning greatly but risks the agent learning to maximize the pseudorewards rather than the true rewards.

## reward-to-go

The expected total reward an agent receives from a given state onward. Direct utility estimation defines the utility of a state as its expected reward-to-go and uses each trial's observed reward-to-go as a sample to estimate that utility.

## risk-averse

Describing an agent whose utility curve has decreasing slope (concave), so that the utility of a lottery is less than the utility of receiving its expected monetary value as a sure thing; such an agent prefers the sure thing to an equal-EMV gamble. Risk aversion is the basis of the insurance industry.

## risk-neutral

Describing an agent whose utility curve is linear, so that it judges gambles purely by their expected monetary value. For small changes in wealth almost any utility curve is approximately linear, so agents are expected to be roughly risk-neutral for small gambles.

## risk-seeking

Describing an agent whose utility curve has increasing slope (convex), as in the "desperate" region at large negative wealth, so that it prefers a gamble over a sure outcome of equal expected monetary value.

## Robbins algebra

A simple set of axioms proposed by Herbert Robbins in 1933 that appeared to define Boolean algebra; no proof could be found until the EQP theorem prover finally produced one.

## Robocup

An international robotic soccer competition used as a standard testbed for research in robotics and multiagent systems; its 2D simulator provides detailed continuous-state motion models for studying perception, control, and coordination.

## robot navigation

A generalization of the route-finding problem in which, rather than following distinct paths, a robot can roam freely and effectively make its own paths. The search space may be two-dimensional for a simple mobile robot but becomes many-dimensional (one dimension per joint angle) for robots with arms and legs, and real robots must also cope with sensor and motor errors and partial observability.

## robotic soccer

A competitive, dynamic, partially observable domain in which teams of robots play soccer, used as a benchmark for perception, control, and multiagent coordination (see RoboCup).

## robotics

The subfield of AI concerned with building agents that can physically manipulate objects and move about in the world.

## robust

Describes a decision (also called minimax) that is chosen to give the best result in the worst case, rather than optimizing expected value, formalized in fields such as control theory, decision analysis, and risk management.

## robust control

An approach to handling uncertainty that considers a set of possible models without assigning probabilities to them and seeks a policy giving the best outcome in the worst case over that set, rather than using probabilistic methods.

## robust control theory

An approach to control that allows for a set of possible models without assigning them probabilities and defines an optimal robust policy as one giving the best outcome in the worst case over that set. It can be viewed as a game between the agent and an adversary who picks the worst possible outcome, yielding a minimax solution.

## ROC curve

The receiver operating characteristic curve, a chart that plots false positives against true positives as a classifier's decision threshold (hyperparameter) is varied, helping visualize good tradeoff points. The area under the ROC curve (AUC) gives a single-number summary.

## rollout

In Monte Carlo tree search, a simulation (also called a playout) that chooses moves in turn for each player from a given state until a terminal position is reached, where the rules of the game determine the outcome. The value of a state is estimated by averaging the utilities of many such rollouts.

## Roomba

A commercial robot vacuum cleaner, cited in AIMA as an example of a widely deployed personal (consumer) robot.

## root mean square

The RMS of a set of values (often samples of a random variable) is the square root of the mean of the squares of those values.

## Root Mean Square (RMS)

The **root mean square (RMS)** of a set of values (often samples of a random variable) is the square root of the mean of the squares of those values:
<br>***square_root[(a1^2 + a2^2 + ... + aN^2) / N], where a1, a2, ..., aN are the values***
<br> As an error estimator, the values are the individual errors, so RMS error is the square root of the mean squared error. AIMA uses RMS error, for example, to measure how far learned utility estimates deviate from their true values.

## rules

Another name for implications (if-then statements): a sentence of the form "if premise then conclusion." Rules are the basic form for representing knowledge in production systems and logical inference.

# S

## S-set

In the version space (candidate elimination) learning algorithm, the S-set is the boundary set consisting of the most specific hypotheses consistent with the examples seen so far. Together with the most-general G-set it delimits the version space of all consistent hypotheses.

## safely explorable

A property of a state space in which some goal state is reachable from every reachable state, so that no action leads to an irreversible dead end. Online search and exploration algorithms are generally only guaranteed to work in state spaces that are safely explorable.

## safety

In reinforcement learning, the concern of how an agent can avoid disaster while trying previously unexplored actions in order to learn how to behave in its environment.

## sample complexity

The number of training examples an algorithm requires—expressed as a function of the error bound and confidence—in order to return, with high probability, a hypothesis whose error is at most a specified amount (i.e., one that is probably approximately correct).

## sample size disparity

A source of bias in machine learning that arises when some groups are represented by far fewer training examples than others, so that models perform worse for the underrepresented groups even when there is no underlying societal bias.

## sample space

In probability theory, the set of all possible worlds (outcomes) under consideration. These possible worlds are mutually exclusive and exhaustive; the sample space is denoted by uppercase omega, with lowercase omega denoting an individual element.

## sampling rate

The frequency at which a continuous (e.g., analog audio) signal is measured and converted into discrete digital values, expressed as samples per second.

## SARSA

A reinforcement-learning algorithm (named for the state-action-reward-state-action quintuple) closely related to Q-learning. Its update backs up the Q-value of the action actually taken in the next state, rather than the best available action; this makes it an on-policy method, in contrast to off-policy Q-learning.

## SAT

The Boolean satisfiability problem: determining whether a sentence in propositional logic can be made true by some assignment of truth values to its symbols. It was the first problem proved to be NP-complete, and many computer-science problems can be cast as SAT.

## satisfiability

A sentence is satisfiable if it is true in (satisfied by) at least one model. Satisfiability can be checked by enumerating models until one is found that makes the sentence true; determining it for propositional logic is the SAT problem.

## satisfiability threshold conjecture

The conjecture that, for every k greater than or equal to 3, there is a threshold ratio of clauses to symbols such that, as the number of symbols grows to infinity, random k-CNF sentences are satisfiable with probability approaching 1 below the threshold and 0 above it. It remains unproven, though such threshold behavior is empirically common.

## satisficing

Accepting solutions that are not optimal but merely "good enough." For example, satisficing search uses an inadmissible heuristic to explore fewer nodes and find a suboptimal-but-adequate solution more quickly.

## scaled orthographic projection

A simplified imaging model, used in place of perspective projection, appropriate when the depths of all points on an object fall within a narrow range. The perspective scaling factor is approximated by a constant, so scene coordinates project to the image plane by a fixed scale; foreshortening still occurs due to object tilt.

## scanning lidars

Active range sensors (lidar = light detection and ranging) that emit laser beams and measure the reflected beam, sweeping across the environment via mirrors or rotating elements. They yield centimeter-accurate range measurements at long distances and outperform time-of-flight cameras at longer ranges and in bright daylight.

## scene

In computer vision, the physical three-dimensional world that is imaged by a sensor. Light from the scene passes through the imaging geometry to form an image on the sensor's grid of pixels.

## schedule

In planning, an assignment of start (and end) times to all the actions of a problem, consistent with ordering and resource constraints. For example, the earliest and latest start times of all actions together constitute a schedule for the problem.

## schedulers

Systems that perform scheduling—assigning times and resources to a set of actions or jobs subject to constraints (as in job-shop scheduling or spacecraft operation planning). Automated planner-schedulers combine planning with such time and resource allocation.

## scheduling

The problem of assigning start times and resources to a set of actions subject to ordering and resource constraints (for example, limited staff or equipment), so that all actions can be carried out.

## schema

In genetic algorithms, a pattern (template) describing a set of individuals sharing certain fixed components; for example, the schema 246***** describes all 8-queens states whose first three queens are in columns 2, 4, and 6. If a schema's average fitness exceeds the population mean, its instances tend to grow in number over generations.

## Scrabble

A word game in which players form high-scoring words from letter tiles; cited in AIMA as a game where computer programs surpass amateur human play, notably in generating high-scoring moves.

## sealed-bid auction

An auction mechanism in which each bidder submits a single secret bid, typically in writing, and the bids are opened together; the highest bidder wins. It requires much less communication than an ascending-bid auction.

## sealed-bid second-price auction

An auction (also called a Vickrey auction) in which bidders submit sealed bids and the highest bidder wins but pays the price of the second-highest bid. Bidding one's true value is a dominant strategy, making it truth-revealing; it is widely used in distributed AI systems.

## search

The process by which a problem-solving agent, using atomic state representations, simulates sequences of actions to find a path from the initial state to a goal before acting in the world. Algorithms are informed (able to estimate distance to the goal) or uninformed.

## search algorithm

An algorithm that takes a search problem as input and returns a solution, or an indication of failure, typically by building a search tree over the state-space graph.

## search cost

A measure of the resources (typically time, often counted as the number of nodes generated or expanded) consumed by a search algorithm in finding a solution, as distinct from the cost of the solution path itself.

## search tree

A tree superimposed over the state-space graph in which the root is the initial state, each node corresponds to a state, and edges correspond to actions; it represents the paths explored from the initial state while trying to reach a goal. It is distinct from the state space (which may be infinite), since multiple tree nodes can correspond to the same state.

## segmentation

The process of partitioning an image into groups (regions) of pixels that are similar in visual properties such as brightness, color, and texture, so that boundaries between regions correspond to large changes in those attributes—ideally aligning with object boundaries.

## selection

In a genetic algorithm, the process of choosing which individuals will become parents of the next generation—for example, selecting individuals with probability proportional to their fitness, or randomly picking a subset and keeping the fittest of it.

## selection problem

The task of identifying which of several options (for example, which bandit arm) is best, as opposed to maximizing cumulative reward while learning.

## semantic ambiguity

Ambiguity in the meaning of an utterance, in which a phrase can be understood to denote more than one distinct situation. It often arises from syntactic ambiguity, as when different parses of a sentence yield different meanings.

## semantics

The rules that define the meaning of sentences in a language—specifically, the truth of a sentence with respect to each possible world (model). In propositional logic, a model fixes a truth value for every proposition symbol, and the semantics determine each sentence's truth.

## semi-supervised learning

A learning setting in which a small number of labeled examples is supplied along with a large collection of unlabeled examples; the algorithm uses the labeled data together with structure in the unlabeled data to improve its hypothesis.

## semidynamic

Describes an environment that does not itself change with the passage of time while the agent deliberates, but in which the agent's performance score does change over time. Chess played with a clock is semidynamic (as opposed to fully dynamic taxi driving or static crossword puzzles).

## semiotics

The study of signs and symbols and how they convey meaning—the relationship between symbols, what they denote, and their interpreters.

## semisupervised parsing

A parsing approach that begins with a small set of manually parsed sentences (trees) and augments the training data with a large number of unparsed sentences to improve the parser.

## sensitivity analysis

An analysis of how much the output of a model changes as its parameters are varied. In probabilistic and decision-theoretic systems it is especially important because the probabilities are typically learned or estimated (and thus uncertain); it identifies which parameters have the largest effect on the expected utility or the chosen decision, and how large a perturbation is needed to change the decision.

## sensor interface layer

In a robot's layered software architecture, the lowest level that mediates between the physical sensors and actuators and the higher-level representation and reasoning components—the code that translates raw sensor readings into percepts and higher-level action commands into motor signals.

## sensor Markov assumption

The assumption that the current evidence (sensor reading) depends only on the current state, i.e., P(E_t | X_{0:t}, E_{1:t-1}) = P(E_t | X_t). It lets the sensor model be specified as a single conditional distribution of the observation given the current state.

## sensorless

Describing a problem (also called a conformant problem) in which the agent's percepts provide no information at all. The agent must reason and act over belief states rather than physical states; a solution is a fixed sequence of actions (not a conditional plan) that is guaranteed to reach a goal from any possible starting state, effectively coercing the world into a goal state.

## sentiment analysis

A text-classification task in which a model examines a piece of text, such as a product or movie review, and outputs a classification of the sentiment expressed (for example, positive or negative).

## sequence form

A compact representation of an extensive-form (imperfect-information) game that is only linear in the size of the game tree rather than exponential. Instead of representing strategies, it represents paths through the tree (one per terminal node), allowing standard linear-programming methods to solve much larger games.

## sequence to sequence

A class of neural network models that take an input sequence and produce an output sequence, commonly used for tasks such as machine translation.

## sequential

A property of a task environment in which the current decision affects future decisions, so actions have long-term consequences and cannot be considered in isolation. It contrasts with an episodic environment, where each percept-action episode is independent of the others.

## sequential importance sampling

An approximate inference method that propagates a set of weighted samples forward through time, updating each sample's importance weight as new evidence arrives; it is the basis for particle filtering.

## sequential Monte Carlo

A family of sampling algorithms for filtering in temporal models, of which particle filtering is the central example. Samples ("particles") are propagated forward and periodically resampled in proportion to their likelihood given the evidence; also known in statistics as sequential importance sampling with resampling (SIR).

## set of support

A resolution refinement strategy in which every resolution step is required to involve at least one clause from a designated set (the set of support), with each resolvent added back into that set. Choosing the set of support so that the remaining sentences are jointly satisfiable (e.g., using the negated query) preserves completeness while greatly reducing the search space and producing goal-directed proofs.

## set semantics

An interpretation of a state (or belief state) as a set of ground atoms, where atoms present are true and atoms absent are false. Applying an action updates this set by removing atoms in the action's delete list and adding those in its add list; it requires that the state not contain both an atom and its negation.

## set-cover problem

The problem of choosing the minimum number of sets whose union covers a target. In planning it arises when estimating a heuristic by counting the minimum number of relaxed actions whose combined effects satisfy the goal. It is NP-hard, but a greedy algorithm approximates the optimal cover within a logarithmic factor (at the cost of admissibility).

## set-level

A planning-graph heuristic that estimates the cost of a conjunctive goal as the level in the planning graph at which all the goal literals appear together without any pair being mutually exclusive (mutex).

## SGD

Abbreviation for stochastic gradient descent, a variant of gradient descent that estimates the gradient from a small randomly selected subset of training examples at each step.

## shading

The variation in the intensity of light received from different portions of a surface in a scene, determined by the scene geometry and the reflectance properties of the surfaces. Because a surface facing the light source appears brighter and one facing away appears darker, shading serves as a cue to the 3D shape of a surface.

## shadow

A region of a surface that appears dark because the surface points cannot see (are occluded from) the light source, or receive light only at a tangential angle. Following Lambert's cosine law, such patches reflect little light and produce dark image pixels.

## shape

The three-dimensional geometric form of a surface or object in a scene, which vision systems recover from image cues such as shading, texture, and motion. Recovering shape from images ("shape from X") is a central problem in computer vision.

## shaping theorem

A result in reinforcement learning stating that adding a potential-based term of the form gamma*Phi(s') - Phi(s) to the reward function leaves the optimal policy unchanged, allowing extra guidance (reward shaping) without altering what behavior is optimal.

## shared models

The practice of sharing trained models rather than raw data, so that parties can benefit from learning while keeping their underlying data private.

## shaving

A constraint-reasoning technique that tentatively assigns a value to a variable and, if this assignment leads to inconsistency via constraint propagation, permanently removes that value from the variable's domain, thereby pruning the search space.

## shift-reduce parsing

A deterministic parsing strategy that processes a sentence word by word, at each step choosing either to shift the next word onto a stack or to reduce the top constituents on the stack according to a grammar rule.

## shortcuts

Artificial edges added to a graph that represent an optimal multi-action path between two nodes, used by route-finding algorithms to save time. For example, precomputed shortcuts between major cities let a search cover a large portion of a route in a single step.

## shoulder

In a state-space landscape, a plateau (flat region) from which upward progress is still possible via an exit at one edge, as opposed to a flat local maximum. Allowing sideways moves across a plateau can succeed if it is really a shoulder, letting hill climbing eventually resume its ascent.

## sibyl attack

The use of many fake identities (sybils)—for example, thousands of login IDs created by a single dishonest user—to confound or manipulate a reputation or recommendation system. It illustrates identity uncertainty, the difficulty of knowing which logical terms actually refer to the same underlying object.

## sideways move

A move in local search (e.g., hill climbing) to a neighboring state with the same objective value as the current state, allowed in order to traverse a plateau in the hope that it is a shoulder. Since it risks wandering forever on a flat local maximum, the number of consecutive sideways moves is typically capped.

## sigmoid layer

In a neural network, a layer that applies the logistic (sigmoid) function to a weighted sum of its inputs plus a bias, producing an output squashed into the range between 0 and 1.

## sigmoid perceptron

A perceptron-style unit that passes its weighted sum of inputs through the logistic (sigmoid) activation function instead of a hard threshold, producing a smooth, differentiable output between 0 and 1. This differentiability makes it trainable by gradient methods and suitable for Boolean classification.

## significance test

A statistical procedure that begins by assuming a null hypothesis (no underlying pattern), then measures how far the observed data deviate from what pure chance would produce. If such a deviation would be statistically unlikely under the null hypothesis (conventionally a probability of 5% or less), the pattern is judged significant—used, for example, in decision-tree pruning to decide whether an attribute's information gain is meaningful.

## similarity networks

A graphical representation, related to Bayesian networks, that helps an expert specify a probabilistic model by focusing on pairs of competing hypotheses and modeling only the variables relevant to distinguishing them, then combining the local models into a full network.

## simple reflex agent

The simplest kind of agent, which selects actions based only on the current percept, using condition-action rules, and ignores the rest of the percept history.

## simulated annealing

A stochastic local search algorithm, inspired by the metallurgical annealing process, that resembles hill climbing but selects random moves. An improving move is always accepted; a worsening move is accepted with a probability that decreases with the move's badness and with a "temperature" parameter that is gradually lowered, allowing early escape from local optima while converging as the temperature drops.

## simulation

In game-playing search such as Monte Carlo tree search, a playout (also called a rollout) that chooses moves according to a playout policy until the game is finished, in order to estimate the value of a state.

## simultaneous localization and mapping

The robotics problem, abbreviated SLAM, of building a map of an unknown environment while simultaneously estimating the robot's own location within that map.

## simultaneous localization and mapping (SLAM)

The problem in which a robot must build a map of an unknown environment while at the same time determining its own location within that map. Because location and map are mutually dependent and uncertain, SLAM is solved with probabilistic techniques such as Kalman/extended Kalman filters, particle filters, and expectation-maximization.

## single agent

A property of a task environment in which only one agent is acting, so there are no other agents whose behavior must be modeled or reasoned about (in contrast to a multiagent environment). Solving a crossword puzzle alone is a single-agent environment, whereas chess is two-agent.

## singly connected

Describing a Bayesian network (also called a polytree) in which there is at most one undirected path between any two nodes. Exact inference in singly connected networks has time and space complexity linear in the size of the network, unlike the potentially exponential cost in multiply connected networks.

## singular

Describing a matrix that has no inverse. AIMA notes that for a singular matrix the inverse does not exist (whereas for a nonsingular matrix it can be computed efficiently); singularity arises, for example, when data are linearly dependent.

## singular extensions

A technique to mitigate the horizon effect in game search: a clearly best move that stands out from the alternatives is allowed to extend the search beyond the normal depth limit. Because such moves are rare, this adds little cost.

## singularity

The hypothesized point—also called the technological singularity—at which self-improving computers reach a superhuman level of performance and then rapidly improve themselves further, driven by an "intelligence explosion." AIMA notes that extrapolating from falling computation costs to a singularity is questionable, since technologies typically follow S-shaped curves rather than sustained exponential growth.

## situation

In the situation calculus, a logical term denoting a snapshot of the world, produced from an initial situation by a sequence of actions; fluents are evaluated relative to situations.

## situation calculus

A method of describing planning problems in first-order logic. It uses successor-state axioms (as SATPLAN does in propositional logic), but first-order logic allows more flexible and succinct axioms. It has advanced the theoretical understanding of planning, though it has had limited practical impact because first-order provers are less developed than propositional satisfiability solvers.

## skeletonization

A motion-planning approach that reduces the robot's free configuration space to a one-dimensional network (a "skeleton") of curves, so that path planning is reduced to finding a route along this graph. Classic realizations include methods based on Voronoi diagrams and visibility graphs.

## skip-gram

A word model in which one counts words that appear near each other while skipping one or more intervening words, used to capture word co-occurrence and build word representations.

## skip-gram negative sampling

A training method (abbreviated SGNS, part of the word2vec package) that learns word embeddings from a skip-gram model by contrasting observed word pairs against randomly sampled ("negative") word pairs.

## Skolem constant

In first-order logic, the new constant symbol introduced to replace an existentially quantified variable during Existential Instantiation, denoting some specific but unnamed object.

## Skolemization

The process of removing existential quantifiers from a logical sentence during conversion to conjunctive normal form. In the simple case an existential variable is replaced by a new Skolem constant (as in Existential Instantiation); more generally, when the existential is inside universal quantifiers, it is replaced by a Skolem function whose arguments are the enclosing universally quantified variables.

## slack

In scheduling, the amount of time an action can be delayed without lengthening the overall plan, computed as the difference between an action's latest possible start time and its earliest possible start time (LS − ES). Actions on the critical path have zero slack.

## slant

A geometric property of a surface describing how much it is tilted away from the viewer (the angle between the surface normal and the line of sight). Estimating surface slant (together with tilt) from cues such as texture gradients is a classic problem in recovering three-dimensional shape from images.

## sliding window

An object-detection technique in which a small rectangular window is moved across the image and the contents of each window position are classified (e.g., by a CNN). High-scoring detections are kept and conflicts among overlapping windows are resolved to yield a final set of located objects.

## sliding-block puzzles

A family of standardized problems in which numbered tiles must be slid around a board into a goal configuration; the 8-puzzle and 15-puzzle are well-known examples.

## small-scale learning

A learning regime, emphasized in traditional statistics and early machine learning, in which the number of training examples is small (dozens to low thousands). Here generalization loss is dominated by approximation error (the true function not being in the hypothesis space) and estimation error (too few examples to control variance), rather than by computational limits, in contrast to large-scale learning.

## SMDP

A semi-Markov decision process, an extension of an MDP in which actions can take differing (possibly nondeterministic) amounts of time to complete, useful for representing hierarchical or temporally extended behaviors.

## smoothing

In temporal probabilistic inference, the task of computing the posterior distribution over a past state given all evidence up to the present. Because it incorporates later evidence, smoothing yields a better estimate of an earlier state than was available at that time.

## soccer

Robotic soccer (as in the RoboCup competition) is a standard testbed for AI and robotics, featuring a competitive, dynamic, partially observable multi-agent environment. It is cited as an example of a physical game requiring perception, real-time control, and coordination among teammates.

## social laws

Conventions that are so widespread that they govern allowable actions for all agents in an environment (for example, driving on a particular side of the road). Like human language conventions, what matters is not the specific rule chosen but that all agents agree on and follow it.

## societal bias

Bias present in society that machine learning models can absorb from their training data and then reproduce or amplify in their predictions.

## Socratic reasoner

A reasoning agent characterized by the classical syllogistic style of deduction—drawing conclusions from general premises about particular cases (e.g., "Socrates is a man; all men are mortal; therefore Socrates is mortal").

## soft margin

A support vector machine classifier that permits some training examples to fall on the wrong side of the decision boundary, assigning each such example a penalty proportional to the distance it would need to move to be classified correctly. This tolerates noisy, non-separable data instead of insisting on a perfect linear separator.

## softmax

A function used in a neural network output layer that converts a vector of real-valued scores into a normalized probability distribution over discrete classes.

## softmax function

A function used in the output layer of a neural network that converts a vector of real-valued inputs into a vector of nonnegative numbers that sum to 1, interpretable as a probability distribution over classes. Because it exponentiates the inputs, it accentuates differences among them, while remaining smooth and differentiable; the sigmoid is a special two-class case of the softmax.

## software architecture

The overall organizational structure of a robot's control system, specifying how components (for sensing, planning, control, etc.) are decomposed and interact. Classic examples include the three-layer architecture and Brooks's reactive subsumption architecture.

## sokoban

A grid-world puzzle in which the agent must push a number of boxes scattered about the grid to designated storage locations. At most one box occupies a cell; the agent moves a box only by walking into it when the cell beyond is empty, and it cannot push a box into a wall or another box.

## solution

A sequence of actions leading from the initial state to a goal state (a path from initial state to goal). In fully observable, deterministic, known environments the solution is a fixed action sequence; in partially observable or nondeterministic environments a solution is instead a branching strategy (contingency plan) that prescribes different actions depending on the percepts received.

## sonar sensors

Active range finders that emit directional sound waves and measure the time and intensity of the reflected signal to estimate the distance to nearby objects. Sonar is the technology of choice for autonomous underwater vehicles and was widely used in early indoor robotics.

## sound

A property of an inference algorithm (also called truth-preserving): it derives only sentences that are entailed by the knowledge base. An unsound procedure can announce conclusions that do not actually follow. Model checking, where applicable, is a sound procedure.

## spam detection

A text-classification task that distinguishes spam email from legitimate (non-spam) email, a canonical application of machine-learning classifiers to natural-language text.

## sparse

Describing a system that is locally structured, meaning each subcomponent interacts directly with only a bounded number of others regardless of the total number of components. Sparse (locally structured) systems, such as well-structured Bayesian networks, grow in complexity linearly rather than exponentially, making large domains tractable.

## sparse model

A model in which many parameters (weights) are exactly zero, so that only a subset of features is actually used. L1 regularization tends to produce sparse models, effectively performing feature selection, whereas L2 regularization does not drive weights to zero.

## spatial reasoning

Reasoning about the geometric and topological structure of space—for example, about the relative positions, connectivity, and regions of objects—needed for tasks like navigation. AIMA notes that much useful spatial reasoning can be done qualitatively (e.g., the region connection calculus) without a full metric representation.

## specialization

The operation of making a hypothesis more restrictive so that it covers fewer examples, used in version-space and inductive learning to remove false positives; more generally, the relation in which a more specific category is a subcategory of a more general one, inheriting its properties while adding further constraints.

## specular reflection

Reflection in which incoming light leaves a surface in a lobe of directions determined by the direction from which the light arrived (as with a mirror). When the lobe is narrow, distinct objects can be resolved in the reflection; for broader lobes the surface shows bright highlights.

## specularities

The small, bright patches produced by specular reflection on surfaces whose reflectance lobe is somewhat broad (e.g., metal, plastic, painted, or wet surfaces). They move as the surface or light source moves; away from these patches the surface behaves as if diffuse, so many surfaces can be modeled as diffuse plus specularities.

## speech act

An utterance considered as an action performed by a speaker. It is up to the hearer to determine what type of act the speaker intended, such as a question, statement, promise, warning, or command. Handling speech acts is part of pragmatics; for example, a grammar can be extended so that a command is a verb phrase whose implicit subject is the hearer.

## Speech Recognition

The task of converting spoken audio input into a sequence of words. It is a classic AI/perception problem in which systems (increasingly using deep learning) transcribe speech; by 2017 the best systems reached human-level word error rates on conversational speech.

## speedy search

An unbounded-cost search algorithm; a version of greedy best-first search whose heuristic estimates the number of actions needed to reach a goal, ignoring the costs of those actions, so as to find some solution quickly.

## split point

For a decision-tree attribute with continuous values, a split point is an inequality (threshold) test that partitions the examples, e.g., "value > c." Good split points are found by sorting the attribute's values and considering only thresholds lying between adjacent examples that differ in classification.

## stable

Describes a solution or point that no rational agent has an incentive to move away from. For example, a Nash equilibrium is a stable point of a game because no player can gain by unilaterally deviating; likewise a matching is stable if no pair of agents both prefer each other to their current partners.

## stable model

The semantics underlying answer set programming, used to give meaning to logic programs with negation. A stable model is a set of ground atoms that is a consistent, self-supporting (minimal) interpretation of the program's rules; it can be seen as a refinement of circumscription or of negation as failure.

## standard lottery

A reference lottery used to assess utilities on a normalized scale: the agent is asked to choose between a prize and a lottery whose probability of the best outcome is adjusted until the agent is indifferent, thereby fixing the prize's utility.

## standard normal distribution

The special case of the Gaussian (normal) distribution with mean 0 and variance 1. It is the reference form of the normal distribution to which other normal variables can be rescaled by subtracting the mean and dividing by the standard deviation.

## standardized problem

A problem, such as the Rubik's Cube, designed to illustrate or exercise problem-solving methods, as opposed to a real-world problem whose formulation is idiosyncratic and whose solutions people actually use.

## standardizing apart

The technique of renaming the variables in one of two logical sentences before unifying them, so that the two sentences share no variable names. This avoids spurious name clashes that would otherwise prevent correct unification and inference.

## Starcraft

A real-time strategy video game used as a challenging AI testbed. StarCraft II is partially observable, multi-agent, nondeterministic, dynamic, and unknown, requiring management of many units over long time horizons; DeepMind's AlphaStar reached strong human-level play.

## start symbol

In a formal grammar, the distinguished nonterminal symbol that denotes the complete set of strings of the language, and from which all valid derivations begin. For English this is typically Sentence; for arithmetic it might be Expr, and for a programming language, Program.

## state abstraction

A many-to-one mapping from the states of a problem's ground (detailed) representation to a smaller set of abstract states. Used to construct heuristics and reduce problem size; the simplest form is to ignore some fluents.

## state estimation

The process of maintaining an agent's belief state—the set of possible states it could currently be in—as it receives percepts in a partially observable environment. Also called monitoring or filtering, it is a core function of any intelligent system.

## state space

The set of all possible states the environment can be in, forming part of the formal definition of a search problem. It can be represented as a graph whose vertices are states and whose edges are actions.

## state space graph

A graph whose vertices are states and whose edges are actions, defined by the initial state, the Actions function, and the Result function; a search tree can be superimposed on part of it, and a single state may be reachable by multiple paths.

## state-space landscape

A conceptual view of a local-search problem in which each state is a point whose "elevation" is given by an objective (or cost) function. The goal is to find the highest peak (a global maximum, for hill climbing) or lowest valley (a global minimum, for gradient descent).

## states

The possible configurations that an environment can be in; the set of all such configurations is the state space of a search problem.

## static

A property of a task environment: an environment is static if it does not change while the agent is deliberating. Static environments are easier to handle because the agent need not monitor the world or track the passage of time while deciding. (If only the performance score changes with time, the environment is semidynamic.)

## statically stable

Describing a robot posture or gait that remains balanced under gravity without motion, so the robot does not fall over even when standing still; for example, a hexapod gait that keeps three legs on the ground at all times.

## stationarity

The assumption, underlying learning from examples, that the process generating the data does not change over time (each example is drawn from the same distribution); without it, no guarantees about generalization can be made.

## stationarity assumption

The machine-learning assumption that future examples will resemble past ones—specifically, that every example is drawn from the same fixed prior probability distribution and is independent of previous examples (i.e., the examples are independent and identically distributed, i.i.d.).

## stationary distribution

For a Markov chain, the distribution over states that, once reached, remains unchanged under further transitions (it satisfies the fixed-point equation defined by the transition kernel). If the chain is ergodic, there is exactly one such distribution.

## stationary process

A process of change governed by laws that do not themselves change over time, so the transition (and sensor) models are the same for every time step. AIMA also calls this a time-homogeneous process; it lets one specify a single conditional probability table for all times rather than a different one per step.

## statistics

The field, emerging from the formalization of probability combined with the availability of data, concerned with collecting, analyzing, and drawing inferences from data.

## steepest ascent

The hill-climbing strategy of moving on each iteration to the neighboring state with the highest value, that is, in the direction of greatest improvement, terminating at a peak where no neighbor is better.

## stemming

A text-processing step that reduces inflected or derived words to a common root or stem (e.g., mapping "running," "runs," and "ran" toward "run") so that variant forms are treated as the same term.

## step cost

The cost incurred by taking a single action, given by the action-cost function ACTION-COST(s, a, s'). AIMA assumes action costs are additive, so the total cost of a path is the sum of its individual step (action) costs, and an optimal solution has the lowest total path cost.

## step size

A small constant, usually denoted alpha, that scales each update in gradient-based (steepest-ascent) search. If it is too small, too many steps are needed; if too large, the search may overshoot the optimum; techniques such as line search adaptively adjust it.

## stereo vision

Also called binocular stereopsis: inferring depth from two images of a scene taken from slightly different viewpoints. Depth is recovered from the disparity between the positions of a scene point in the two images, which first requires solving the correspondence problem of matching points across the two images.

## stochastic

A model of an environment is stochastic if it explicitly quantifies uncertainty with probabilities (e.g., "a 25% chance of rain"). AIMA distinguishes this from "nondeterministic," in which possibilities are listed but not assigned probabilities.

## stochastic beam search

A variant of local beam search that maintains diversity among its k states by choosing successors randomly with probability proportional to their value, rather than always keeping the top k successors. It is analogous to stochastic hill climbing.

## stochastic dominance

A relationship between two options with uncertain outcomes: one option stochastically dominates another (on an attribute where more is better) if its probability distribution assigns at least as much probability to high values across the whole range, so the dominated option can be discarded regardless of expected values.

## stochastic games

Games that include a random element, such as the roll of dice (e.g., backgammon), combining chance with skill. They are handled by adding chance nodes to the game tree so that expected values are computed over the possible random outcomes.

## stochastic gradient descent

A faster variant of gradient descent (abbreviated SGD) that, at each step, randomly selects a small number of training examples and updates the parameters based only on those, rather than on the entire dataset.

## stochastic hill climbing

A hill-climbing variant that chooses at random among the available uphill moves, with selection probability optionally varying with the move's steepness. It usually converges more slowly than steepest ascent but can find better solutions; first-choice hill climbing implements it by generating random successors until one improves on the current state.

## stochastic policy

A policy that, in each state, specifies a probability distribution over actions rather than a single deterministic action. Making the policy a smooth, probabilistic function of its parameters (e.g., via a softmax) avoids the discontinuities that hinder gradient-based optimization of deterministic policies; it is useful in partially observable and adversarial settings.

## stopping rule

In bandit problems, a strategy that runs one process up to some stopping time and then switches permanently to another; the optimal policy can be characterized in this form.

## stopping time

The finite time at which an optimal bandit strategy first switches away from the arm it has been pulling, for example the first time it pulls a second arm.

## straight-line distance

A heuristic function for route-finding that estimates the cost to the goal as the straight-line ("as the crow flies") Euclidean distance on the map between the current location and the goal. It cannot be derived from the problem's actions alone and requires world knowledge that it correlates with actual road distance; it is admissible because it never overestimates.

## strategic form

A representation of a game (also called normal form) as a matrix in which each player chooses a single strategy simultaneously and the entries give the resulting payoff (utility) to each player for every combination of strategies.

## strategy

In game theory, a strategy is what was elsewhere called a policy: a specification of what action to take. A pure strategy is a deterministic policy (for a single-move game, just a single action), whereas a mixed strategy is a randomized policy that selects actions according to a probability distribution.

## strategy profile

An assignment of a strategy to each player in a game. Given a strategy profile, the game's outcome is determined as a numeric value (utility) for each player; if players use mixed strategies, the outcome is defined by expected utility.

## strategy-proof

A property of a mechanism in which every agent has a dominant strategy—one that works best regardless of the strategies chosen by others—so that agents can act without needing to reason about others' behavior. When the dominant strategy involves revealing one's true value, the mechanism is also called truth-revealing or incentive compatible.

## strict dominance

A relationship between two options in which one has a value at least as good on every attribute (and better on some), so that the dominated option can be rejected outright without combining attributes into a single utility.

## strictly stable

Describing a controller that is able to return to and then stay near a reference trajectory or signal, as opposed to one that fails to remain anywhere near its reference.

## strong AI

The assertion (contrasted with weak AI) that machines which act intelligently are actually consciously thinking, not merely simulating thought. Over time the term shifted to also mean "human-level AI" or "general AI": programs able to solve an arbitrarily wide variety of tasks, including novel ones, as well as a human can.

## structural EM

A variant of the EM algorithm for learning Bayesian networks that can update the network structure as well as its parameters. It uses the current structure to compute expected counts in the E-step, then uses those counts to score potential new structures in the M-step, allowing several structural changes without recomputing expected counts each time. Its search space is over structures rather than over both structures and parameters.

## structured representation

A representation in which a state is described in terms of objects, each of which may have attributes of its own as well as relationships to other objects. More expressive than atomic or factored representations, it underlies relational databases, first-order logic, first-order probability models, and much of natural language understanding.

## stuff

The portion of reality that defies obvious individuation into distinct objects. Unlike "things" (count nouns such as aardvarks), stuff (mass nouns such as butter, water, or energy) has the property that any part of it is also an instance of the same kind—cutting a lump of butter yields two lumps of butter, but cutting an aardvark does not yield two aardvarks.

## subcategory

A category that is a subset of another category (the terms subcategory, subclass, and subset are used interchangeably); e.g., Basketballs is a subcategory of Balls. Subcategories organize knowledge through inheritance, since members inherit the properties of the categories they belong to.

## subgoal independence

The assumption that the cost of solving a conjunction of subgoals can be approximated by the sum of the costs of solving each subgoal independently. It is optimistic when subplans interact negatively (e.g., one deletes a goal achieved by another) and pessimistic (hence inadmissible) when subplans contain redundant actions.

## subject-verb agreement

The grammatical requirement that a verb agree with its subject in features such as person and number (e.g., third-person singular present tense); a constraint enforced in AIMA via augmented grammars and categories carrying such features.

## subjectivist

The view that probabilities characterize an agent's degrees of belief rather than any external physical fact about the world. The subjective Bayesian version allows any self-consistent assignment of prior probabilities to propositions but requires proper Bayesian updating as evidence arrives.

## subproblem

A reduced version of a given problem, whose optimal solution cost provides an admissible heuristic (a lower bound) for the full problem. For example, the cost of moving only tiles 1-4 and the blank into place in the 8-puzzle bounds the cost of solving the whole puzzle; such heuristics are stored in pattern databases.

## substitution

An assignment of values (bindings) to variables that, when applied to a logical expression, replaces its variables; also called a binding list. Unification produces the substitution that makes two expressions identical, and a substitution is the answer returned by ASKVARS giving specific values bound to each variable of a query.

## subsumption

The relation by which a more general logical sentence covers (is satisfied by) more specific instances, so that everything satisfying the more specific description also satisfies the more general one; queries can be ordered by generality, where a more general query subsumes the more specific ones obtained from it by substitution.

## subsumption architecture

A framework (Brooks, 1986) for assembling reactive robot controllers out of finite state machines whose nodes test sensor variables, whose arcs emit messages to motors or other machines, and which include internal timers—yielding augmented finite state machines (AFSMs). Higher layers can suppress or override lower ones, and there is no central world model; it lets a programmer compose increasingly complex controllers bottom-up.

## subsumption lattice

The lattice formed by ordering the queries that can be generated from a fact by generalization (variable substitution). Each node's child is obtained from its parent by a single substitution, and the highest common descendant of two nodes results from applying their most general unifier. It can be used to index stored facts for faster retrieval.

## successor

A new node (also called a child node) generated by applying an action to a given node's state during search; the state it contains is the result of that action, and the originating node is its parent.

## successor nodes

The new nodes generated when a node in the search tree is expanded, one child node for each applicable action and its resulting state.

## successor-state axiom

A logical axiom (in the situation calculus) that specifies the truth value of a fluent at the next time step in terms of what is true now and the action taken: the fluent is true afterward if and only if the action makes it true, or it was already true and the action does not make it false.

## Sudoku

A popular constraint satisfaction puzzle played on an 81-square board partly filled with digits 1-9; the goal is to fill the remaining squares so that no digit repeats in any row, column, or 3x3 box (each such row, column, or box is called a unit).

## sum of squared differences

A dissimilarity measure (SSD) used, for example, in optical flow and stereo matching to compare two blocks of pixels: it sums the squared differences of corresponding pixel values, so smaller values indicate closer matches.

## superpixels

Regions produced by over-segmenting an image so that no true object boundary is missed (though extra false boundaries may appear). Because there may be only hundreds of superpixels rather than millions of raw pixels, they substantially reduce the computational cost of subsequent algorithms while respecting object boundaries.

## supervised learning

A type of learning in which the agent observes input-output pairs and learns a function mapping inputs to outputs. Each output is called a label (e.g., an image labeled "bus" or "pedestrian"), and the learned function predicts the appropriate label for new inputs.

## support vector machine

A supervised learning model class (SVM), the most popular "off-the-shelf" approach in the early 2000s, that constructs a decision boundary maximizing the margin between classes; it is defined by a subset of the training points (the support vectors), can use the kernel trick to form nonlinear boundaries in a higher-dimensional feature space, and is a good method to try when no specialized prior knowledge about the domain is available.

## surely expanded

Describing nodes that any optimal search algorithm using a given admissible heuristic must expand, because their evaluation f-cost is strictly less than the cost of an optimal solution.

## surely expanded nodes

The nodes with f-cost strictly less than the optimal solution cost, which any A*-like search with an admissible heuristic is guaranteed to expand.

## surveillance

The use of technology to monitor people, for example through networks of cameras; raises significant privacy concerns discussed among AI's societal impacts.

## surveillance cameras

Cameras deployed in large numbers to monitor public and private spaces; their proliferation (millions in some countries) is a prominent example of technology used for surveillance.

## survey propagation

A message-passing algorithm for solving hard satisfiability problems, effective near the phase-transition region where problems are most difficult.

## switching Kalman filter

An extension of the Kalman filter that runs multiple Kalman filters, each assuming a different mode of behavior, and combines their predictions using a weighting that depends on how well each filter fits the data; the standard solution when a single linear model is inadequate.

## symbolic differentiation

A technique for computing derivatives by manipulating symbolic mathematical expressions, used (alongside automatic differentiation) as a basis for modern implementations of gradient computation such as back-propagation.

## symmetry reduction

A pruning technique that removes all but one of the symmetric branches of a search tree, since exploring symmetric alternatives yields no new information.

## symmetry-breaking constraint

A constraint added to a CSP to eliminate redundant symmetric solutions (value symmetry), thereby reducing the search space. For example, imposing an arbitrary ordering (such as requiring three color values to be in alphabetical order) ensures that only one of the interchangeable symmetric assignments remains valid.

## synchro drive

A robot locomotion mechanism in which all wheels turn and steer together in unison so that they always point in the same direction, letting the robot translate in any direction while maintaining a fixed orientation.

## synchronic

Concerning relationships among entities or facts holding at a single point in time, as opposed to diachronic relationships that hold across or over time.

## synchronization

An assumption for multiagent planning in which a global clock is available to all agents, every action takes the same amount of time, and actions at each step of the joint plan occur simultaneously, so agents execute in lockstep (some performing no-ops while waiting). It coordinates the timing of multiple agents so that their actions proceed in a consistent order.

## syntactic ambiguity

Ambiguity arising when a phrase has multiple parses—e.g., "I smelled a wumpus in 2,2," where the prepositional phrase can attach to either the noun or the verb. Syntactic ambiguity (also called structural ambiguity) often leads to semantic ambiguity, since different parses correspond to different meanings.

## Syntactic categories

Groupings such as noun phrase or verb phrase that constrain which words are likely at each point in a sentence and provide a framework for the sentence's meaning.

## syntactic sugar

An extension to or abbreviation of a standard syntax that does not change the semantics; any sentence using it can be "desugared" into an equivalent sentence in the ordinary notation (for example, infix operators in first-order logic).

## syntactic theory

An approach to representing mental objects (such as knowledge or belief) in which the objects are represented as character strings—a complex term denoting a list of symbols. For instance, CanFly(Clark) can be encoded as the list of its symbols; this approach requires care because it can lead to paradoxes.

## syntax

The specification, given by a representation language, of which sequences of symbols form well-formed sentences, independent of their meaning. For example, in ordinary arithmetic "x + y = 4" is a well-formed sentence while "x4y+ =" is not.

## synthesis

The (typically logic-based) construction of an artifact—such as a program, algorithm, or hardware circuit—from a formal specification, often by constructively proving a theorem asserting that an object meeting the specification exists (deductive synthesis).

## systematic

A property required for a search algorithm to be complete in an infinite state space: it must explore the space in a way that eventually reaches every state connected to the initial state.

# T

## table lookup

Producing an action by looking it up in a stored table indexed by the input rather than computing it. A table-driven agent keeps an explicit table mapping every possible percept sequence to an action; the book notes this approach is doomed because the table would be astronomically large for any real environment.

## tabled logic programming

An approach to logic programming that applies dynamic programming, using efficient storage and retrieval mechanisms to memoize (cache) the solutions to subgoals so they need not be recomputed.

## tabu search

A variant of hill climbing, from operations research, that maintains a tabu list of recently visited states that may not be revisited. This improves efficiency when searching graphs and can allow the algorithm to escape from some local minima.

## tactile sensors

Touch sensors, such as whiskers, bump panels, or pressure-sensitive skin, that respond to physical contact; they lie at the close end of range sensing.

## taxonomy

A taxonomic hierarchy formed by organizing categories via subclass (subset) relations, so that properties are inherited from more general categories to more specific ones. Examples include the biological classification of species, the Dewey Decimal system, and tax-code categories.

## Taylor expansion

A representation of a function as a sum of terms computed from its derivatives at a point; truncating it (a first-degree Taylor expansion) gives a local linearization of the function, as used in the extended Kalman filter.

## technological singularity

The name given by Vinge to I. J. Good's predicted "intelligence explosion," a hypothesized point at which machines, by recursively designing ever-better machines, so rapidly surpass human intelligence that subsequent progress becomes unpredictable to humans.

## technological unemployment

Job loss caused by machines and automation replacing human workers; a term coined by John Maynard Keynes, illustrated historically by automated looms, farm machinery, and ATMs displacing workers.

## template

In information extraction, a fixed structure with slots to be filled from text, used to recognize and pull out instances of a stereotyped relation or event.

## temporal logic

A special-purpose logic that assumes facts hold at particular times, and that those times (which may be points or intervals) are ordered.

## temporal-difference

A reinforcement-learning method that adjusts utility estimates toward the values that would hold at equilibrium when the estimates are correct, using the difference between successive predictions as an error signal to update a state's value toward the reward plus the estimated value of the successor state.

## temporal-projection

Predicting the state of the world at future times by projecting forward from the current situation through the effects of actions and events.

## tensors

In deep learning, the general name for the multidimensional arrays (of any number of dimensions) that hold the values flowing through a neural network's computational graph.

## term

In first-order logic, a logical expression that refers to an object. Terms include constant symbols and variables, and a complex term is formed by a function symbol followed by a parenthesized list of terms as arguments.

## terminal states

The states in a game where the game has ended; the utility (payoff) function defines the final numeric value for each player in a terminal state.

## terminal test

A test on a game state that is true when the game is over and false otherwise, distinguishing terminal states from those in which play continues.

## test set

A set of examples, separate from the training set, used to measure how well a learned model generalizes; a hypothesis generalizes well if it accurately predicts the outputs of the test set.

## text classification

The task of assigning a text document to one of a set of predefined categories, for example classifying an email as spam or not spam, or a news story by topic.

## Text-to-speech

Speech synthesis: the process of converting written text into spoken sound, the inverse of speech recognition.

## texture

In computer vision, a pattern on a surface that can be sensed visually and is usually roughly regular; a common model is a repetitive pattern of elements called texels. Texture is a property of an image patch rather than of a single pixel.

## the no free lunch theorem

A result stating that, when averaged over all possible learning tasks, every supervised learning algorithm performs equally well on new test points, implying no single learner is universally superior.

## theorem proving

Determining entailment by applying rules of inference directly to the sentences in a knowledge base to construct a proof of the desired sentence, without consulting models; it can be more efficient than model checking.

## Thompson sampling

A strategy for the bandit problem in which the next arm is chosen according to the probability that it is actually the optimal arm, given the observations so far, thereby balancing exploration and exploitation.

## thrashing

In memory-bounded search, the behavior in which an algorithm spends most of its time repeatedly regenerating and discarding the same nodes because it cannot hold them all in memory, analogous to thrashing in disk paging systems.

## three-layer architecture

A popular hybrid robot software architecture combining a reactive layer (low-level control), an executive/sequencing layer (which links the other two), and a deliberative layer (high-level planning).

## threshold function

A function that maps a real-valued input to a binary output, returning 1 when the input meets or exceeds a threshold and 0 otherwise; applying it to a linear combination of inputs yields a linear classifier such as the perceptron.

## tiling

A method of representing a continuous state space by covering it with overlapping regions (tiles); the features of an input are encoded by which tiles it falls into. Used for function approximation in reinforcement learning.

## tilt

The rotation of a camera or sensor up or down, one of the standard degrees of freedom (with pan) for aiming a viewing device.

## time and tense

Aspects of natural language that convey when an event occurs relative to the moment of speaking, requiring representations that relate linguistic tense to points or intervals on a time line.

## time line

An ordered representation of time as a sequence of points or intervals, along which events and the durations between them can be located and reasoned about.

## time of flight camera

A depth-sensing camera that measures the distance to each point in a scene by timing how long emitted light takes to travel to the surface and reflect back.

## time to answer

A measure of how quickly a system returns a response to a query, used in evaluating question-answering and information-retrieval systems.

## time well-spent

A movement, associated with Tristan Harris, that advocates designing technology to give users more well-rounded and beneficial choices rather than maximizing engagement.

## time-homogeneous

Describing a process of change whose governing laws stay fixed over time; the rules that determine how the world state evolves do not themselves change.

## tit-for-tat

A strategy for repeated games in which a player refuses (or cooperates) on the first move and thereafter simply copies whatever the opponent did on the previous move.

## Toeplitz matrix

A matrix whose structure (constant values along each descending diagonal) lets matrix multiplication implement a 1-D convolution, as arises when a convolutional kernel is applied.

## topological sort

A linear ordering of the nodes of a directed acyclic graph such that every edge points from an earlier node to a later one, used, for example, to order variables consistently with a set of dependencies.

## total functions

In first-order logic, functions that must return a value for every possible input tuple in the domain, so every object is mapped somewhere by the function.

## total Turing Test

An extended Turing test that requires the machine to interact with objects and people in the real world, so passing it demands computer vision, speech recognition, and robotics in addition to conversation.

## toy problem

A problem given a concise, exact description and used as a benchmark for comparing algorithms, in contrast to a real-world problem whose formulation is idiosyncratic and whose solutions people actually use.

## trace

A record of the sequence of steps an algorithm or agent takes as it executes, used to follow or debug its behavior.

## tractability

The property of a problem being solvable in a practical amount of time; roughly, a problem is intractable if the time to solve instances grows exponentially with the size of the instances.

## tragedy of the commons

A situation in which agents each acting in their own self-interest overuse a shared common resource, yielding an outcome that is worse for everyone than if they had cooperated.

## trail

In logic-programming implementations, a data structure that records the variable bindings made during unification so that they can be undone when the system backtracks.

## training curve

A plot of a learning system's performance (such as accuracy or error) against the amount of training experience, showing how the model improves as it processes more data.

## training set

The collection of examples given to a learning algorithm, from which it induces a hypothesis or model.

## transfer learning

A learning approach that improves performance on one task by reusing knowledge or experience gained from studying other, related tasks.

## transfer model

In machine translation, a model that translates by converting a source-language representation into a target-language representation at some intermediate level of analysis, rather than word by word.

## transhumanism

A social movement that advocates using technology, including AI and biotechnology, to enhance human capabilities and transcend current biological limitations.

## transhumanist

Relating to a social movement that anticipates and welcomes a future in which humans are augmented by, merged with, or replaced by robotic and biotechnological inventions.

## transient failure

A sensor failure in which the sensor intermittently returns incorrect (nonsense) values regardless of the true state of the world, such as a battery meter occasionally reading 0 when the battery is actually charged.

## transient failure model

A sensor model that explicitly assigns a nonzero probability to the sensor returning a completely wrong value, giving beliefs enough "inertia" to ride out temporary sensor glitches without a catastrophic change in the estimated state.

## transition kernel

The function defining a Markov chain that gives the probability of moving to state x' from state x; together with an initial state it fully specifies the chain.

## transition model

A description of what each action does, specifying the state (or distribution over states) that results from performing a given action in a given state.

## transition probability

In a stochastic model such as a Markov chain or MDP, the probability of moving to a particular next state given the current state (and, in an MDP, the action taken).

## transparency

An aspect of trust in which the workings of a system are open to the user, so that nothing is hidden and (ideally) the system can explain its behavior.

## transpose

The matrix obtained by interchanging the rows and columns of a matrix, so that entry (i, j) becomes entry (j, i).

## transposition table

A cache in game-tree search that stores the evaluation of positions already seen, so that a position reached by a different order of moves (a transposition) need not be re-searched.

## traveling salesperson problem

A touring problem (the TSP) in which the agent must visit every city exactly once by the shortest route and return to the start; it is a well-known NP-hard combinatorial optimization problem.

## treasure hunt

A myopic information-gathering problem (also called the least-cost testing sequence problem) with n locations, each holding treasure with some independent probability and costing a certain amount to inspect; the goal is to choose an efficient order in which to check them.

## tree decomposition

A method of solving a constraint satisfaction problem by grouping its variables into overlapping clusters arranged in a tree, so that the problem becomes tractable when the clusters are small.

## tree selection policy

In Monte Carlo tree search, the rule for choosing which moves to explore in the game tree, focusing computation on important parts by balancing exploration of little-tried states against exploitation of states that have performed well.

## tree width

A measure of how close a graph is to being a tree: the tree width of a tree decomposition is one less than the size of its largest cluster, and the tree width of the graph is the minimum over all its tree decompositions. Low tree width permits efficient inference.

## tree-like search

A search that does not check for redundant paths, treating the state space as if it were a tree even though it is really a graph; contrasted with graph search, which detects repeated states.

## treebank

A corpus of sentences annotated with their syntactic parse trees, such as the Penn Treebank, used to train and evaluate parsers.

## triangle inequality

The principle that one side of a triangle cannot exceed the sum of the other two; in heuristic search it underlies the consistency condition h(n) <= c(n,a,n') + h(n').

## trust

The confidence users must have in the systems they rely on; earned through processes such as verification and validation, certification, and transparency.

## truth

The relationship whereby a sentence is true in a possible world (model) when the world is arranged as the sentence asserts; semantics defines the truth of each sentence with respect to each possible world.

## truth value

In propositional logic, one of the values (true or false) that a model assigns to a proposition symbol.

## truth-preserving

Describing an inference rule that derives only conclusions that are true in every model in which the premises are true, so it never infers a falsehood from truths; such a rule is sound.

## truth-revealing

Describing a mechanism such as an auction (also called truthful or incentive compatible) in which each participant's best strategy is to reveal their true value or preferences.

## turbo decoding

An iterative decoding technique for error-correcting codes in which decoders pass probabilistic estimates back and forth; it turns out to be an instance of loopy belief propagation.

## Turing Test

A test proposed by Alan Turing (1950) as a thought experiment to sidestep the question "Can a machine think?": a computer passes if a human interrogator, after posing written questions, cannot tell whether the written responses come from a person or a computer. Passing it would require natural language processing, knowledge representation, automated reasoning, and machine learning.

## type A strategy

A game-playing strategy (Shannon's term) that considers all possible moves to a certain depth and then applies a heuristic evaluation function, exploring a wide but shallow portion of the tree.

## type B strategy

A game-playing strategy (Shannon's term) that ignores moves that look bad and follows promising lines as far as possible, exploring a deep but narrow portion of the tree.

## type constraint

A restriction that limits the values a variable, slot, or argument may take to members of a specified type or category.

## type signature

A specification of the type of each argument a function (or relation) takes and the type of value it returns.

# U

## UCB1

The upper-confidence-bound formula used as a selection policy in Monte Carlo tree search, ranking each candidate move by combining its estimated value with a bonus that favors moves that have been explored fewer times.

## UCT

Short for "Upper Confidence bounds applied to Trees," the most widely used selection policy in Monte Carlo tree search; it uses the UCB1 formula to decide which child node to explore next.

## ultraintelligent machine

I. J. Good's term for a machine that can far surpass all the intellectual activities of any person; because designing machines is one such activity, an ultraintelligent machine could design even better machines, leading to an "intelligence explosion."

## unary constraint

The simplest type of constraint in a constraint satisfaction problem: one that restricts the value of a single variable on its own.

## unbiased

Describing an estimator whose expected error is zero, so that on average its value equals the true quantity being estimated.

## unbounded cost search

A search setting in which a solution of any cost is acceptable as long as it can be found quickly; speedy search, a greedy best-first variant that estimates the number of actions to a goal, is an example.

## uncertainty

The condition in which an agent does not know for sure the current state of the world or the outcomes of its actions; acting under uncertainty is handled by reasoning with probabilities and maximizing expected utility.

## underconstrained

Describing a problem that has relatively few constraints for its variables, so that it has many solutions and tends to be easy to solve.

## understanding

In natural language processing, deriving the meaning of an utterance, building an internal representation of what was said rather than merely processing its surface form.

## unification

The process, used by all first-order inference algorithms, of finding a substitution that makes two logical expressions look identical; the UNIFY algorithm takes two sentences and returns such a substitution if one exists.

## unifier

A substitution returned by unification that makes two expressions identical when applied to them; the most general unifier is the one that makes the fewest commitments needed to unify them.

## uniform convergence theory

A tradition of sample-complexity analysis originating in statistics (Vapnik and Chervonenkis) that studies how quickly empirical estimates converge to true values across a hypothesis class, providing theoretical guarantees on learning.

## uniform-cost search

A best-first search (also known as Dijkstra's algorithm) that expands the node with the lowest path cost from the start, spreading out in waves of uniform path cost; it is optimal when step costs are nonnegative.

## Unimate

The first commercial robot, an arm whose name stood for "universal automation"; the first one was sold to General Motors in 1961 for use in manufacturing.

## uninformative prior

A prior distribution deliberately chosen to encode little or no prior knowledge, for example by setting the prior mean to zero and giving it a very large variance so the data dominate the inference.

## uninformed search

A search strategy (also called blind search) that is given no clue about how close a state is to the goal beyond the problem definition, and so cannot estimate distance to a goal.

## unintended side-effects

A failure mode of a utility function in which an agent, while maximizing its stated objective, produces harmful or undesirable changes to the world that were not accounted for; one remedy is designing low-impact agents that penalize such changes.

## unique action axioms

Axioms asserting that distinct actions are distinct so that no two actions can occur at the same time; in the book's logical formulation of planning these appear as action exclusion axioms, which prevent interfering actions from being taken simultaneously.

## unique string axiom

The unique-names assumption expressed as an axiom: distinct constant symbols (names or strings) are required to refer to distinct objects, so that different symbols denote different entities.

## unique-names assumption

The assumption that every distinct constant symbol refers to a distinct object in the world, so that no two differently-named constants can denote the same thing.

## unit clause

A clause containing exactly one literal, which fixes the truth value of that literal in any satisfying assignment.

## unit preference

A resolution strategy that prefers to perform resolutions in which one of the clauses is a unit clause, since resolving with unit clauses tends to produce shorter clauses and drive the search toward the empty clause.

## unit propagation

An inference process, resembling forward chaining, that repeatedly assigns the value forced by each unit clause and simplifies the remaining clauses; it is central to modern satisfiability solvers.

## unit resolution

A resolution inference rule in which one of the two clauses being resolved is a unit clause, yielding a shorter clause.

## units

The individual computing elements (entries) of a neural network, called units rather than neurons to avoid overly anthropomorphic language.

## units function

A function used in an ontology to attach a unit of measure to a numeric quantity, so that measures such as lengths can be represented (for example, the length of an object expressed in centimeters).

## universal approximator theorem

A family of results showing that a simple neural network with a single hidden layer (a linear transform, a nonlinear activation, and a linear output) can approximate any continuous function on a compact domain to arbitrary accuracy; it does not, however, guarantee that the network will actually learn the target function.

## universal grammar

The hypothesis that there is an innate body of grammatical structure common to all human languages, which every child possesses from birth.

## Universal Instantiation

The inference rule that lets us infer any sentence obtained by substituting a ground term for a universally quantified variable, and can be applied repeatedly to the same axiom.

## unknown

Describing an environment about which the agent does not know the "laws of physics"—the outcomes (or outcome probabilities) of its actions—so it must learn how the environment works. This is distinct from partial observability.

## unobservable

Describing an environment in which the agent's sensors give no information at all about the state, so the agent cannot perceive its situation.

## unrolling

Expanding a dynamic Bayesian network into a static network by replicating its slices across time to accommodate an observation sequence, allowing standard inference algorithms to be applied.

## unstable

A property of a learning algorithm, such as decision-tree learning, whereby a small change in the training data (even one new example) can produce a large change in the resulting model.

## unsupervised clustering

The problem of discerning multiple categories in a collection of unlabeled examples, grouping similar objects together without being told the categories in advance.

## unsupervised learning

Learning in which the agent finds patterns in the input without any explicit feedback; the most common unsupervised learning task is clustering.

## unsupervised parsing

An approach in which a grammar is learned directly from a collection of unparsed sentences, without access to the parse trees of those sentences.

## unsupervised translation

A form of unsupervised learning in which a translation system is learned without parallel (paired) bilingual training data.

## upper confidence bound

An exploration heuristic (UCB) that selects actions by adding an exploration bonus to estimated value, favoring options that have been tried only a few times; it appears both in bandit problems and in Monte Carlo tree search.

## upper confidence bounds on trees

A Monte Carlo tree search selection policy (UCT) that ranks each possible move by an upper confidence bound formula, balancing a move's estimated value against the uncertainty of that estimate to favor promising but under-explored lines.

## upper ontology

A general, top-level ontology of broad concepts (such as objects, events, time, and substances), so called because such graphs are drawn with the general concepts at the top; more specialized, domain-specific ontologies fit underneath it.

## Urban Challenge

The 2007 DARPA competition in which autonomous vehicles had to drive through a mock urban environment with traffic while obeying the rules of the road; it was won by the car BOSS.

## utilitarianism

The philosophical view, promoted by Bentham and Mill, that rational, welfare-maximizing decision making should be applied to all spheres of human activity.

## utility

A number expressing the desirability of a state or outcome to an agent; a rational agent's utility function internalizes its performance measure, and the agent acts so as to maximize its expected utility.

## utility function

A function that assigns a real-valued utility to outcomes (e.g., U(x_1,...,x_n)), summarizing an agent's preferences so that higher values correspond to more preferred outcomes; representation theorems establish when preferences can be captured by such a function of a simple form.

## utility independence

An extension of preference independence to lotteries: a set of attributes is utility independent of another set if preferences among lotteries over the first set do not depend on the values of the attributes in the second, allowing the utility function to be decomposed.

## Utility nodes

Nodes in a decision network (drawn as diamonds), also called value nodes, that represent the agent's utility function; they may encode utility over outcome states or the expected utility associated with each action (an action-utility function).

## utility theory

The theory used to represent and reason with an agent's preferences via utilities, which are combined with probabilities to form the general framework of rational decision making (decision theory).

# V

## vague

A property of real-world propositions whose boundaries are not clear-cut, so that borderline cases have no definite yes/no answer (for example, whether Vienna is a large city or a person is tall). Logic works cleanly in domains where every proposition has clear boundaries, but real domains often have vague boundaries; one response is fuzzy logic, which lets propositions have a degree of truth between 0 and 1.

## validation set

A held-out portion of the data, also called a development set or dev set, used to evaluate candidate models and choose the best one (for example, to tune hyperparameters), kept separate from both the training set and the final test set so that the test set stays untouched until evaluation.

## validity

A property of a sentence in logic: a sentence is valid if it is true in all models (under every interpretation). Valid sentences are also known as tautologies; they are necessarily true, and every valid sentence is logically equivalent to True.

## value

The specific setting assigned to an attribute (variable) in a factored representation of the world; more generally, a quantity such as a utility or a state's worth used by an agent in decision making.

## value alignment

The problem of ensuring that the values or objectives put into a machine are the ones we really want, so that the machine pursues our true intentions rather than a misspecified or literally-interpreted goal. Also called the value alignment problem.

## value function

Also called an ordinal utility function; a function that gives only a preference ranking over states, sufficient in deterministic settings where the actual utility numbers do not matter, only their relative order.

## value iteration

An algorithm for solving Markov decision processes, based on the Bellman equation, that repeatedly applies the Bellman update to every state's utility estimate; convergence follows from the update acting as a contraction.

## value of a statistical life

A monetary measure of how much a life is worth, used by government agencies (such as the EPA, FDA, and Department of Transportation) to weigh the costs and benefits of regulations and safety interventions.

## value of perfect information

The expected gain in utility from acquiring exact evidence about the value of some as-yet-unobserved variable before making a decision; it quantifies how much a piece of perfect information is worth.

## value symmetry

A form of symmetry in constraint satisfaction problems in which values are interchangeable, so that any permutation of the values in a solution yields another equally valid solution (e.g., permuting the color labels in map coloring); it is often handled by adding symmetry-breaking constraints.

## vanishing point

The point in the image plane at which the projections of a family of parallel lines converge. Lines sharing the same direction in space share the same vanishing point (as with railway tracks that appear to meet in the distance).

## variable

One of the components of a factored representation: states are defined by an n-dimensional vector of variables (also called attributes), each of which takes a value. Moving through a state space corresponds to changing one or more of these variables.

## variable elimination

An inference algorithm that evaluates probabilistic queries by summing out variables one at a time, representing intermediate results as factors (matrices over the values of their argument variables) and combining them with pointwise operations.

## variational approximation

An approximate-inference method that proposes a reduced version of the original problem which is simple to work with but resembles the original as closely as possible, so that inference becomes an optimization. In many cases strict upper and lower bounds can be obtained.

## variational autoencoder

A generative latent-variable model (VAE) that establishes a general connection between autoencoders and probabilistic generative models by training an encoder and decoder using variational inference.

## variational lower bound

A tractable objective that variational methods optimize in place of the intractable log-likelihood of a model, providing a lower bound on that log-likelihood.

## variational parameters

The adjustable parameters that describe the reduced problem in a variational approximation; they are tuned to minimize a distance function between the original and the reduced problem, often by solving the equations that set the derivative of that distance to zero.

## variational posterior

An introduced distribution q(z) over latent variables that variational methods use to approximate the true, intractable posterior distribution.

## VC dimension

The Vapnik-Chervonenkis dimension; a measure of the capacity of a hypothesis class that is roughly analogous to, but more general than, the ln|H| measure from PAC analysis, and that can be applied to continuous hypothesis spaces.

## VCG

See Vickrey-Clarke-Groves.

## vector

An ordered tuple of n coordinates specifying a point or direction in an n-dimensional space; for example, a state may be defined by an n-dimensional vector of variables.

## vector field histograms

A robot obstacle-avoidance technique (Borenstein and Koren, 1991) that summarizes local sensor data as a histogram of obstacle density in different directions, letting the robot steer toward directions with low obstacle density.

## vehicle interface layer

The software layer in an autonomous vehicle that mediates between high-level control decisions and the vehicle's low-level actuators (steering, throttle, brakes), translating commands and abstracting away the hardware.

## verification

The process of proving that a program or system meets its specification and behaves correctly, for example by formal proof rather than by testing on sample inputs.

## verification and validation

The engineering process (V&V) by which a system is checked to earn trust: verification confirms the system was built correctly and validation confirms it meets its intended requirements.

## version space

The set of all hypotheses that are consistent with the examples seen so far. The candidate-elimination algorithm maintains this set and shrinks it as each new example rules out inconsistent hypotheses.

## Vickrey auction

A sealed-bid second-price auction in which the highest bidder wins but pays the second-highest bid; named after William Vickrey, this design makes truthful bidding a dominant strategy.

## Vickrey-Clarke-Groves

A general mechanism design (VCG) that is both utility-maximizing (it maximizes the sum of all parties' utilities) and truth-revealing (each agent's dominant strategy is to report its true value). Each winning agent pays a tax equal to the loss its presence imposes on the other agents.

## Video games

Complex game environments (such as StarCraft II) that are partially observable, stochastic, and involve many units acting in real time within high-dimensional, near-continuous state spaces.

## virtual adversarial training

A semi-supervised technique based on consistency regularization that trains a model to assign the same label to an unlabeled example and to a slightly perturbed version of it.

## virtual counts

Prior observations encoded by the hyperparameters of a Beta (or Dirichlet) prior: the hyperparameters behave exactly as if one had started with a uniform prior and already seen that many actual outcomes of each type.

## virtual support vector machine

A support vector machine trained on extra examples produced by applying known invariance transformations to the support vectors, so that the resulting classifier respects those invariances.

## Viterbi algorithm

A dynamic-programming algorithm that computes the single most likely sequence of hidden states given a sequence of observations, commonly applied in hidden Markov models.

## VLSI layout

A search/optimization problem of positioning millions of components and connections on a chip to minimize area and circuit delays and maximize yield, split into cell layout (placing functional cells without overlap) and channel routing (routing each wire through the gaps between cells).

## vocabulary

The set of logic-level names (predicates, functions, and constants) chosen to represent the important concepts of a domain; once these choices are made, the resulting vocabulary is known as the ontology of the domain.

## Voronoi graph

The set of edges and vertices of the regions of a Voronoi diagram, whose points are equidistant from two or more obstacles. Used in robot motion planning to produce paths that stay as far from obstacles as possible rather than hugging them.

# W

## weak AI

The idea, in a distinction introduced by John Searle (1980), that machines could act as if they were intelligent. It is contrasted with strong AI, the assertion that machines that do so are actually consciously thinking rather than merely simulating thinking.

## weak learning

A setting in which a base learner need only do slightly better than random guessing. Boosting can combine such weak learners into an accurate ensemble, overcoming any amount of bias in the base model as long as each hypothesis is better than random.

## weight

A real-valued coefficient in a model, such as the coefficients of a linear function; the model's output is changed by adjusting the relative weight of each term, and learning largely consists of tuning these weights.

## weight decay

A regularization technique for neural networks in which large weight values are penalized during training, encouraging the model to keep weights small and thereby reducing overfitting.

## weight space

The space defined by all possible settings of a model's weights. Learning that minimizes a loss can be pictured as searching this space (for example by gradient descent) for a point of (local) minimum loss.

## weighted A* search

A variant of A* whose evaluation function is f(n) = g(n) + W x h(n) with W greater than 1, weighting the heuristic more heavily. This focuses the frontier toward the goal, finding a solution faster at the cost of possible suboptimality.

## weighted linear function

An evaluation function expressed as a weighted sum of features, w_1 f_1(s) + w_2 f_2(s) + ... + w_n f_n(s), where each feature f_i of a state is scaled by a learned weight w_i.

## weighted model counting

A form of SAT solving to which Bayes net inference can be reduced; it counts satisfying assignments of a Boolean formula while weighting each assignment, yielding the probability mass associated with a query.

## weighted training set

A training set in which each example has an associated weight indicating how much it should count during training (a weight of 3 acts like three copies of the example). Boosting increases the weights of misclassified examples so later hypotheses focus on them.

## well-calibrated

A fairness/probability property whereby all individuals assigned the same score by a model have approximately the same true probability of being a positive instance, regardless of protected attributes such as race.

## wicked problems

Problems characterized by underspecified, contradictory, and continually changing requirements, making them resistant to clean formulation and definitive solution.

## wide content

In philosophy of mind, the view that the content of a mental state depends partly on the agent's external environment and history, not only on the agent's internal brain state. Contrasted with narrow content.

## Widrow-Hoff rule

A gradient-based online least-squares update rule, also called the delta rule, that adjusts each parameter in proportion to the error between the predicted and observed value, moving parameters in the direction that decreases the squared error.

## winner's curse

A phenomenon in competitive bidding auctions in which the winning bidder tends to have overestimated the true value of the item; closely related to the optimizer's curse and post-decision disappointment.

## Winnow algorithm

An online learning algorithm for linear threshold functions that updates weights multiplicatively rather than additively, performing well when only a few of many features are relevant.

## word embedding

A learned low-dimensional, dense vector representation of a word, embedded within a higher-dimensional space so that words with similar meaning or usage lie near one another.

## word2vec

An open-source package for learning word embeddings; it includes the skip-gram negative sampling (SGNS) approach for producing dense word vectors from text.

## WordNet

An open-source, hand-curated, machine-readable dictionary that organizes words and their senses, widely used as a lexical resource in natural language processing applications.

## workspace representation

A representation of a robot's environment in terms of the physical (Cartesian) space, called the workspace, in which the robot body and obstacles reside, as opposed to configuration space.

## wrapper

A program that surrounds a data source and converts its inputs and outputs into a standard form; in information extraction, a wrapper exploits the regular formatting of a source such as a web page to pull out structured attribute values.

## wumpus world

A grid-based cave environment used in Chapter 7 as a running example for knowledge-based agents. The agent must find gold while avoiding pits and the wumpus, reasoning from local percepts such as breezes and stenches to infer where the hidden hazards lie.

# Z

## zero-shot

Performing a task on inputs never seen during training; for example, a multilingual translation model that has learned to encode and decode many languages can translate between language pairs that were absent from its training data.

## zero-sum games

Games in which what is good for one player is just as bad for the other, so there is no "win-win" outcome; the players' total payoffs sum to a constant (traditionally called zero). Chess and similar deterministic, two-player, turn-taking, perfect-information games are the classic examples.
