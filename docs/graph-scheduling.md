# Graph scheduling

## Fundamentals
* Scheduling an SPN is a static task scheduling problem: Assigning tasks of an application to (suitable) processors.
*  Application presented by a DAG; nodes represent tasks and edges represent intertask data dependencies
  * Node label: Computation cost (expected computation time)
  * Node edge: Intertask communication cost (expected communication time)
* Usually, intertask communication time is assumed to be zero for tasks scheduled to the same processor
* NP-hard in the general case [A Guide to the Theory of NP-Completeness.]
  * NP-completeness proven even for restricted case like all jobs require one time unit [NP-complete scheduling problems]
* Categories of algorithms:
  * List scheduling:
    1. Make an ordered list of tasks by assigning some priority to each task
    2. Loop: Select a processor for the highest priority task that is not scheduled yet
  * Clustering: 
    1. Cluster DAG into clusters. All tasks on the same cluster execute on the same processor.
    2. Order the tasks in each cluster
    * DAG with a given clustering but without an ordering is called a *clustered DAG*
    * A *scheduled DAG* adds pseudo edges representing the execution ordering to the clustered DAG
      * Task starting times are not explicitly given in a scheduled DAG but can be computed by traversing it
    * A communication edge weight becomes zero when the source and destination tasks are in the same cluster
    * Critical path: Longest path inclduing communication and computation costs
    * **Dominant sequence**: Critical path of the **scheduled** DAG (not the clustered DAG!)
    * Idea: Start with unit clusters. Then R
    repeatitaly zero edges by moving tasks from one cluster to another.
    * Optionally: Merge clusters to reduce number of processors [Sarkar]
  * Duplication based:
  * Guided random search:
  * Genetic algorithm:
    * Encoding, initial popluation, evaluating, selection, crossover and mutation. See dedicated section below.


    
* Algorithms that work with a **restricted** number of processors:
  * 
* Algorithms that consider **heterogenous** processors: Tasks require a different computation time on different processors
  * Topcuoglu2002 (3034): HEFT/CPOP "Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing" --> first heterogenous algorithm!
  * Arabnejad2014 (435):  "List Scheduling Algorithm for Heterogeneous Systems by an Optimistic Cost Table"
  * Daoud2008 (232): "A high performance algorithm for static task scheduling in heterogeneous distributed computing systems"
* Algorithms that consider **constrained interconnections**:
  * Sih1933 (942): "A Compile-Time Scheduling Heuristic for Interconnection-Constrained Heterogeneous Processor Architectures"
  * El-Rewini1990 (509): "Scheduling Parallel Program Tasks onto Arbitrary Target Machines" - TODO: Are you sure?
* Algorithms that use **task duplication** to reduce communication overhead: 
  * "Duplication heuristics are more effective, in general, for fine grain task graphs and for networks with high communication latencies" (from Bansal2003)
  * Ahmed1998 (311): "On Exploiting Task Duplication in Parallel Program Scheduling"
  * Park1997: "DFRN: a new approach for duplication based scheduling for distributed memory multiprocessor systems"
  * Bansal2003 (145): An Improved Duplication Strategy for Scheduling Precedence Constrained Graphs in Multiprocessor Systems
* Algorithms that **reduce** the number of required processors for a given schedule:
  * Dozdag2009: "Compaction of Schedules and a Two-Stage Approach for Duplication-Based DAG Scheduling"
    * "To the best of our knowledge, this is the first algorithm to address this highly unexplored aspect of DAG scheduling"
* Algorithms that consider large **communication delays**:
  * Giroudeau does most research here. Eg. Giroudeau2005: Theory and definition: "Complexity and approximation for precedence constrained scheduling problems with large communication delays"
* Algorithms for **BSP**:
  * Boeres1998 (10): Static Scheduling Using Task Replication for LogP and BSP Models
  * Papp2023 (0): DAG Scheduling in the BSP Model
  * Fujimoto2003 (1): On Approximation of the Bulk Synchronous Task Scheduling Problem

## Genetic algorithms
Source: Wang1997
1. **Encoding**: All possible solutions encoded as a set of strings (chromosome). Each chromosome represents one solution to the problem. A set of chromosome are reffered to as a population
2. **Initial population**: Algorithm starts with an initial population, choosen random or by hand.
3. **Evaluate** the quality of each chromosome (here: completion time)
4. **Selection**: Eliminate or duplicate chromosomes based on its relative quality
5. **Crossover**: Some pairs of chromosomes are selected and some components are exchanged to form two valid chromosomes
6. **Mutation**: Some chromosomes are mutated into another valid chromosomes.
* Wang1997 (439): Scheduling for heterogenous computing!
  * Chromosome represents the subtask-to-machine assignments (matching) and the execution ordering of the subtasks assigned to the same machine. Tuple of:
    * Matching string: Mapping of task to machine (task -> machine)
    * Scheduling string: Topological sort of the DAG
  * Scheduling of global data item transfers and the relative ordering of subtasks assigned to different machines are determined in the evaluation step.
  * **Initial population**: Random assignements of subtasks to processors. Topological sort of scheduling strings, then mutated trough mutation operator. Additionally, some chromosomes are calculated by nonevolutioniary baseline heuristrics.
  * **Selection**: fittest (here: fastest) chromosome has highest propability to be duplicated
    * rank-based roulette wheel selection scheme, i.e., propability depends on rank
    * value-based roulette wheel selection scheme, i.e., propability dependsa on value
    * Elitism: Overall best chromosomes are stored and compared with
  * **Crossover operator**: 
    * Scheduling string: Randomly choose some pairs of scheduling strings. For each pair, randomly generate a cut-off point, which divides the scheduling strings into top and bottom parts. Then, subtasks in each bottom part are reordered: The new ordering of subtasks in one bottom part is the relative positions of these subtasks in the other oridinal scheduling string in the pair.
    * Matching string: Randome choose some pairs of matching strings. Randomly choose cutoff points and exachange bottom parts.
  * **Mutation operator**:
    * Scheduling string: Randomly choose scheduling strings. For each string, choose a victim subtask. Determine valid range into which the subtask can be placed without violating data dependencies. Move subtask to random position in valid range.
    * Matching string: Randomly choose matching strings. On each matching strings, randomly change a subtask->machine mapping to another machine.

## Comparison to HLS schedulers
* HLS schedulers create an execution schedule for a DCG. We already got this schedule!
* Our task is to "cluster" a DAG and assign clusters to processors.
* HLS scheduling usually schedules loops, which form directed cyclic graphs. This is opposed to SPNs, which are DAGs. Most (all?) static task scheduling algorithms are made specially for DAGs.
* Each processor is considered to have its own set of registers and its own memory. Transfers are only possible at the end of a job. 

## Algorithms
Basics about DAG scheduling: "On the Granularity and Clustering of Directed Acyclic Task Graphs" (1933)
Comparison Papers:
  * Kwok1999 (1405): "Static scheduling algorithms for allocating directed task graphs to multiprocessors"
    * Algorithms for optimally schedulling a DAG in polynomial time are known for three cases:
      * Uniform node-weighted free-tree to an arbitrary number of processors [Hu1961 in $O(n)$]
      * Uniform node-weighted DAG to two processors [Coffmann1972 in $O(n^2)$, Sethi1976 in almost $O(n)$]
        * Hu and Coffmann use node-labeling methods that produce optimal scheduling lists
      * Uniform node-weighted interval-ordered DAG to an arbitrary number of processors [Papadimitriou and Yannakakis1979 in $O(n)$]
        * A DAG is called interval-ordered if every two precedence-related nodes can be mapped to two nonoverlapping intervals on the real number line [Fishburn 1985].
        * **SPNs are interval-ordered DAGs!**
    * Nomenclature: Parallel Program Scheduling -> Scheduling and Mapping -> Static Scheduling 
  * Ahmad1996 (112): "Analysis, evaluation, and comparison of algorithms for scheduling task graphs on parallel processors"
  * Maurya2018 (44): "On benchmarking task scheduling algorithms for **heterogeneous** computing systems"
  * Vucha(2014) (4): "A Case Study: Task Scheduling Methodologies for High Speed Computing Systems"

Table of algorithms:
| Short | Long | Author | Year | Citations | Type | Heterogenous | Constrained Interconnections | Duplication | Reduce #processors | Restricted #processors | BSP | Time complexity | Paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DSC** | Dominant Sequence Clustering | Yang | 1994 | 717 | Clustering | No | No | No | No | No | No | $O((v+e)\log v)$ | [Yang1994] |
| **DCP** | Dynamic Critical-Path Scheduling | Kwok | 1996 | 843 | List | No | No | No| No | No | No | ? | [Kwok1996] |
| **HEFT** | Heterogenous Earliest Finish Time | Topcuoglu | 2002 | 3025 | List | Yes | No| No | No | No | No | ? | [Topcuoglu2002] |
| **CPOP** | Critical-Path-on-a-Processor | Topcuoglu | 2002 | 3025 | List | Yes | No | No| No | No | No | ? | [Topcuoglu2002] |
|  | Scheduling interval-ordered tasks | Papadoimitriou | 1979 | 240 | List | No | No | No| No | Yes? | No | $O(v+p)$ | [Papadimitriou1979] |
| PLW | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| TCSD | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| CPFD | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? ? | | $O(v^4)$? | ? |
| SC | ? | Bozdag | 2009 | 67 | Reducing | -- | -- | Yes | Yes | No | No | $O(v^3)$ | [Bozdag2009] |

## References
[Yang1994]: DSC: Scheduling Parallel Tasks on an Unbounded Number of Processors
[Kwok1996]: DCP: Dynamic Critical-Path Scheduling: An Effective Technique for Allocating Task Graphs to Multiprocessors
[Topcuoglu2002]: HEFT/CPOP: Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing
[Bozdag2009]: Compaction of Schedules and a Two-Stage Approach for Duplication-Based DAG Scheduling
[Papadimitriou1979]: Scheduling Interval-Ordered Tasks


* (207) Palis1996: unnmaed - From "Task Clustering and Scheduling for Distributed Memory Parallel Architectures"
* (69) Ahmad1999: unnamed - From "On Parallelizing the Multiprocessor Scheduling Problem"
  * Focus on parallel algorithm
* CASS-II: From "An Efficient Task Clustering Heuristic for Scheduling DAGs on Multiprocessors" (2007)
  * "For task clustering with no duplication, the DSC algorithm of Gerasoulis and Yang is empirically the best known algorithm to date in terms of both speed and solution quality."
* Hakem2005: Dynamic critical path scheduling (DCPS): From "Dynamic critical path scheduling parallel programs onto multiprocessors"
  * "DCPS has a time complexity of O(e+vlogv), as opposed to DSC algorithm O((e+v) logv) which is the best known algorithm."
* Arabnejad2014: Predict Earliest Finish Time (PEFT): From "List Scheduling Algorithm for Heterogeneous Systems by an Optimistic Cost Table"

* (613) Hwang1989: Earliest Task First (ETF): From "Scheduling Precedence Graphs in Systems with Interprocessor Communication Times" https://epubs.siam.org/doi/10.1137/0218016
* Earliest Ready Task (ERT): From "Multiprocessor scheduling with interprocessor communication delays" (1988)

Papers:
* An Efficient Task Clustering Heuristic for Scheduling DAGs on Multiprocessors

Algorithms for clound environments:
* Multi Objective Task Scheduling Using Modified Ant Colony Optimization in Cloud Computing

## SPNs and interval-orderedness
* A DAG is called interval-ordered if every two precedence-related nodes can be mapped to two nonoverlapping intervals on the real number line [Fishburn 1985].
* SPNs are interval-ordered DAGs (I think...)
* Very good mathematical definition and parallel algorithm:" Scheduling Interval Ordered Tasks in Parallel"
* Interval-ordered DAGs can be scheduled in $O(n)$ time. Papadimitriou and Yannakakis1979: "Scheduling Interval-Ordered Tasks"

## Scheduling for BSP
* The DSC algorithm uses the *macro-dataflow model*, which asumes that a task receives all input before starting execution in parallel, exeutes to completion without interrutption, and immediately sends the output to all successor tasks. (source: Yang1994)
* This is not the case for BSP, in which communication can only happen at the end of a superstep.
* We are only searching for a clustering. The scheduling within the cluster doesnt matter because data is only transferred at the end of a superstep. 
* Research:
  * Calinescu1996 (7): Bulk Synchronous Parallel Scheduling of Uniform Dags
    * Partition uniform DAG resulting from a loop nest into sub-DAGs called tiles
    * The DAG consisting of the tiles is called tile DAG. It is scheduled with the wavefront-method, resulting in hyperplanes = supersteps = compute sets
      * The hyperplane method of Lamport [The Parallel Execution of DO Loops] is used to schedule the tile dag for parallel execution. This widely applied method (also known as wave front scheduling ["Compiler transformations for high-performance computing", "Optimizing Supercompilers for Supercomputers."]) consists of successively scheduling for concurrent execution the intersections of the iteration space ((O..p1/(K-l) - 1) K in our case) with a family of (K - 1)-dimensional parallel hyperplanes. The number of hyperplanes in this family gives the number of supersteps required to accomplish the computation (i.e., the schedule length).
    * Num of compute sets = *schedule length*
  * MSA (multi-stage scheduling approach)" from: Boeres2006 (6): Static scheduling using task replication for LogP and BSP models"
    * No electronic version available, but ULB Stadtmitte has a copy
  * Scheduling of tasks for BSP without precedence contraints (no DAG!): ~~Han2020: Scheduling Placement-Sensitive BSP Jobs with Inaccurate Execution Time Estimation~~


