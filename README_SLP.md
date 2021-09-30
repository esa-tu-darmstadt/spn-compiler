# SLP-Vectorization #

The SLP directory contains 11 files, each dealing with a different SLP topic.

Please note that a _superword_ in this project describes an SLP vector containing the elements. This term was chosen because a _vector_ is slightly overloaded with meanings in C++. 

* Analysis.h
  * Was used for analyzing topological mixing in vectors (i.e. how many vectors contained elements with different topological depths)
  * Deprecated, can be deleted sometime in the future.

* CostModel.h
  * Contains the cost model, which assigns cost to scalar operations, superwords and entire patterns using a visitor pattern.

* GraphConversion.h
  * A very important file that contains the ConversionManager class. The conversion manager keeps track of created vector operations, extractions and maintains a ConversionState. The conversion state is responsible for remembering which scalar/superword values have been computed already. The conversion manager is also responsible for gracefully resetting the function state in case an SLP graph is not deemed profitable. 

* PatternVisitors.h
  * Contains the visitor template and the LeafPatternVisitor, which can determine the scalar values that need to be computed for every leaf pattern (e.h. a BroadcastInsertPattern needs a scalar broadcast value and scalar insert values).

* ScoreModel.h
  * Contains the Look-Ahead-Score score model from the original Look-Ahead SLP publication. Also contains the XOR chain model.

* Seeding.h
  * Contains the classes used for top-down and bottom-up seeding.

* SLPGraph.h
  * Contains the superword logic and the logic for actual SLP graphs (nodes and multinodes). Note: there is no explicit SLPGraphEdge class or something similar.

* SLPGraphBuilder.h
  * Contains a graph builder that constructs SLP graphs as described in Porpodas et al. [[1]](https://dl.acm.org/doi/10.1145/3168807).

* SLPPatternMatch.h
  * Responsible for selecting the best patterns based on the cost model and the current conversion state

* SLPVectorizationPatterns.h
  * The individual patterns that can be applied to superwords and their match and rewrite logic. They were designed in a somewhat similar fashion compared to MLIR's pattern rewrite framework

* Util.h
  * Some utility functions, such as _vectorizable(...)_ or _commutative(...)_.

### Known Issues ###
* Seeding: Bottom-up seeding currently does not work properly. As it hasn't been used in a while, it wasn't kept in a state consistent with the rest of the framework
* ShufflePattern: With shuffle patterns enabled, the output of the kernels sometimes does not match the expected output. This might be due to the reordering changing semantics and the shuffle pattern accessing elements with changed semantics by accident.
* The SPN compiler options are replicated inside the util class. This is a little bit annoying.

References
-----
[[1] Vasileios Porpodas, Rodrigo C. O. Rocha, and Luís F. W. Góes. 2018. Look-ahead SLP: auto-vectorization in the presence of commutative operations. In Proceedings of the 2018 International Symposium on Code Generation and Optimization (CGO 2018). Association for Computing Machinery, New York, NY, USA, 163–174.](https://dl.acm.org/doi/10.1145/3168807)