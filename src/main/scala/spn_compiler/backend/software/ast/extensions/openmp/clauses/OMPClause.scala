package spn_compiler.backend.software.ast.extensions.openmp.clauses

trait OMPClause

trait OMPParallelClause extends OMPClause

trait OMPWorksharingLoopClause extends OMPClause

trait OMPParallelWorksharingLoopClause extends OMPParallelClause with OMPWorksharingLoopClause

