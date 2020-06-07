//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_GRAPHIRNODE_H
#define SPNC_GRAPHIRNODE_H

#include <string>
#include <utility>
#include <vector>
#include <memory>

namespace spnc {
  /*
   * Forward declaration of types related to the visitor to break circular dependency.
   */
  class Visitor;
  typedef std::shared_ptr<void> arg_t;

  ///
  /// Base class of all nodes of the graph-based IR.
  ///
  class GraphIRNode {

  public:
    /// Constructor.
    /// \param id Unique ID of the node.
    explicit GraphIRNode(std::string id);

    /// Simple string representation of the node.
    /// \return Simplified string representation.
    virtual std::string dump() const;

    /// Print graph-IR node to output stream.
    /// \param os Output stream.
    /// \param node Node to print.
    /// \return Reference to the output stream.
    friend std::ostream& operator<<(std::ostream& os, const GraphIRNode& node);

    /// Get the unique ID.
    /// \return Unique ID as string.
    std::string id() const;

    /// Part of the visitor pattern, accept the visitor.
    /// \param visitor Visitor.
    /// \param arg Pass-through argument.
    virtual void accept(Visitor& visitor, arg_t arg) = 0;

    virtual ~GraphIRNode() {}

  private:
    std::string _id;

  };

  using NodeReference = GraphIRNode*;

  ///
  /// SPN feature (input variable).
  ///
  class InputVar : public GraphIRNode {

  public:
    /// Constructor.
    /// \param id Unique string ID.
    /// \param index Index of the feature, corresponding to the index of the feature in the input vector.
    InputVar(std::string id, int index);

    /// Index of the feature.
    /// \return Index.
    int index() const;

    std::string dump() const override;

    void accept(Visitor& visitor, arg_t arg) override;

  private:
    int _index;
  };

  ///
  /// Histogram bucket, consisting of an inclusive lower bound,
  /// an exclusive upper bound and the associated value.
  struct HistogramBucket {
    ///
    /// Inclusive lower bound.
    int lowerBound;
    ///
    /// Exclusive upper bound.
    int upperBound;
    ///
    /// Associated probability value.
    double value;
  };

  ///
  /// Histogram as SPN leaf node.
  ///
  class Histogram : public GraphIRNode {
  public:
    /// Constructor.
    /// \param id Unique string ID.
    /// \param indexVar Associated input feature.
    /// \param buckets List of HistogramBucket contained in this histogram.
    Histogram(std::string id, InputVar* indexVar, const std::vector<HistogramBucket>& buckets);

    /// Get the associated input feature.
    /// \return Input variable.
    InputVar& indexVar() const;

    /// Get the list of buckets associated with this histogram.
    /// \return Bucket list.
    const std::vector<HistogramBucket>& buckets() const;

    void accept(Visitor& visitor, arg_t arg) override;

  private:
    InputVar* _indexVar;

    std::vector<HistogramBucket> _buckets;
  };

  class Gauss : public GraphIRNode {
  public:
    Gauss(std::string id, InputVar* indexVar, double mean,
          double stddev);

    InputVar& indexVar() const;

    double mean() const;

    double stddev() const;

    void accept(Visitor &visitor, arg_t arg) override;

  private:
    InputVar* _indexVar;

    double _mean;
    double _stddev;
  };
  ///
  /// Addend for a WeightedSum, consisting of the child node and the corresponding weight.
  ///
  struct WeightedAddend {
    ///
    /// Child node.
    NodeReference addend;
    ///
    /// Associated weight.
    double weight;
  };

  ///
  /// N-ary weighted sum, where a weight is associated with each child node.
  ///
  class WeightedSum : public GraphIRNode {

  public:
    /// Constructor.
    /// \param id Unique string ID.
    /// \param addends List of WeightedAddend.
    WeightedSum(std::string id, const std::vector<WeightedAddend>& addends);

    /// Get the list of addends.
    /// \return Weighted addends.
    const std::vector<WeightedAddend>& addends() const;

    void setAddends(std::vector<WeightedAddend> newAddends);
    void accept(Visitor& visitor, arg_t arg) override;

  private:
    std::vector<WeightedAddend> _addends;
  };

  ///
  /// Simple (non-weighted), n-ary sum.
  class Sum : public GraphIRNode {
  public:
    /// Constructor.
    /// \param id Unique string ID.
    /// \param addends List of addends.
    Sum(std::string id, const std::vector<NodeReference>& addends);

    /// Get the list of addends.
    /// \return Child nodes.
    const std::vector<NodeReference>& addends() const;

    void setAddends(std::vector<NodeReference> newAddends);
    void accept(Visitor& visitor, arg_t arg) override;

  private:
    std::vector<NodeReference> _addends;
  };

  ///
  /// N-ary product.
  class Product : public GraphIRNode {
  public:
    /// Constructor.
    /// \param id Unique string ID.
    /// \param multiplicands List of multiplicands.
    Product(std::string id, const std::vector<NodeReference>& multiplicands);

    /// Get the list of multiplicands.
    /// \return Child nodes.
    const std::vector<NodeReference>& multiplicands() const;

    void setMultiplicands(std::vector<NodeReference> newMultiplicands);
    void accept(Visitor& visitor, arg_t arg) override;

  private:
    std::vector<NodeReference> _multiplicands;
  };

}

#endif //SPNC_GRAPHIRNODE_H
