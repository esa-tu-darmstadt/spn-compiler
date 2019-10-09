//
// Created by ls on 10/9/19.
//

#include <cassert>
#include "BinaryTreeTransform.h"

NodeReference BinaryTreeTransform::binarizeTree(const NodeReference& rootNode) {
    rootNode->accept(*this, rootNode);
    return updated_nodes.at(rootNode->id());
}

void BinaryTreeTransform::visitHistogram(Histogram &n, arg_t arg) {
    updated_nodes.emplace(n.id(), std::static_pointer_cast<Histogram>(arg));
}

void BinaryTreeTransform::visitProduct(Product &n, arg_t arg) {
    std::vector<NodeReference> newChildren;
    for(auto& c : *n.multiplicands()){
        c->accept(*this, c);
        newChildren.push_back(updated_nodes.at(c->id()));
    }
    updated_nodes.emplace(n.id(), splitChildren<Product>(newChildren, n.id()));
}

void BinaryTreeTransform::visitSum(Sum &n, arg_t arg) {
    std::vector<NodeReference> newChildren;
    for(auto& c : *n.addends()){
        c->accept(*this, c);
        newChildren.push_back(updated_nodes.at(c->id()));
    }
    updated_nodes.emplace(n.id(), splitChildren<Sum>(newChildren, n.id()));
}

template<class T>
NodeReference BinaryTreeTransform::splitChildren(const std::vector<NodeReference> &children,
        const std::string& prefix) const {
    if(children.size()==1){
        return children[0];
    }
    else if(children.size()==2){
        return std::make_shared<T>(prefix, children);
    }
    else if(children.size()>2){
        std::size_t const half = children.size() / 2;
        std::vector<NodeReference> split_left(children.begin(), children.begin()+half);
        std::vector<NodeReference> split_right(children.begin()+half, children.end());
        auto leftChild = splitChildren<T>(split_left, prefix+"l");
        auto rightChild = splitChildren<T>(split_right, prefix+"r");
        return std::make_shared<T>(prefix, std::vector<NodeReference>{leftChild, rightChild});
    }
    assert(false);
}

void BinaryTreeTransform::visitWeightedSum(WeightedSum &n, arg_t arg) {
    std::vector<WeightedAddend> newChildren;
    for(auto& c : *n.addends()){
        c.addend->accept(*this, c.addend);
        newChildren.push_back(WeightedAddend{updated_nodes.at(c.addend->id()), c.weight});
    }
    updated_nodes.emplace(n.id(), splitWeightedChildren(newChildren, n.id()));
}

NodeReference BinaryTreeTransform::splitWeightedChildren(const std::vector<WeightedAddend> &children,
                                                         const std::string& prefix) const {
    if(children.size()==2){
        return std::make_shared<WeightedSum>(prefix, children);
    }
    else if(children.size()==3){
        auto rightChild = WeightedAddend{splitWeightedChildren(std::vector<WeightedAddend>{children[1],
                                                                                           children[2]}, prefix+"r"), 1.0};
        return std::make_shared<WeightedSum>(prefix, std::vector<WeightedAddend>{children[0], rightChild});
    }
    else if(children.size()>3){
        std::size_t const half = children.size() / 2;
        std::vector<WeightedAddend> split_left(children.begin(), children.begin()+half);
        std::vector<WeightedAddend> split_right(children.begin()+half, children.end());
        auto leftChild = splitWeightedChildren(split_left, prefix+"l");
        auto rightChild = splitWeightedChildren(split_right, prefix+"r");
        return std::make_shared<Sum>(prefix, std::vector<NodeReference>{leftChild, rightChild});
    }
    assert(false);
}
