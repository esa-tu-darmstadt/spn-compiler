#include "LoSPNtoFPGA/Primitives/Primitives.hpp"

namespace mlir::spn::fpga::primitives {

PrimitiveBuilder *getPrimitiveBuilder() {
  return prim.get();
}

void initPrimitiveBuilder(MLIRContext *ctxt, Value clk) {
  prim = std::make_unique<PrimitiveBuilder>(ctxt, clk);
}

Value constant(int64_t value, uint32_t bitWidth) {
  return prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    IntType::get(prim->context, false, bitWidth),
    ::llvm::APInt(bitWidth, value)
  ).getResult();
}

}

namespace mlir::spn::fpga::primitives {

ExpressionWrapper ExpressionWrapper::operator&(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator|(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator+(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator-(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator>(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator>=(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator<(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator<=(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator==(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator!=(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator()(const std::string& fieldName) const {
  return ExpressionWrapper::make<FieldExpression>(*this, fieldName);
}

ExpressionWrapper ExpressionWrapper::operator()(size_t hi, size_t lo) const {
  return ExpressionWrapper::make<BitsExpression>(*this, hi, lo);
}

ExpressionWrapper ExpressionWrapper::operator()(size_t bitIndex) const {

}

}