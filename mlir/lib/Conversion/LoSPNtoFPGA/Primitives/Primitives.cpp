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

namespace mlir::spn::fpga::primitives::operators {

std::shared_ptr<BinaryExpression> operator&(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_AND);
}

std::shared_ptr<BinaryExpression> operator|(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_OR);
}

std::shared_ptr<BinaryExpression> operator+(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_ADD);
}

std::shared_ptr<BinaryExpression> operator-(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_SUB);
}

std::shared_ptr<BinaryExpression> operator>(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_GT);
}

std::shared_ptr<BinaryExpression> operator>=(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_GEQ);
}

std::shared_ptr<BinaryExpression> operator<(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_LT);
}

std::shared_ptr<BinaryExpression> operator<=(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_LEQ);
}

std::shared_ptr<BinaryExpression> operator==(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_EQ);
}

std::shared_ptr<BinaryExpression> operator!=(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b) {
  return std::make_unique<BinaryExpression>(a, b, Expression::Operation::OP_NEQ);
}

}

namespace mlir::spn::fpga::primitives {
template <class T>
FModuleOp Module<T>::modOp;


}