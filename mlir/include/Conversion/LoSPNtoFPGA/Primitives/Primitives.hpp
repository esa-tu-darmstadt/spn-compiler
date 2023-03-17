#pragma once

#include <memory>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/HWArith/HWArithTypes.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"

namespace mlir::spn::fpga::primitives {

using namespace ::mlir;
using namespace ::circt::firrtl;

class PrimitiveBuilder {
public:
  MLIRContext *context;
  OpBuilder builder;
  Value clk;
public:
  PrimitiveBuilder(MLIRContext *context, Value clk):
    context(context),
    builder(context),
    clk(clk) {}

  Value getClock() {
    return clk;
  }
};

inline std::unique_ptr<PrimitiveBuilder> prim;

PrimitiveBuilder *getPrimitiveBuilder();
void initPrimitiveBuilder(MLIRContext *ctxt, Value clk);

Value constant(int64_t value, uint32_t bitWidth);

inline Type UInt(uint32_t bitWidth) {
  return IntType::get(prim->context, false, bitWidth);
}

inline Type SInt(uint32_t bitWidth) {
  return IntType::get(prim->context, true, bitWidth);
}

// can be used to build any kind of expression

/*
  Value a, b, c, d;
  ...
  Value e = (  (a & b) | c + d  ).build(primBuilder);
 */

class Expression {
public:
  enum Operation {
    OP_AND, OP_OR, OP_NEG, OP_ADD, OP_SUB, OP_GT, OP_GEQ,
    OP_LT, OP_LEQ, OP_EQ, OP_NEQ,


  };

  virtual ~Expression() {}
  virtual Value build() const = 0;
};

class ValueExpression : public Expression {
  Value value;
public:
  // lift value to expression
  ValueExpression(Value value): value(value) {}

  Value build() const override {
    return value;
  }
};

inline std::shared_ptr<Expression> lift(Value val) {
  return std::make_shared<ValueExpression>(val);
}

class UnaryExpression : public Expression {
  std::shared_ptr<Expression> operand;
  Operation operation;
public:
  UnaryExpression(UnaryExpression&) = delete;
  UnaryExpression(const UnaryExpression&) = delete;
  UnaryExpression(UnaryExpression&&) = delete;

  UnaryExpression(std::shared_ptr<Expression> operand, Operation operation):
    operand(operand),
    operation(operation) {}

  Value build() const override {
    Value input = operand->build();
    Value output;

    switch (operation) {
      case OP_NEG:
        output = prim->builder.create<NotPrimOp>(
          prim->builder.getUnknownLoc(),
          input
        ).getResult();
        break;
      default:
        assert(false && "invalid unary operation");
    }

    assert(output);
    return output;
  }
};

class BinaryExpression : public Expression {
  std::shared_ptr<Expression> lhs;
  std::shared_ptr<Expression> rhs;
  Operation operation;
public:
  //BinaryExpression(BinaryExpression&) = delete;
  //BinaryExpression(const BinaryExpression&) = delete;
  //BinaryExpression(BinaryExpression&&) = delete;

  BinaryExpression(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, Operation operation):
    lhs(lhs),
    rhs(rhs),
    operation(operation) {}

  Value build() const override {
    Value leftInput = lhs->build();
    Value rightInput = rhs->build();
    Location loc = prim->builder.getUnknownLoc();
    Value output;

    switch (operation) {
      case OP_AND:
        output = prim->builder.create<AndPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_OR:
        output = prim->builder.create<OrPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_ADD:
        output = prim->builder.create<AddPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_SUB:
        output = prim->builder.create<SubPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_GT:
        output = prim->builder.create<GTPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_GEQ:
        output = prim->builder.create<GEQPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_LT:
        output = prim->builder.create<LTPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_LEQ:
        output = prim->builder.create<LEQPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_EQ:
        output = prim->builder.create<EQPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_NEQ:
        output = prim->builder.create<NEQPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      default:
        assert(false && "invalid binary operation");
    }

    assert(output);
    return output;
  }
};

namespace operators {

std::shared_ptr<BinaryExpression> operator&(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator|(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator+(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator-(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator>(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator>=(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator<(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator<=(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator==(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);
std::shared_ptr<BinaryExpression> operator!=(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b);

}

class BitsExpression : public Expression {
  std::shared_ptr<Expression> operand;
  uint32_t hi, lo;
public:
  BitsExpression(std::shared_ptr<Expression> operand, uint32_t hi, uint32_t lo):
    operand(operand), hi(hi), lo(lo) {}

  Value build() const override {
    return prim->builder.create<BitsPrimOp>(
      prim->builder.getUnknownLoc(),
      operand->build(),
      prim->builder.getI32IntegerAttr(hi),
      prim->builder.getI32IntegerAttr(lo)
    ).getResult();
  }
};

inline std::shared_ptr<BitsExpression> bits(std::shared_ptr<Expression> of, uint32_t hi, uint32_t lo) {
  return std::make_shared<BitsExpression>(of, hi, lo);
}



class Statement {
};

class Reg : public Statement {
  RegOp regOp;
public:
  Reg(Type type) {
    // TODO
    Value clk = constant(1, 1);

    regOp = prim->builder.create<RegOp>(
      prim->builder.getUnknownLoc(),
      type,
      clk //prim->getClock()
    );
  }

  operator Value() {
    return regOp.getResult();
  }

  operator std::shared_ptr<Expression>() {
    return lift(*this);
  }

  void operator<<(std::shared_ptr<Expression> what) {
    Value input = what->build();
    prim->builder.create<ConnectOp>(
      prim->builder.getUnknownLoc(),
      regOp,
      input
    );
  }
};

}