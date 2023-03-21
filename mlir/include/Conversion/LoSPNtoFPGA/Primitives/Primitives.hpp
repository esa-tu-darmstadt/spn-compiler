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
//using namespace ::circt::hw;

class PrimitiveBuilder {
public:
  MLIRContext *context;
  OpBuilder builder;

  ModuleOp root;
  CircuitOp circuitOp;
  Block *lastInsertionBlock;

  Value clk;
public:
  PrimitiveBuilder(MLIRContext *context, Value clk):
    context(context),
    builder(context),
    clk(clk) {

    root = builder.create<ModuleOp>(
      builder.getUnknownLoc()
    );

    builder.setInsertionPointToStart(
      &root.getBodyRegion().front()
    );

    circuitOp = builder.create<CircuitOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("MyCircuit")
    );

    builder = circuitOp.getBodyBuilder();
  }

  Value getClock() {
    return clk;
  }

  void dump() {
    assert(succeeded(root.verify()));
    root.dump();
  }

  void beginModule() {
    lastInsertionBlock = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
  }

  void endModule() {
    builder.setInsertionPointToEnd(lastInsertionBlock);
  }
};

inline std::unique_ptr<PrimitiveBuilder> prim;

PrimitiveBuilder *getPrimitiveBuilder();
void initPrimitiveBuilder(MLIRContext *ctxt, Value clk);

Value constant(int64_t value, uint32_t bitWidth);

inline circt::firrtl::IntType UInt(uint32_t bitWidth) {
  return circt::firrtl::IntType::get(prim->context, false, bitWidth);
}

inline circt::firrtl::IntType SInt(uint32_t bitWidth) {
  return circt::firrtl::IntType::get(prim->context, true, bitWidth);
}

inline BundleType Bundle(ArrayRef<BundleType::BundleElement> elements) {
  return BundleType::get(elements, prim->context);
}

// can be used to build any kind of expression

/*
  Value a, b, c, d;
  ...
  Value e = (  (a & b) | c + d  ).build(primBuilder);

  auto x = ...;

  // Problem: x can be any expression-ptr and access operators cannot be defined
  // outside the class they operate on.
  auto isValid = x["bits"]["valid"];


 */

class Expression {
public:
  enum Operation {
    OP_AND, OP_OR, OP_NEG, OP_ADD, OP_SUB, OP_GT, OP_GEQ,
    OP_LT, OP_LEQ, OP_EQ, OP_NEQ
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

inline std::shared_ptr<Expression> UInt(int64_t value, uint32_t bitWidth) {
  Value val = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    UInt(bitWidth),
    ::llvm::APInt(bitWidth, value)
  ).getResult();

  return std::make_shared<ValueExpression>(val);
}

inline std::shared_ptr<Expression> SInt(int64_t value, uint32_t bitWidth) {
  Value val = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    SInt(bitWidth),
    ::llvm::APInt(bitWidth, value)
  ).getResult();

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

class FieldExpression : public Expression {
  std::shared_ptr<Expression> operand;
  std::string fieldName;
public:
  FieldExpression(std::shared_ptr<Expression> operand, const std::string& fieldName):
    operand(operand), fieldName(fieldName) {}

  Value build() const override {
    return prim->builder.create<SubfieldOp>(
      prim->builder.getUnknownLoc(),
      operand->build(),
      fieldName
    ).getResult();
  }
};

inline std::shared_ptr<FieldExpression> field(std::shared_ptr<Expression> of, const std::string& fieldName) {
  return std::make_shared<FieldExpression>(of, fieldName);
}

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

// Idea: We create an automatic promoting of std::shared_ptr such that we can then define
// all kinds of custom operators for our needs.
class ExpressionPointer {
  std::shared_ptr<Expression> ptr;  
public:
  ExpressionPointer(std::shared_ptr<Expression> ptr): ptr(ptr) {}
  // back conversion
  operator std::shared_ptr<Expression>() { return ptr; }

  // define your custom operators here
  std::shared_ptr<FieldExpression> operator[](const std::string& fieldName) const {
    return std::make_shared<FieldExpression>(ptr, fieldName);
  }
};

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


// ::circt::hw::PortInfo doesn't really fit into the design of primitives
struct Port {
  enum Direction { Input, Output };

  Direction direction;
  std::string name;
  Type type;
};

template <class ConcreteModule>
class Module {
protected:
  // The template is a trick to enforce that modOp exists per concrete module class.
  static FModuleOp modOp;

  std::vector<Port> ports;
  std::unordered_map<std::string, size_t> portIndices;
  //std::unordered_map<std::string, size_t> inPortIndices;
  //std::unordered_map<std::string, size_t> outPortIndices;

  InstanceOp instOp;

  void declareOnce() {
    // check if it has already been declared
    if (modOp)
      return;

    // if not, declare
    size_t portIndex = 0;
    for (const Port& port : ports)
      portIndices[port.name] = portIndex++;

    std::vector<PortInfo> portInfos;
    for (const Port& port : ports)
      portInfos.emplace_back(
        prim->builder.getStringAttr(port.name),
        port.type,
        port.direction == Port::Direction::Input ? Direction::In : Direction::Out,
        prim->builder.getStringAttr("TestModule")
      );

    Block *lastInsertion = prim->builder.getInsertionBlock();
    prim->builder.setInsertionPointToEnd(prim->circuitOp.getBodyBlock());

    modOp = prim->builder.create<FModuleOp>(
      prim->builder.getUnknownLoc(),
      prim->builder.getStringAttr("TestModule"),
      portInfos
    );

    prim->builder.setInsertionPointToEnd(modOp.getBodyBlock());
    ConcreteModule::body();
    prim->builder.setInsertionPointToEnd(lastInsertion);
  }

  void instantiate() {
    instOp = prim->builder.create<InstanceOp>(
      prim->builder.getUnknownLoc(),
      modOp,
      prim->builder.getStringAttr("TestModuleInstance")
    );
  }

protected:
  Module(std::initializer_list<Port> ports):
    ports(ports) {
    // This constructor is used for declaration and instantiation at the same time.
    // Declaration can happen at most once.
    declareOnce();
    instantiate();
  }

  // only valid at the runtime of ConcreteModule::body()
  static void getArgument(const std::string& name);

  virtual ~Module() {}
public:


  //Value operator()(const std::string& name) {
  //  return instOp.getResults()[inPortIndices.at(name)];
  //}
};

class TestModule : public Module<TestModule> {
public:
  TestModule():
    Module<TestModule>({
      Port{.direction = Port::Direction::Input, .name = "x", .type = UInt(6)},
      Port{.direction = Port::Direction::Output, .name = "y", .type = UInt(7)},
      Port{.direction = Port::Direction::Output, .name = "z", .type = UInt(8)}
    }) {

  }

  static void body() {
    using namespace operators;

    auto valid = UInt(1, 1);
    auto ready = UInt(1, 1);
    auto count = UInt(123, 16);

    Value canEnqueue = (valid & ready & (count < lift(constant(8, 16))))->build();

    auto lower = bits(count, 3, 0);
    auto someField = field(lower, "someField");

    auto reg = Reg(UInt(32));
    reg << count + count;
    reg << count + count;
  }
};
 
}