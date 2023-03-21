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

  void setClock(Value value) {
    clk = value;
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

// This class ensures that everything lands on the heap and that the references
// therefore not point to destructed stack objects.
class ExpressionWrapper {
  std::shared_ptr<Expression> ptr;
public:
  ExpressionWrapper() {}
  ExpressionWrapper(std::shared_ptr<Expression> ptr): ptr(ptr) {}

  template <class T, class ...Args>
  static ExpressionWrapper make(Args&&...args) {
    return ExpressionWrapper(std::make_shared<T>(args...));
  }

  Value build() const {
    return ptr->build();
  }

  // We can now implement all the operators we couldn't before.

  ExpressionWrapper operator&(ExpressionWrapper b) const;
  ExpressionWrapper operator|(ExpressionWrapper b) const;
  ExpressionWrapper operator+(ExpressionWrapper b) const;
  ExpressionWrapper operator-(ExpressionWrapper b) const;
  ExpressionWrapper operator>(ExpressionWrapper b) const;
  ExpressionWrapper operator>=(ExpressionWrapper b) const;
  ExpressionWrapper operator<(ExpressionWrapper b) const;
  ExpressionWrapper operator<=(ExpressionWrapper b) const;
  ExpressionWrapper operator==(ExpressionWrapper b) const;
  ExpressionWrapper operator!=(ExpressionWrapper b) const;

  // field
  ExpressionWrapper operator()(const std::string& fieldName) const;
  // bits
  ExpressionWrapper operator()(size_t hi, size_t lo) const;
  // single bit
  ExpressionWrapper operator()(size_t bitIndex) const;
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

inline ExpressionWrapper lift(Value val) {
  return ExpressionWrapper::make<ValueExpression>(val);
}

inline ExpressionWrapper UInt(int64_t value, uint32_t bitWidth) {
  Value val = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    UInt(bitWidth),
    ::llvm::APInt(bitWidth, value)
  ).getResult();

  return ExpressionWrapper::make<ValueExpression>(val);
}

inline ExpressionWrapper SInt(int64_t value, uint32_t bitWidth) {
  Value val = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    SInt(bitWidth),
    ::llvm::APInt(bitWidth, value)
  ).getResult();

  return ExpressionWrapper::make<ValueExpression>(val);
}

class UnaryExpression : public Expression {
  ExpressionWrapper operand;
  Operation operation;
public:
  UnaryExpression(UnaryExpression&) = delete;
  UnaryExpression(const UnaryExpression&) = delete;
  UnaryExpression(UnaryExpression&&) = delete;

  UnaryExpression(ExpressionWrapper operand, Operation operation):
    operand(operand),
    operation(operation) {}

  Value build() const override {
    Value input = operand.build();
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
  ExpressionWrapper lhs;
  ExpressionWrapper rhs;
  Operation operation;
public:
  //BinaryExpression(BinaryExpression&) = delete;
  //BinaryExpression(const BinaryExpression&) = delete;
  //BinaryExpression(BinaryExpression&&) = delete;

  BinaryExpression(ExpressionWrapper lhs, ExpressionWrapper rhs, Operation operation):
    lhs(lhs),
    rhs(rhs),
    operation(operation) {}

  Value build() const override {
    Value leftInput = lhs.build();
    Value rightInput = rhs.build();
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
  ExpressionWrapper operand;
  uint32_t hi, lo;
public:
  BitsExpression(ExpressionWrapper operand, uint32_t hi, uint32_t lo):
    operand(operand), hi(hi), lo(lo) {}

  Value build() const override {
    return prim->builder.create<BitsPrimOp>(
      prim->builder.getUnknownLoc(),
      operand.build(),
      prim->builder.getI32IntegerAttr(hi),
      prim->builder.getI32IntegerAttr(lo)
    ).getResult();
  }
};

inline ExpressionWrapper bits(ExpressionWrapper of, uint32_t hi, uint32_t lo) {
  return ExpressionWrapper::make<BitsExpression>(of, hi, lo);
}

class FieldExpression : public Expression {
  ExpressionWrapper operand;
  std::string fieldName;
public:
  FieldExpression(ExpressionWrapper operand, const std::string& fieldName):
    operand(operand), fieldName(fieldName) {}

  Value build() const override {
    return prim->builder.create<SubfieldOp>(
      prim->builder.getUnknownLoc(),
      operand.build(),
      fieldName
    ).getResult();
  }
};

inline ExpressionWrapper field(ExpressionWrapper of, const std::string& fieldName) {
  return ExpressionWrapper::make<FieldExpression>(of, fieldName);
}

class Statement {
};

class Reg : public Statement {
  RegOp regOp;
public:
  Reg(Type type) {
    regOp = prim->builder.create<RegOp>(
      prim->builder.getUnknownLoc(),
      type,
      prim->getClock()
    );
  }

  operator Value() {
    return regOp.getResult();
  }

  operator ExpressionWrapper() {
    return lift(*this);
  }

  void operator<<(ExpressionWrapper what) {
    Value input = what.build();
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

  std::string moduleName;
  std::vector<Port> ports;
  bool hasClock;
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
        port.direction == Port::Direction::Input ? Direction::In : Direction::Out
      );

    Block *lastInsertion = prim->builder.getInsertionBlock();
    prim->builder.setInsertionPointToEnd(prim->circuitOp.getBodyBlock());

    modOp = prim->builder.create<FModuleOp>(
      prim->builder.getUnknownLoc(),
      prim->builder.getStringAttr(moduleName),
      portInfos
    );

    Block *bodyBlock = modOp.getBodyBlock();
    prim->builder.setInsertionPointToEnd(bodyBlock);

    if (hasClock)
      prim->setClock(bodyBlock->getArguments()[0]);

    body();
    prim->builder.setInsertionPointToEnd(lastInsertion);
  }

  void instantiate() {
    instOp = prim->builder.create<InstanceOp>(
      prim->builder.getUnknownLoc(),
      modOp,
      prim->builder.getStringAttr("TestModuleInstance")
    );
  }

  void body() {
    static_cast<ConcreteModule *>(this)->body();
  }
protected:
  Module(const std::string& moduleName, std::initializer_list<Port> ports, bool hasClock = true):
    moduleName(moduleName),
    ports(ports),
    hasClock(hasClock) {
    // This constructor is used for declaration and instantiation at the same time.

    // By default every module has the clk as its first argument.
    if (hasClock) {
      this->ports.insert(
        this->ports.begin(),
        Port{
          .direction = Port::Direction::Input,
          .name = "clk",
          .type = ClockType::get(prim->context)
        }
      );
    }
    
    // Declaration can happen at most once.
    declareOnce();
    //instantiate();
  }

  Value getArgument(const std::string& name) {
    size_t index = portIndices.at(name);
    return modOp.getBodyBlock()->getArgument(index);
  }
public:
  virtual ~Module() {}
};

template <class T>
FModuleOp Module<T>::modOp;

class TestModule : public Module<TestModule> {
public:
  TestModule():
    Module<TestModule>(
      "MyCircuit",
      {
        Port{.direction = Port::Direction::Input, .name = "x", .type = UInt(6)},
        Port{.direction = Port::Direction::Output, .name = "y", .type = UInt(6)}
      }) {

  }

  void body() {
    /*
    auto valid = UInt(1, 1);
    auto ready = UInt(1, 1);
    auto count = UInt(123, 16);

    Value canEnqueue = (valid & ready & (count < lift(constant(8, 16))))->build();

    auto lower = bits(count, 3, 0);
    auto someField = field(lower, "someField");

    auto reg = Reg(UInt(32));
    reg << count + count;
    reg << count + count;
     */

    auto x = lift(getArgument("x"));
    auto c = lift(constant(3, 6));
    auto reg = Reg(UInt(7));
    reg << x + c;
    auto reg2 = Reg(UInt(2));
    reg2 << c(1, 0);
  }
};
 
}