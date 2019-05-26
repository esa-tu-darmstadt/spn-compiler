package spn_compiler.server.grpc.spncserver

object SPNCompilerGrpc {
  val METHOD_SPNCOMPILE_JSON: _root_.io.grpc.MethodDescriptor[_root_.spn_compiler.server.grpc.spncserver.CompileRequest, _root_.spn_compiler.server.grpc.spncserver.CompileReply] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("parser.SPNCompiler", "SPNCompile_JSON"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[_root_.spn_compiler.server.grpc.spncserver.CompileRequest])
      .setResponseMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[_root_.spn_compiler.server.grpc.spncserver.CompileReply])
      .build()
  
  val METHOD_SPNCOMPILE_TEXT: _root_.io.grpc.MethodDescriptor[_root_.spn_compiler.server.grpc.spncserver.CompileRequest, _root_.spn_compiler.server.grpc.spncserver.CompileReply] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("parser.SPNCompiler", "SPNCompile_Text"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[_root_.spn_compiler.server.grpc.spncserver.CompileRequest])
      .setResponseMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[_root_.spn_compiler.server.grpc.spncserver.CompileReply])
      .build()
  
  val SERVICE: _root_.io.grpc.ServiceDescriptor =
    _root_.io.grpc.ServiceDescriptor.newBuilder("parser.SPNCompiler")
      .setSchemaDescriptor(new _root_.scalapb.grpc.ConcreteProtoFileDescriptorSupplier(spn_compiler.server.grpc.spncserver.SpncserverProto.javaDescriptor))
      .addMethod(METHOD_SPNCOMPILE_JSON)
      .addMethod(METHOD_SPNCOMPILE_TEXT)
      .build()
  
  /** The SPN compiler service definition.
    */
  trait SPNCompiler extends _root_.scalapb.grpc.AbstractService {
    override def serviceCompanion = SPNCompiler
    /** Compiles a valid SPN, or fails otherwise.
      */
    def sPNCompileJSON(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest): scala.concurrent.Future[_root_.spn_compiler.server.grpc.spncserver.CompileReply]
    def sPNCompileText(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest): scala.concurrent.Future[_root_.spn_compiler.server.grpc.spncserver.CompileReply]
  }
  
  object SPNCompiler extends _root_.scalapb.grpc.ServiceCompanion[SPNCompiler] {
    implicit def serviceCompanion: _root_.scalapb.grpc.ServiceCompanion[SPNCompiler] = this
    def javaDescriptor: _root_.com.google.protobuf.Descriptors.ServiceDescriptor = spn_compiler.server.grpc.spncserver.SpncserverProto.javaDescriptor.getServices().get(0)
    def scalaDescriptor: _root_.scalapb.descriptors.ServiceDescriptor = SpncserverProto.scalaDescriptor.services(0)
  }
  
  /** The SPN compiler service definition.
    */
  trait SPNCompilerBlockingClient {
    def serviceCompanion = SPNCompiler
    /** Compiles a valid SPN, or fails otherwise.
      */
    def sPNCompileJSON(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest): _root_.spn_compiler.server.grpc.spncserver.CompileReply
    def sPNCompileText(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest): _root_.spn_compiler.server.grpc.spncserver.CompileReply
  }
  
  class SPNCompilerBlockingStub(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions = _root_.io.grpc.CallOptions.DEFAULT) extends _root_.io.grpc.stub.AbstractStub[SPNCompilerBlockingStub](channel, options) with SPNCompilerBlockingClient {
    /** Compiles a valid SPN, or fails otherwise.
      */
    override def sPNCompileJSON(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest): _root_.spn_compiler.server.grpc.spncserver.CompileReply = {
      _root_.scalapb.grpc.ClientCalls.blockingUnaryCall(channel, METHOD_SPNCOMPILE_JSON, options, request)
    }
    
    override def sPNCompileText(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest): _root_.spn_compiler.server.grpc.spncserver.CompileReply = {
      _root_.scalapb.grpc.ClientCalls.blockingUnaryCall(channel, METHOD_SPNCOMPILE_TEXT, options, request)
    }
    
    override def build(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions): SPNCompilerBlockingStub = new SPNCompilerBlockingStub(channel, options)
  }
  
  class SPNCompilerStub(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions = _root_.io.grpc.CallOptions.DEFAULT) extends _root_.io.grpc.stub.AbstractStub[SPNCompilerStub](channel, options) with SPNCompiler {
    /** Compiles a valid SPN, or fails otherwise.
      */
    override def sPNCompileJSON(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest): scala.concurrent.Future[_root_.spn_compiler.server.grpc.spncserver.CompileReply] = {
      _root_.scalapb.grpc.ClientCalls.asyncUnaryCall(channel, METHOD_SPNCOMPILE_JSON, options, request)
    }
    
    override def sPNCompileText(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest): scala.concurrent.Future[_root_.spn_compiler.server.grpc.spncserver.CompileReply] = {
      _root_.scalapb.grpc.ClientCalls.asyncUnaryCall(channel, METHOD_SPNCOMPILE_TEXT, options, request)
    }
    
    override def build(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions): SPNCompilerStub = new SPNCompilerStub(channel, options)
  }
  
  def bindService(serviceImpl: SPNCompiler, executionContext: scala.concurrent.ExecutionContext): _root_.io.grpc.ServerServiceDefinition =
    _root_.io.grpc.ServerServiceDefinition.builder(SERVICE)
    .addMethod(
      METHOD_SPNCOMPILE_JSON,
      _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[_root_.spn_compiler.server.grpc.spncserver.CompileRequest, _root_.spn_compiler.server.grpc.spncserver.CompileReply] {
        override def invoke(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest, observer: _root_.io.grpc.stub.StreamObserver[_root_.spn_compiler.server.grpc.spncserver.CompileReply]): Unit =
          serviceImpl.sPNCompileJSON(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
            executionContext)
      }))
    .addMethod(
      METHOD_SPNCOMPILE_TEXT,
      _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[_root_.spn_compiler.server.grpc.spncserver.CompileRequest, _root_.spn_compiler.server.grpc.spncserver.CompileReply] {
        override def invoke(request: _root_.spn_compiler.server.grpc.spncserver.CompileRequest, observer: _root_.io.grpc.stub.StreamObserver[_root_.spn_compiler.server.grpc.spncserver.CompileReply]): Unit =
          serviceImpl.sPNCompileText(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
            executionContext)
      }))
    .build()
  
  def blockingStub(channel: _root_.io.grpc.Channel): SPNCompilerBlockingStub = new SPNCompilerBlockingStub(channel)
  
  def stub(channel: _root_.io.grpc.Channel): SPNCompilerStub = new SPNCompilerStub(channel)
  
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.ServiceDescriptor = spn_compiler.server.grpc.spncserver.SpncserverProto.javaDescriptor.getServices().get(0)
  
}