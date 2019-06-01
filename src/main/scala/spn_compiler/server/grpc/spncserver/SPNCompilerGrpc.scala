package spn_compiler.server.grpc.spncserver

object SPNCompilerGrpc {
  val METHOD_SPNCOMPILE_JSON: _root_.io.grpc.MethodDescriptor[spn_compiler.server.grpc.spncserver.CompileRequest, spn_compiler.server.grpc.spncserver.CompileReply] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("spn_compiler.server.grpc.SPNCompiler", "SPNCompile_JSON"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(new scalapb.grpc.Marshaller(spn_compiler.server.grpc.spncserver.CompileRequest))
      .setResponseMarshaller(new scalapb.grpc.Marshaller(spn_compiler.server.grpc.spncserver.CompileReply))
      .build()
  
  val METHOD_SPNCOMPILE_TEXT: _root_.io.grpc.MethodDescriptor[spn_compiler.server.grpc.spncserver.CompileRequest, spn_compiler.server.grpc.spncserver.CompileReply] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("spn_compiler.server.grpc.SPNCompiler", "SPNCompile_Text"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(new scalapb.grpc.Marshaller(spn_compiler.server.grpc.spncserver.CompileRequest))
      .setResponseMarshaller(new scalapb.grpc.Marshaller(spn_compiler.server.grpc.spncserver.CompileReply))
      .build()
  
  val SERVICE: _root_.io.grpc.ServiceDescriptor =
    _root_.io.grpc.ServiceDescriptor.newBuilder("spn_compiler.server.grpc.SPNCompiler")
      .setSchemaDescriptor(new _root_.scalapb.grpc.ConcreteProtoFileDescriptorSupplier(spn_compiler.server.grpc.spncserver.SpncserverProto.javaDescriptor))
      .addMethod(METHOD_SPNCOMPILE_JSON)
      .addMethod(METHOD_SPNCOMPILE_TEXT)
      .build()
  
  trait SPNCompiler extends _root_.scalapb.grpc.AbstractService {
    override def serviceCompanion = SPNCompiler
    def sPNCompileJSON(request: spn_compiler.server.grpc.spncserver.CompileRequest): scala.concurrent.Future[spn_compiler.server.grpc.spncserver.CompileReply]
    def sPNCompileText(request: spn_compiler.server.grpc.spncserver.CompileRequest): scala.concurrent.Future[spn_compiler.server.grpc.spncserver.CompileReply]
  }
  
  object SPNCompiler extends _root_.scalapb.grpc.ServiceCompanion[SPNCompiler] {
    implicit def serviceCompanion: _root_.scalapb.grpc.ServiceCompanion[SPNCompiler] = this
    def javaDescriptor: _root_.com.google.protobuf.Descriptors.ServiceDescriptor = spn_compiler.server.grpc.spncserver.SpncserverProto.javaDescriptor.getServices().get(0)
  }
  
  trait SPNCompilerBlockingClient {
    def serviceCompanion = SPNCompiler
    def sPNCompileJSON(request: spn_compiler.server.grpc.spncserver.CompileRequest): spn_compiler.server.grpc.spncserver.CompileReply
    def sPNCompileText(request: spn_compiler.server.grpc.spncserver.CompileRequest): spn_compiler.server.grpc.spncserver.CompileReply
  }
  
  class SPNCompilerBlockingStub(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions = _root_.io.grpc.CallOptions.DEFAULT) extends _root_.io.grpc.stub.AbstractStub[SPNCompilerBlockingStub](channel, options) with SPNCompilerBlockingClient {
    override def sPNCompileJSON(request: spn_compiler.server.grpc.spncserver.CompileRequest): spn_compiler.server.grpc.spncserver.CompileReply = {
      _root_.io.grpc.stub.ClientCalls.blockingUnaryCall(channel.newCall(METHOD_SPNCOMPILE_JSON, options), request)
    }
    
    override def sPNCompileText(request: spn_compiler.server.grpc.spncserver.CompileRequest): spn_compiler.server.grpc.spncserver.CompileReply = {
      _root_.io.grpc.stub.ClientCalls.blockingUnaryCall(channel.newCall(METHOD_SPNCOMPILE_TEXT, options), request)
    }
    
    override def build(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions): SPNCompilerBlockingStub = new SPNCompilerBlockingStub(channel, options)
  }
  
  class SPNCompilerStub(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions = _root_.io.grpc.CallOptions.DEFAULT) extends _root_.io.grpc.stub.AbstractStub[SPNCompilerStub](channel, options) with SPNCompiler {
    override def sPNCompileJSON(request: spn_compiler.server.grpc.spncserver.CompileRequest): scala.concurrent.Future[spn_compiler.server.grpc.spncserver.CompileReply] = {
      scalapb.grpc.Grpc.guavaFuture2ScalaFuture(_root_.io.grpc.stub.ClientCalls.futureUnaryCall(channel.newCall(METHOD_SPNCOMPILE_JSON, options), request))
    }
    
    override def sPNCompileText(request: spn_compiler.server.grpc.spncserver.CompileRequest): scala.concurrent.Future[spn_compiler.server.grpc.spncserver.CompileReply] = {
      scalapb.grpc.Grpc.guavaFuture2ScalaFuture(_root_.io.grpc.stub.ClientCalls.futureUnaryCall(channel.newCall(METHOD_SPNCOMPILE_TEXT, options), request))
    }
    
    override def build(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions): SPNCompilerStub = new SPNCompilerStub(channel, options)
  }
  
  def bindService(serviceImpl: SPNCompiler, executionContext: scala.concurrent.ExecutionContext): _root_.io.grpc.ServerServiceDefinition =
    _root_.io.grpc.ServerServiceDefinition.builder(SERVICE)
    .addMethod(
      METHOD_SPNCOMPILE_JSON,
      _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[spn_compiler.server.grpc.spncserver.CompileRequest, spn_compiler.server.grpc.spncserver.CompileReply] {
        override def invoke(request: spn_compiler.server.grpc.spncserver.CompileRequest, observer: _root_.io.grpc.stub.StreamObserver[spn_compiler.server.grpc.spncserver.CompileReply]): Unit =
          serviceImpl.sPNCompileJSON(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
            executionContext)
      }))
    .addMethod(
      METHOD_SPNCOMPILE_TEXT,
      _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[spn_compiler.server.grpc.spncserver.CompileRequest, spn_compiler.server.grpc.spncserver.CompileReply] {
        override def invoke(request: spn_compiler.server.grpc.spncserver.CompileRequest, observer: _root_.io.grpc.stub.StreamObserver[spn_compiler.server.grpc.spncserver.CompileReply]): Unit =
          serviceImpl.sPNCompileText(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
            executionContext)
      }))
    .build()
  
  def blockingStub(channel: _root_.io.grpc.Channel): SPNCompilerBlockingStub = new SPNCompilerBlockingStub(channel)
  
  def stub(channel: _root_.io.grpc.Channel): SPNCompilerStub = new SPNCompilerStub(channel)
  
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.ServiceDescriptor = spn_compiler.server.grpc.spncserver.SpncserverProto.javaDescriptor.getServices().get(0)
  
}