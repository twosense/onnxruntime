package ml.microsoft.onnxruntime;

public class Session {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  public Session(Env env, String modelPath, SessionOptions sessionOptions) {
    initHandle(env, modelPath, sessionOptions);
  }
  @Override
  protected void finalize() throws Throwable {
    dispose();
    super.finalize();
  }
  private long nativeHandle;
  private native void initHandle(Env env, String modelPath, SessionOptions sessionOptions);
  public native void dispose();
  public native Value[] run(RunOptions runOptions, String[] input_names, Value[] input_values,
      String[] output_names);
  public native long getInputCount();
  public native long getOutputCount();
  public native String getInputName(long index, Allocator allocator);
  public native String getOutputName(long index, Allocator allocator);
  public native TypeInfo getInputTypeInfo(long index);
  public native TypeInfo getOutputTypeInfo(long index);
}
