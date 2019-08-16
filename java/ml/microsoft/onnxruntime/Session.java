package ml.microsoft.onnxruntime;

public class Session {
  static {
    System.loadLibrary("onnxruntime_jni");
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
  public native Value[] run(RunOptions runOptions, String[] input_names, Value[] input_values, String[] output_names);
}
