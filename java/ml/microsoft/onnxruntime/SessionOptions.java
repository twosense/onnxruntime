package ml.microsoft.onnxruntime;

public class SessionOptions {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  public SessionOptions() {
    initHandle();
  }
  @Override
  protected void finalize() throws Throwable {
    dispose();
    super.finalize();
  }
  private long nativeHandle;
  private native void initHandle();
  public native void dispose();
  public native void appendNnapiExecutionProvider();
}
