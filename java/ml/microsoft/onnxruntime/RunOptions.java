package ml.microsoft.onnxruntime;

public class RunOptions {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  public RunOptions() {
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
}
