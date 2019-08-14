package ml.microsoft.onnxruntime;

public class Env {
  static {
    System.loadLibrary("onnxruntime_jni");
  }
  public Env() {
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
