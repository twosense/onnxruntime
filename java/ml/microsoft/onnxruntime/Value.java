package ml.microsoft.onnxruntime;

public class Value {
  static {
    System.loadLibrary("onnxruntime_jni");
  }
  private Value() {
  }
  @Override
  protected void finalize() throws Throwable {
    dispose();
    super.finalize();
  }
  private long nativeHandle;
  public native void dispose();
  public native static Value createFloatTensorFromData(AllocatorInfo allocatorInfo, float[] data, long[] shape);
}
