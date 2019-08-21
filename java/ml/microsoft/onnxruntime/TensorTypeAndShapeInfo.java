package ml.microsoft.onnxruntime;

public class TensorTypeAndShapeInfo {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  private TensorTypeAndShapeInfo() {
  }
  private long nativeHandle;
  public native void dispose();
  public native long[] getShape();
  public native long getDimensionsCount();
  public native long getElementCount();
}
